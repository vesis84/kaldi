// nnetbin/nnet-train-mpe-sequential.cc

// Copyright 2011-2013  Brno University of Technology (author: Karel Vesely);  Arnab Ghoshal

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "tree/context-dep.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "decoder/faster-decoder.h"
#include "decoder/decodable-matrix.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

#include "nnet/nnet-trnopts.h"
#include "nnet/nnet-component.h"
#include "nnet/nnet-activation.h"
#include "nnet/nnet-nnet.h"
#include "nnet/nnet-pdf-prior.h"
#include "nnet/nnet-utils.h"
#include "base/timer.h"
#include "cudamatrix/cu-device.h"


namespace kaldi {
namespace nnet1 {

void LatticeAcousticRescore(const Matrix<BaseFloat> &log_like,
                            const TransitionModel &trans_model,
                            const std::vector<int32> &state_times,
                            Lattice *lat) {
  kaldi::uint64 props = lat->Properties(fst::kFstProperties, false);
  if (!(props & fst::kTopSorted))
    KALDI_ERR << "Input lattice must be topologically sorted.";

  KALDI_ASSERT(!state_times.empty());
  std::vector<std::vector<int32> > time_to_state(log_like.NumRows());
  for (size_t i = 0; i < state_times.size(); i++) {
    KALDI_ASSERT(state_times[i] >= 0);
    if (state_times[i] < log_like.NumRows())  // end state may be past this..
      time_to_state[state_times[i]].push_back(i);
    else
      KALDI_ASSERT(state_times[i] == log_like.NumRows()
                   && "There appears to be lattice/feature mismatch.");
  }

  for (int32 t = 0; t < log_like.NumRows(); t++) {
    for (size_t i = 0; i < time_to_state[t].size(); i++) {
      int32 state = time_to_state[t][i];
      for (fst::MutableArcIterator<Lattice> aiter(lat, state); !aiter.Done();
           aiter.Next()) {
        LatticeArc arc = aiter.Value();
        int32 trans_id = arc.ilabel;
        if (trans_id != 0) {  // Non-epsilon input label on arc
          int32 pdf_id = trans_model.TransitionIdToPdf(trans_id);
          arc.weight.SetValue2(-log_like(t, pdf_id) + arc.weight.Value2());
          aiter.SetValue(arc);
        }
      }
    }
  }
}

}  // namespace nnet1
}  // namespace kaldi


int main(int argc, char *argv[]) {
  using namespace kaldi;
  using namespace kaldi::nnet1;
  typedef kaldi::int32 int32;
  try {
    const char *usage =
        "Perform iteration of Neural Network MPE/sMBR training by stochastic "
        "gradient descent.\n"
        "The network weights are updated on each utterance.\n"
        "Usage:  nnet-train-mpe-sequential [options] <model-in> <transition-model-in> "
        "<feature-rspecifier> <den-lat-rspecifier> <ali-rspecifier> [<model-out>]\n"
        "e.g.: \n"
        " nnet-train-mpe-sequential nnet.init trans.mdl scp:train.scp scp:denlats.scp ark:train.ali "
        "nnet.iter1\n";

    ParseOptions po(usage);

    NnetTrainOptions trn_opts; trn_opts.learn_rate=0.00001;
    trn_opts.Register(&po);

    bool binary = true; 
    po.Register("binary", &binary, "Write output in binary mode");

    std::string feature_transform;
    po.Register("feature-transform", &feature_transform, 
                "Feature transform in Nnet format");
    std::string silence_phones_str;
    po.Register("silence-phones", &silence_phones_str, "Colon-separated list "
                "of integer id's of silence phones, e.g. 46:47");

    PdfPriorOptions prior_opts;
    prior_opts.Register(&po);

    bool one_silence_class = false;
    BaseFloat acoustic_scale = 1.0,
        lm_scale = 1.0,
        nce_scale = 1.0,
        nce_gradient_scale = 1.0;
    po.Register("acoustic-scale", &acoustic_scale,
                "Scaling factor for acoustic likelihoods");
    po.Register("lm-scale", &lm_scale,
                "Scaling factor for \"graph costs\" (including LM costs)");
    po.Register("nce-scale", &nce_scale,
                "Scale applied to both lattice-arc scores, only for NCE");
    po.Register("nce-gradient-scale", &nce_gradient_scale,
                "Scale applied to NCE gradient");
    po.Register("one-silence-class", &one_silence_class, "If true, newer "
                "behavior which will tend to reduce insertions.");
    kaldi::int32 max_frames = 6000; // Allow segments maximum of one minute by default
    po.Register("max-frames",&max_frames, "Maximum number of frames a segment can have to be processed");
    bool do_smbr = false;
    po.Register("do-smbr", &do_smbr, "Use state-level accuracies instead of "
                "phone accuracies.");

    std::string use_gpu="yes";
    po.Register("use-gpu", &use_gpu, "yes|no|optional, only has effect if compiled with CUDA");
     
    po.Read(argc, argv);

    if (po.NumArgs() != 6) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        transition_model_filename = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        den_lat_rspecifier = po.GetArg(4),
        ref_ali_rspecifier = po.GetArg(5);

    std::string target_model_filename;
    target_model_filename = po.GetArg(6);

    std::vector<int32> silence_phones;
    if (!kaldi::SplitStringToIntegers(silence_phones_str, ":", false,
                                      &silence_phones))
      KALDI_ERR << "Invalid silence-phones string " << silence_phones_str;
    kaldi::SortAndUniq(&silence_phones);
    if (silence_phones.empty())
      KALDI_LOG << "No silence phones specified.";

    // Select the GPU
#if HAVE_CUDA == 1
    CuDevice::Instantiate().SelectGpuId(use_gpu);
#endif

    Nnet nnet_transf;
    if (feature_transform != "") {
      nnet_transf.Read(feature_transform);
    }

    Nnet nnet;
    nnet.Read(model_filename);
    // using activations directly: remove softmax, if present
    if (nnet.GetComponent(nnet.NumComponents()-1).GetType() ==
        kaldi::nnet1::Component::kSoftmax) {
      KALDI_LOG << "Removing softmax from the nnet " << model_filename;
      nnet.RemoveComponent(nnet.NumComponents()-1);
    } else {
      KALDI_LOG << "The nnet was without softmax " << model_filename;
    }
    nnet.SetTrainOptions(trn_opts);

    // Read the class-frame-counts, compute priors
    PdfPrior log_prior(prior_opts);

    // Read transition model
    TransitionModel trans_model;
    ReadKaldiObject(transition_model_filename, &trans_model);

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessLatticeReader den_lat_reader(den_lat_rspecifier);
    RandomAccessInt32VectorReader ref_ali_reader(ref_ali_rspecifier);

    CuMatrix<BaseFloat> feats_transf, nnet_out, nnet_diff;
    Matrix<BaseFloat> nnet_out_h;

    Timer time;
    double time_now = 0;
    KALDI_LOG << "TRAINING STARTED";

    int32 num_done = 0, num_nce_lat = 0, num_no_den_lat = 0,
      num_other_error = 0;

    kaldi::int64 total_frames = 0;
    double total_frame_acc = 0.0, utt_frame_acc;
    kaldi::int64 total_frames_nce = 0;
    double total_lat_nce = 0.0, lat_nce;

    // do per-utterance processing
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();

      // 1) get feature matrix,
      const Matrix<BaseFloat> &mat = feature_reader.Value();
      if (mat.NumRows() > max_frames) {
        KALDI_WARN << "Skipping " << utt << ", too long utterance."
          << ", #frames " << mat.NumRows() << " exceeds " << max_frames << ".";
        num_other_error++;
        continue;
      }

      // get dims,
      int32 num_frames = mat.NumRows(),
          num_pdfs = nnet.OutputDim();

      // 2) get denominator lattice, preprocess,
      if (!den_lat_reader.HasKey(utt)) {
        KALDI_WARN << "Skipping " << utt << ", missing lattice.";
        num_no_den_lat++;
        continue;
      }
      Lattice den_lat = den_lat_reader.Value(utt);
      if (den_lat.Start() == -1) {
        KALDI_WARN << "Skipping " << utt << ", empty lattice.";
        num_other_error++;
        continue;
      }
      // remove acoustic scores,
      fst::ScaleLattice(fst::AcousticLatticeScale(0.0), &den_lat);
      // sort topologically if needed,
      kaldi::uint64 props = den_lat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&den_lat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }
      // get state times, ending time of lattice,
      vector<int32> state_times;
      int32 max_time = kaldi::LatticeStateTimes(den_lat, &state_times);
      KALDI_ASSERT(max_time == mat.NumRows());

      // 3) Forward pass through NN,
      // propagate through feature transform,
      nnet_transf.Feedforward(CuMatrix<BaseFloat>(mat), &feats_transf);
      // propagate through the nnet (assuming it's w/o softmax),
      nnet.Propagate(feats_transf, &nnet_out);
      // subtract the log_prior,
      if (prior_opts.class_frame_counts != "") {
        log_prior.SubtractOnLogpost(&nnet_out);
      }
      // move nn-output to host,
      nnet_out_h.Resize(num_frames, num_pdfs, kUndefined);
      nnet_out.CopyToMat(&nnet_out_h);
      // release the buffers we don't need anymore,
      feats_transf.Resize(0,0);
      nnet_out.Resize(0,0);

      // 4) Rescore den-lat. with the nn-output,
      LatticeAcousticRescore(nnet_out_h, trans_model, state_times, &den_lat);
      if (acoustic_scale != 1.0 || lm_scale != 1.0)
        fst::ScaleLattice(fst::LatticeScale(lm_scale, acoustic_scale), &den_lat);

      // 5) Evaluate objective function, get the gradient,
      kaldi::Posterior post; //< buffer for gradient,
      if (ref_ali_reader.HasKey(utt)) {
        // Objective function : MPE/sMBR,
 
        // read alignment, check the length,
        const std::vector<int32> &ref_ali = ref_ali_reader.Value(utt);
        KALDI_ASSERT(ref_ali.size() == num_frames);

        // objective function FW-BW,
        //  smbr : use state-level accuracies, i.e. sMBR estimation
        //  mpfe : use phone-level accuracies, i.e. MPFE (minimum phone frame error)
        utt_frame_acc = LatticeForwardBackwardMpeVariants(
            trans_model, silence_phones, den_lat, ref_ali, 
            (do_smbr?"smbr":"mpfe"), one_silence_class, &post);

        // logging,
        KALDI_VLOG(2) << num_done+1 << " " << utt << ", " 
                      << (do_smbr ? "smbr" : "mpfe")
                      << " accuracy " << utt_frame_acc/num_frames << ", "
                      << den_lat.NumStates() << " states, "
                      << fst::NumArcs(den_lat) << " arcs, "
                      << num_frames << " frames.";
        total_frame_acc += utt_frame_acc;
        total_frames += num_frames;
      } else {
        // Do NCE,
      
        // re-scale scores in lattice,
        if (nce_scale != 1.0)
          fst::ScaleLattice(fst::LatticeScale(nce_scale, nce_scale), &den_lat);

        // objective function,
        lat_nce = LatticeForwardBackwardNce(trans_model, den_lat, &post).Value();

        // scale the gradient (equivalent to scaling the learning rate),
        ScalePosterior(nce_gradient_scale, &post);
        
        // logging,
        KALDI_VLOG(2) << num_done+1 << " " << utt << ", "
                      << " NCE " << (lat_nce/num_frames) << ", "
                      << den_lat.NumStates() << " states, "
                      << fst::NumArcs(den_lat) << " arcs, "
                      << num_frames << " frames.";
        total_lat_nce += lat_nce;
        total_frames_nce += num_frames;
        num_nce_lat++;
      }

      // 6) convert the Posterior to CuMatrix,
      PosteriorToMatrixMapped(post, trans_model, &nnet_diff);
      nnet_diff.Scale(-1.0); // need to flip the sign of derivative,
      KALDI_VLOG(3) << MomentStatistics(nnet_diff);

      // 7) backpropagate through the nnet,
      nnet.Backpropagate(nnet_diff, NULL);
      nnet_diff.Resize(0,0); // release GPU memory,

      // logging,
      num_done++;
      if (num_done % 100 == 0) {
        time_now = time.Elapsed();
        KALDI_VLOG(1) << "After " << num_done << " utterances: time elapsed = "
                      << time_now/60 << " min; processed " << total_frames/time_now
                      << " frames per second."; 
#if HAVE_CUDA==1        
        // check the GPU is not overheated
        CuDevice::Instantiate().CheckGpuHealth();
#endif
      }
    }

    // append softmax, write the model,
    KALDI_LOG << "Appending the softmax " << target_model_filename;
    nnet.AppendComponent(new Softmax(nnet.OutputDim(),nnet.OutputDim()));
    nnet.Write(target_model_filename, binary);

    time_now = time.Elapsed();
    KALDI_LOG << "TRAINING FINISHED; "
              << "Time taken = " << time_now/60 << " min; processed "
              << (total_frames/time_now) << " frames per second.";

    KALDI_LOG << "Done " << num_done << " files, "
              << num_nce_lat << " with NCE objective, "
              << num_no_den_lat << " with no lattices, "
              << num_other_error << " with other errors.";

    KALDI_LOG << "Overall average " << (do_smbr?"smbr":"mpfe") 
              << " frame-accuracy is " << (total_frame_acc/total_frames) 
              << " over " << total_frames << " frames.";
    KALDI_LOG << "Overall average Negative Conditional Entropy is "
              << (total_lat_nce/total_frames_nce) << " over " << total_frames_nce
              << " frames.";


#if HAVE_CUDA == 1
    CuDevice::Instantiate().PrintProfile();
#endif

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
