// nnet/nnet-parametric-relu.h

// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)

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


#ifndef KALDI_NNET_NNET_PARAMETRIC_RELU_H_
#define KALDI_NNET_NNET_PARAMETRIC_RELU_H_

#include <string>

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

namespace kaldi {
namespace nnet1 {

class ParametricRelu : public UpdatableComponent {
 public:
  ParametricRelu(int32 dim_in, int32 dim_out):
    UpdatableComponent(dim_in, dim_out),
    max_alpha_(dim_out), min_beta_(dim_out),
    max_alpha_corr_(dim_out), min_beta_corr_(dim_out),
    clip_gradient_(0.0)
  { }
  ~ParametricRelu()
  { }

  Component* Copy() const { return new ParametricRelu(*this); }
  ComponentType GetType() const { return kParametricRelu; }

  void InitData(std::istream &is) {
    // define options
    float alpha = 1, beta = 0.25,
    max_learn_rate_coef = 0.000005,
    min_learn_rate_coef = 0.00005,
    max_fix_coef = 0.0,
    min_fix_coef = 0.0,
    clip_gradient = 0.0;

    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
      /**/ if (token == "<Alpha>")  ReadBasicType(is, false, &alpha);
      else if (token == "<Beta>")  ReadBasicType(is, false, &beta);
      else if (token == "<MaxLearnRateCoef>")  ReadBasicType(is, false, &max_learn_rate_coef);
      else if (token == "<MinLearnRateCoef>")  ReadBasicType(is, false, &min_learn_rate_coef);
      else if (token == "<FixMaxCoef>")  ReadBasicType(is, false, &max_fix_coef);
      else if (token == "<FixMinCoef>")   ReadBasicType(is, false, &min_fix_coef);
      else if (token == "<FixMinCoef>")  ReadBasicType(is, false, &min_fix_coef);
      else if (token == "<ClipGradient>")  ReadBasicType(is, false, &clip_gradient);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (Alpha|Beta|MaxLearnRateCoef|MinLearnRateCoef|MaxFixCoef|MinFixCoef|ClipGradient)";
    }

    //
    // Initialize trainable parameters,
    //

    Vector<BaseFloat> veca(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input alpha
      veca(i) = alpha;
    }
    max_alpha_ = veca;
    //
    Vector<BaseFloat> vecb(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input beta
      vecb(i) = beta;
    }
    min_beta_ = vecb;
    //
    alpha_ = alpha;
    beta_ = beta;
    max_learn_rate_coef_ = max_learn_rate_coef;
    min_learn_rate_coef_ = min_learn_rate_coef;
    max_fix_coef_ = max_fix_coef;
    min_fix_coef_ = min_fix_coef;
    clip_gradient_ = clip_gradient;
    KALDI_ASSERT(clip_gradient_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
        case 'A': ExpectToken(is, binary, "<Alpha>");
          ReadBasicType(is, binary, &alpha_);
          break;
        case 'B': ExpectToken(is, binary, "<Beta>");
          ReadBasicType(is, binary, &beta_);
          break;
        case 'M': ExpectToken(is, binary, "<MaxLearnRateCoef>");
          ReadBasicType(is, binary, &max_learn_rate_coef_);
          ExpectToken(is, binary, "<MinLearnRateCoef>");
          ReadBasicType(is, binary, &min_learn_rate_coef_);
          break;
        case 'F': ExpectToken(is, binary, "<FixMaxCoef>");
          ReadBasicType(is, binary, &max_fix_coef_);
          ExpectToken(is, binary, "<FixMinCoef>");
          ReadBasicType(is, binary, &min_fix_coef_);
          break;
         case 'C': ExpectToken(is, binary, "<ClipGradient>");
          ReadBasicType(is, binary, &clip_gradient_);
          break;
        default:
          std::string token;
          ReadToken(is, false, &token);
          KALDI_ERR << "Unknown token: " << token;
      }
    }
    // ParametricRelu scaling parameters
    max_alpha_.Read(is, binary);
    min_beta_.Read(is, binary);
    KALDI_ASSERT(max_alpha_.Dim() == output_dim_);
    KALDI_ASSERT(min_beta_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<Alpha>");
    WriteBasicType(os, binary, alpha_);
    WriteToken(os, binary, "<Beta>");
    WriteBasicType(os, binary, beta_);
    WriteToken(os, binary, "<MaxLearnRateCoef>");
    WriteBasicType(os, binary, max_learn_rate_coef_);
    WriteToken(os, binary, "<MinLearnRateCoef>");
    WriteBasicType(os, binary, min_learn_rate_coef_);
    WriteToken(os, binary, "<FixMaxCoef>");
    WriteBasicType(os, binary, max_fix_coef_);
    WriteToken(os, binary, "<FixMinCoef>");
    WriteBasicType(os, binary, min_fix_coef_);
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);

    // ParametricRelu scaling parameters

    if (!binary) os << "\n";
     max_alpha_.Write(os, binary);
     min_beta_.Write(os, binary);
  }

  int32 NumParams() const {
    return max_alpha_.Dim() + min_beta_.Dim();
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
    int32 max_alpha_num_elem = max_alpha_.Dim();
    int32 min_beta_num_elem = min_beta_.Dim();
    gradient->Range(0, max_alpha_num_elem).CopyFromVec(Vector<BaseFloat>(max_alpha_corr_));
    gradient->Range(max_alpha_num_elem, min_beta_num_elem).CopyFromVec(Vector<BaseFloat>(min_beta_corr_));
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
    int32 max_alpha_num_elem = max_alpha_.Dim();
    int32 min_beta_num_elem = min_beta_.Dim();
    params->Range(0, max_alpha_num_elem).CopyFromVec(Vector<BaseFloat>(max_alpha_));
    params->Range(max_alpha_num_elem, min_beta_num_elem).CopyFromVec(Vector<BaseFloat>(min_beta_));
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
    int32 max_alpha_num_elem = max_alpha_.Dim();
    int32 min_beta_num_elem = min_beta_.Dim();
    max_alpha_.CopyFromVec(params.Range(0, max_alpha_num_elem));
    min_beta_.CopyFromVec(params.Range(max_alpha_num_elem, min_beta_num_elem));
  }

  std::string Info() const {
    return std::string("\n  max_alpha") +
      MomentStatistics(max_alpha_) +
      ", max-lr-coef " + ToString(max_learn_rate_coef_) +
      "\n  min_beta" + MomentStatistics(min_beta_) +
      ", min-lr-coef " + ToString(min_learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  max_alpha_grad") +
      MomentStatistics(max_alpha_corr_) +
      ", max-lr-coef " + ToString(max_learn_rate_coef_) +
      "\n  min_beta_grad" + MomentStatistics(min_beta_corr_) +
      ", min-lr-coef " + ToString(min_learn_rate_coef_);
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
    out->ParametricRelu(in, max_alpha_, min_beta_);
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by activations (alpha, beta)
    // ((dE/da)*w) === out_diff, f(y) == out,
    //  (out > 0 ) ? out_diff * alpha : out_diff * beta
    in_diff->DiffParametricRelu(out, out_diff, max_alpha_, min_beta_);
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
    const BaseFloat alr = opts_.learn_rate * max_learn_rate_coef_;
    const BaseFloat blr = opts_.learn_rate * min_learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    // compute gradient (incl. momentum)
    if (!max_fix_coef_) {  // the alpha parameter is learnable
       input_max_.Resize(input.NumRows(), input.NumCols());
       input_max_.CopyFromMat(input);
       input_max_.ApplyFloor(0.0);
       max_alpha_corr_.AddRowSumMat(1.0, input_max_, mmt);
       max_alpha_.AddVec(-alr, max_alpha_corr_);
    }
    if (!min_fix_coef_) {  // the beta parameter is learnable
       input_min_.Resize(input.NumRows(), input.NumCols());
       input_min_.CopyFromMat(input);
       input_min_.ApplyCeiling(0.0);
       min_beta_corr_.AddRowSumMat(1.0, input_min_, mmt);
       min_beta_.AddVec(-blr, min_beta_corr_);
    }
    if (clip_gradient_ > 0.0) {  // gradient clipping
      max_alpha_corr_.ApplyFloor(-clip_gradient_);
      min_beta_corr_.ApplyFloor(-clip_gradient_);
      max_alpha_corr_.ApplyCeiling(clip_gradient_);
      min_beta_corr_.ApplyCeiling(clip_gradient_);
    }
  }

  /// Accessors to the component parameters,
  const CuVectorBase<BaseFloat>& GetAlpha() const { return max_alpha_; }

  void SetAlpha(const CuVectorBase<BaseFloat>& max_alpha) {
    KALDI_ASSERT(max_alpha.Dim() == max_alpha_.Dim());
    max_alpha_.CopyFromVec(max_alpha);
  }
  const CuVectorBase<BaseFloat>& GetBeta() const { return min_beta_; }

  void SetBeta(const CuVectorBase<BaseFloat>& min_beta) {
    KALDI_ASSERT(min_beta.Dim() == min_beta_.Dim());
    min_beta_.CopyFromVec(min_beta);
  }

 private:
  CuVector<BaseFloat> max_alpha_;
  CuVector<BaseFloat> min_beta_;

  CuVector<BaseFloat> max_alpha_corr_;
  CuVector<BaseFloat> min_beta_corr_;

  CuMatrix<BaseFloat> input_max_;
  CuMatrix<BaseFloat> input_min_;

  BaseFloat clip_gradient_;
  BaseFloat alpha_;
  BaseFloat beta_;
  BaseFloat max_learn_rate_coef_;
  BaseFloat min_learn_rate_coef_;
  BaseFloat max_fix_coef_;
  BaseFloat min_fix_coef_;
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PARAMETRIC_RELU_H_
