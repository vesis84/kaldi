// nnet/nnet-parametric-relu.h

<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
// Copyright 2016 Brno University of Technology (author: Murali Karthick B)
//           2011-2014  Brno University of Technology (author: Karel Vesely)
=======
// Copyright 2011-2014  Brno University of Technology (author: Karel Vesely)
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h

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
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    alpha_(dim_out), 
    beta_(dim_out),
    alpha_corr_(dim_out),
    beta_corr_(dim_out),
=======
    max_alpha_(dim_out), min_beta_(dim_out),
    max_alpha_corr_(dim_out), min_beta_corr_(dim_out),
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    clip_gradient_(0.0)
  { }
  ~ParametricRelu()
  { }

  Component* Copy() const { return new ParametricRelu(*this); }
  ComponentType GetType() const { return kParametricRelu; }

  void InitData(std::istream &is) {
    // define options
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    float init_alpha = 1, init_beta = 0.25;
    float fix_alpha = 0.0, fix_beta = 0.0;
=======
    float alpha = 1, beta = 0.25,
    max_learn_rate_coef = 0.000005,
    min_learn_rate_coef = 0.00005,
    max_fix_coef = 0.0,
    min_fix_coef = 0.0,
    clip_gradient = 0.0;
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h

    // parse config
    std::string token;
    while (is >> std::ws, !is.eof()) {
      ReadToken(is, false, &token);
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
      /**/ if (token == "<InitAlpha>")  ReadBasicType(is, false, &init_alpha);
      else if (token == "<InitBeta>")  ReadBasicType(is, false, &init_beta);
      else if (token == "<BiasLearnRateCoef>")  ReadBasicType(is, false, &bias_learn_rate_coef_);
      else if (token == "<LearnRateCoef>")  ReadBasicType(is, false, &learn_rate_coef_);
      else if (token == "<FixAlpha>")  ReadBasicType(is, false, &fix_alpha);
      else if (token == "<FixBeta>")   ReadBasicType(is, false, &fix_beta);
      else if (token == "<ClipGradient>")  ReadBasicType(is, false, &clip_gradient_);
      else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                  << " (InitAlpha|InitBeta|BiasLearnRateCoef|LearnRateCoef|FixAlpha|FixBeta|ClipGradient)";
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    }

    //
    // Initialize trainable parameters,
    //

    Vector<BaseFloat> veca(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input alpha
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
      veca(i) = init_alpha;
    }
    alpha_ = veca;
=======
      veca(i) = alpha;
    }
    max_alpha_ = veca;
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    //
    Vector<BaseFloat> vecb(output_dim_);
    for (int32 i = 0; i < output_dim_; i++) {
      // elements of vector is initialized with input beta
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
      vecb(i) = init_beta;
    }
    beta_ = vecb;
    //
    fix_alpha_ = fix_alpha;
    fix_beta_ = fix_beta;
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    KALDI_ASSERT(clip_gradient_ >= 0.0);
  }

  void ReadData(std::istream &is, bool binary) {
    // Read all the '<Tokens>' in arbitrary order,
    while ('<' == Peek(is, binary)) {
      int first_char = PeekToken(is, binary);
      switch (first_char) {
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
        case 'B': ExpectToken(is, binary, "<BiasLearnRateCoef>");
          ReadBasicType(is, binary, &bias_learn_rate_coef_);
          ExpectToken(is, binary, "<LearnRateCoef>");
          ReadBasicType(is, binary, &learn_rate_coef_);
          break;
        case 'F': ExpectToken(is, binary, "<FixAlpha>");
          ReadBasicType(is, binary, &fix_alpha_);
          ExpectToken(is, binary, "<FixBeta>");
          ReadBasicType(is, binary, &fix_beta_);
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
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
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    alpha_.Read(is, binary);
    beta_.Read(is, binary);
    KALDI_ASSERT(alpha_.Dim() == output_dim_);
    KALDI_ASSERT(beta_.Dim() == output_dim_);
  }

  void WriteData(std::ostream &os, bool binary) const {
    WriteToken(os, binary, "<BiasLearnRateCoef>");
    WriteBasicType(os, binary, bias_learn_rate_coef_);
    WriteToken(os, binary, "<LearnRateCoef>");
    WriteBasicType(os, binary, learn_rate_coef_);
    WriteToken(os, binary, "<FixAlpha>");
    WriteBasicType(os, binary, fix_alpha_);
    WriteToken(os, binary, "<FixBeta>");
    WriteBasicType(os, binary, fix_beta_);
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    WriteToken(os, binary, "<ClipGradient>");
    WriteBasicType(os, binary, clip_gradient_);

    // ParametricRelu scaling parameters

    if (!binary) os << "\n";
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
     alpha_.Write(os, binary);
     beta_.Write(os, binary);
  }

  int32 NumParams() const {
    return alpha_.Dim() + beta_.Dim();
=======
     max_alpha_.Write(os, binary);
     min_beta_.Write(os, binary);
  }

  int32 NumParams() const {
    return max_alpha_.Dim() + min_beta_.Dim();
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }

  void GetGradient(VectorBase<BaseFloat>* gradient) const {
    KALDI_ASSERT(gradient->Dim() == NumParams());
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    gradient->Range(0, alpha_num_elem).CopyFromVec(Vector<BaseFloat>(alpha_corr_));
    gradient->Range(alpha_num_elem, beta_num_elem).CopyFromVec(Vector<BaseFloat>(beta_corr_));
=======
    int32 max_alpha_num_elem = max_alpha_.Dim();
    int32 min_beta_num_elem = min_beta_.Dim();
    gradient->Range(0, max_alpha_num_elem).CopyFromVec(Vector<BaseFloat>(max_alpha_corr_));
    gradient->Range(max_alpha_num_elem, min_beta_num_elem).CopyFromVec(Vector<BaseFloat>(min_beta_corr_));
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }

  void GetParams(VectorBase<BaseFloat>* params) const {
    KALDI_ASSERT(params->Dim() == NumParams());
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    params->Range(0, alpha_num_elem).CopyFromVec(Vector<BaseFloat>(alpha_));
    params->Range(alpha_num_elem, beta_num_elem).CopyFromVec(Vector<BaseFloat>(beta_));
=======
    int32 max_alpha_num_elem = max_alpha_.Dim();
    int32 min_beta_num_elem = min_beta_.Dim();
    params->Range(0, max_alpha_num_elem).CopyFromVec(Vector<BaseFloat>(max_alpha_));
    params->Range(max_alpha_num_elem, min_beta_num_elem).CopyFromVec(Vector<BaseFloat>(min_beta_));
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }

  void SetParams(const VectorBase<BaseFloat>& params) {
    KALDI_ASSERT(params.Dim() == NumParams());
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    int32 alpha_num_elem = alpha_.Dim();
    int32 beta_num_elem = beta_.Dim();
    alpha_.CopyFromVec(params.Range(0, alpha_num_elem));
    beta_.CopyFromVec(params.Range(alpha_num_elem, beta_num_elem));
  }

  std::string Info() const {
    return std::string("\n  alpha") +
      MomentStatistics(alpha_) +
      ", bias-lr-coef " + ToString(bias_learn_rate_coef_) +
      "\n  beta" + MomentStatistics(beta_) +
      ", lr-coef " + ToString(learn_rate_coef_);
  }
  std::string InfoGradient() const {
    return std::string("\n  alpha_grad") +
      MomentStatistics(alpha_corr_) +
      ", bias-lr-coef " + ToString(bias_learn_rate_coef_) +
      "\n  beta_grad" + MomentStatistics(beta_corr_) +
      ", lr-coef " + ToString(learn_rate_coef_);
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }

  void PropagateFnc(const CuMatrixBase<BaseFloat> &in,
                    CuMatrixBase<BaseFloat> *out) {
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    // Multiply activations by ReLU scalars (max_alpha, min_beta):
    // // out = in * (in >= 0.0 ? in * max_alpha : in * min_beta)
    out->ParametricRelu(in, alpha_, beta_);
=======
    out->ParametricRelu(in, max_alpha_, min_beta_);
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }

  void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in,
                        const CuMatrixBase<BaseFloat> &out,
                        const CuMatrixBase<BaseFloat> &out_diff,
                        CuMatrixBase<BaseFloat> *in_diff) {
    // multiply error derivative by activations (alpha, beta)
    // ((dE/da)*w) === out_diff, f(y) == out,
    //  (out > 0 ) ? out_diff * alpha : out_diff * beta
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    in_diff->DiffParametricRelu(out, out_diff, alpha_, beta_);
=======
    in_diff->DiffParametricRelu(out, out_diff, max_alpha_, min_beta_);
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
  }


  void Update(const CuMatrixBase<BaseFloat> &input,
              const CuMatrixBase<BaseFloat> &diff) {
    // we use following hyperparameters from the option class
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
    const BaseFloat alr = opts_.learn_rate * bias_learn_rate_coef_;
    const BaseFloat blr = opts_.learn_rate * learn_rate_coef_;
    const BaseFloat mmt = opts_.momentum;
    if (clip_gradient_ > 0.0) {  // gradient clipping
      alpha_corr_.ApplyFloor(-clip_gradient_);
      beta_corr_.ApplyFloor(-clip_gradient_);
      alpha_corr_.ApplyCeiling(clip_gradient_);
      beta_corr_.ApplyCeiling(clip_gradient_);
    }
    // compute gradient (incl. momentum)
    if (!fix_alpha_) {  // the alpha parameter is learnable
       in_alpha_.Resize(input.NumRows(), input.NumCols());
       in_alpha_.CopyFromMat(input);
       in_alpha_.ApplyFloor(0.0);
       in_alpha_.MulElements(diff);
       alpha_corr_.AddRowSumMat(1.0, in_alpha_, mmt);
       alpha_.AddVec(-alr, alpha_corr_);
    }
    if (!fix_beta_) {  // the beta parameter is learnable
       in_beta_.Resize(input.NumRows(), input.NumCols());
       in_beta_.CopyFromMat(input);
       in_beta_.ApplyCeiling(0.0);
       in_beta_.MulElements(diff);
       beta_corr_.AddRowSumMat(1.0, in_beta_, mmt);
       beta_.AddVec(-blr, beta_corr_);
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
    }
  }

  /// Accessors to the component parameters,
<<<<<<< 0fab8268c73f12a84e7e69864988f3f3c6fc153e
  const CuVectorBase<BaseFloat>& GetAlpha() const { return alpha_; }

  void SetAlpha(const CuVectorBase<BaseFloat>& alpha) {
    KALDI_ASSERT(alpha.Dim() == alpha_.Dim());
    alpha_.CopyFromVec(alpha);
  }
  const CuVectorBase<BaseFloat>& GetBeta() const { return beta_; }

  void SetBeta(const CuVectorBase<BaseFloat>& beta) {
    KALDI_ASSERT(beta.Dim() == beta_.Dim());
    beta_.CopyFromVec(beta);
  }

 private:
  CuVector<BaseFloat> alpha_; /// < Vector of 'alphas', one value per neuron.
  CuVector<BaseFloat> beta_; /// < Vector of 'betas', one value per neuron.

  CuVector<BaseFloat> alpha_corr_; /// < Vector of 'alpha' updates.
  CuVector<BaseFloat> beta_corr_; /// < Vector of 'beta' updates.

  CuMatrix<BaseFloat> in_alpha_;
  CuMatrix<BaseFloat> in_beta_;

  BaseFloat clip_gradient_;
  BaseFloat fix_alpha_;
  BaseFloat fix_beta_;
=======
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
>>>>>>> 	new file:   src/nnet/nnet-parametric-relu.h
};

}  // namespace nnet1
}  // namespace kaldi

#endif  // KALDI_NNET_NNET_PARAMETRIC_RELU_H_
