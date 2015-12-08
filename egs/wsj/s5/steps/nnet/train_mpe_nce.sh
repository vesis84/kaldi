#!/bin/bash
# Copyright 2013-2015  Brno University of Technology (author: Karel Vesely)  
# Apache 2.0.

# Sequence-discriminative MPE/sMBR training of DNN.
# 4 iterations (by default) of Stochastic Gradient Descent with per-utterance updates.
# We select between MPE/sMBR optimization by '--do-smbr <bool>' option.

# For the numerator we have a fixed alignment rather than a lattice--
# this actually follows from the way lattices are defined in Kaldi, which
# is to have a single path for each word (output-symbol) sequence.


# Begin configuration section.
cmd=run.pl
num_iters=4
acwt=0.1
lmwt=1.0
learn_rate=0.00001
halving_factor=1.0 #ie. disable halving
do_smbr=true
exclude_silphones=true # exclude silphones from approximate accuracy computation
unkphonelist= # exclude unkphones from approximate accuracy computation (overrides exclude_silphones)
one_silence_class=true # true : reduce insertions in sMBR/MPE FW/BW, more stable training,
nce_scale=1.0
nce_gradient_scale=0.5
xent_gradient_scale=0.2
nce_fast=false
verbose=3
ivector=

seed=777    # seed value used for training data shuffling
skip_cuda_check=false
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -euxo pipefail

if [ $# -ne 6 ]; then
  echo "Usage: $0 <data> <lang> <srcdir> <ali> <denlats> <exp>"
  echo " e.g.: $0 data/train_all data/lang exp/tri3b_dnn exp/tri3b_dnn_ali exp/tri3b_dnn_denlats exp/tri3b_dnn_smbr"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --num-iters <N>                                  # number of iterations to run"
  echo "  --acwt <float>                                   # acoustic score scaling"
  echo "  --lmwt <float>                                   # linguistic score scaling"
  echo "  --learn-rate <float>                             # learning rate for NN training"
  echo "  --do-smbr <bool>                                 # do sMBR training, otherwise MPE"
  
  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4
denlatdir=$5
dir=$6

model=$srcdir/final.mdl
nnet=$srcdir/final.nnet
feature_transform=$srcdir/final.feature_transform
class_frame_counts=$srcdir/ali_train_pdf.counts

for f in $model $nnet $feature_transform $class_frame_counts $srcdir/{final.mdl,tree} $data/feats.scp $alidir/ali.1.gz $denlatdir/lat.scp; do
  [ ! -f $f ] && echo "$0: Missing $f" && exit 1;
done

# check if CUDA compiled in,
if ! $skip_cuda_check; then cuda-compiled || { echo "Error, CUDA not compiled-in!"; exit 1; } fi

mkdir -p $dir/log

silphonelist=`cat $lang/phones/silence.csl`
cp $srcdir/{final.mdl,tree} $dir
cp $nnet $dir/0.nnet; nnet=$dir/0.nnet
cp $feature_transform $dir/final.feature_transform

[ -f $srcdir/prior_counts ] && class_frame_counts=$srcdir/prior_counts
cp $class_frame_counts $dir/ali_train_pdf.counts

# Frames with '--silence-phones' are excluded from FW-BW computation,
mpe_silphones_arg= #empty
$exclude_silphones && mpe_silphones_arg="--silence-phones=$silphonelist" # all silphones
[ ! -z $unkphonelist ] && mpe_silphones_arg="--silence-phones=$unkphonelist" # unk only

# Shuffle the feature list for the SGD!
# By shuffling features, we have to use lattices with random access (indexed by .scp file).
cat $data/feats.scp | utils/shuffle_list.pl --srand $seed > $dir/train.scp

### FEATURE EXTRACTION PIPELINE
# import config,
cmvn_opts=
delta_opts=
D=$srcdir
[ -e $D/norm_vars ] && cmvn_opts="--norm-means=true --norm-vars=$(cat $D/norm_vars)" # Bwd-compatibility,
[ -e $D/cmvn_opts ] && cmvn_opts=$(cat $D/cmvn_opts)
[ -e $D/delta_order ] && delta_opts="--delta-order=$(cat $D/delta_order)" # Bwd-compatibility,
[ -e $D/delta_opts ] && delta_opts=$(cat $D/delta_opts)
#
# Create the feature stream,
feats="ark,o:copy-feats scp:$dir/train.scp ark:- |"
# apply-cmvn (optional),
[ ! -z "$cmvn_opts" -a ! -f $data/cmvn.scp ] && echo "$0: Missing $data/cmvn.scp" && exit 1
[ ! -z "$cmvn_opts" ] && feats="$feats apply-cmvn $cmvn_opts --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
# add-deltas (optional),
[ ! -z "$delta_opts" ] && feats="$feats add-deltas $delta_opts ark:- ark:- |"
# add-pytel transform (optional),
[ -e $D/pytel_transform.py ] && feats="$feats /bin/env python $D/pytel_transform.py |"

# add-ivector (optional),
if [ -e $D/ivector_dim ]; then
  ivector_dim=$(cat $D/ivector_dim)
  [ -z $ivector ] && echo "Missing --ivector, they were used in training! (dim $ivector_dim)" && exit 1
  ivector_dim2=$(copy-vector "$ivector" ark,t:- | head -n1 | awk '{ print NF-3 }') || true
  [ $ivector_dim != $ivector_dim2 ] && "Error, i-vector dimensionality mismatch! (expected $ivector_dim, got $ivector_dim2 in $ivector)" && exit 1
  # Append to feats
  feats="$feats append-vector-to-feats ark:- '$ivector' ark:- |"
fi

### Record the setup,
[ ! -z "$cmvn_opts" ] && echo $cmvn_opts >$dir/cmvn_opts
[ ! -z "$delta_opts" ] && echo $delta_opts >$dir/delta_opts
[ -e $D/pytel_transform.py ] && cp {$D,$dir}/pytel_transform.py
[ -e $D/ivector_dim ] && cp {$D,$dir}/ivector_dim
###

### ALIGNMENT PIPELINE,
# Assuming all alignments will fit into memory,
ali="ark:gunzip -c $alidir/ali.*.gz |"

### LATTICE PIPELINE,
# The lattices are indexed by SCP, each file is gzipped separately,
lats="scp:$denlatdir/lat.scp"

# Run several iterations of the MPE/sMBR training
cur_mdl=$nnet
x=1
while [ $x -le $num_iters ]; do
  echo "Pass $x (learnrate $learn_rate)"
  if [ -f $dir/${x}.nnet ]; then
    echo "Skipped, file $dir/$x.nnet exists"
  else
    # Run epoch,
    $cmd $dir/log/mpe.${x}.log \
     nnet-train-mpe-nce-sequential \
       --feature-transform=$feature_transform \
       --class-frame-counts=$class_frame_counts \
       --acoustic-scale=$acwt \
       --lm-scale=$lmwt \
       --nce-scale=$nce_scale \
       --nce-gradient-scale=$nce_gradient_scale \
       --xent-gradient-scale=$xent_gradient_scale \
       --nce-fast=$nce_fast \
       --learn-rate=$learn_rate \
       --do-smbr=$do_smbr \
       --verbose=$verbose \
       --one-silence-class=$one_silence_class \
       $mpe_silphones_arg \
       $cur_mdl $alidir/final.mdl "$feats" "$lats" "$ali" $dir/${x}.nnet
  fi
  cur_mdl=$dir/${x}.nnet
  # report the progress,
  grep -B 3 "Overall average Negative Conditional Entropy" $dir/log/mpe.${x}.log | sed -e 's|.*)||'
  # 
  x=$((x+1))
  learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
done

(cd $dir; [ -e final.nnet ] && unlink final.nnet; ln -s $((x-1)).nnet final.nnet)


echo "MPE/sMBR training finished"

if [ -e $dir/prior_counts ]; then
  echo "Priors are already re-estimated, skipping... ($dir/prior_counts)"
else
  echo "Re-estimating priors by forwarding the training set."
  . cmd.sh
  nj=$(cat $alidir/num_jobs)
  steps/nnet/make_priors.sh --cmd "$train_cmd" --nj $nj $data $dir
fi

# Done.
