#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

get_word_counts.py data/text data/word_counts

# decide on the vocabulary.
get_vocab.py --num-words=20000 data/word_counts  > data/vocab_20k.txt

# local/srilm_baseline.sh



prepare_int_data.sh data/text data/vocab_20k.txt data/int_20k
get_counts.sh data/int_20k 3 data/counts_20k_3

mkdir -p data/optimize_20k_3
get_initial_metaparameters.py \
   --ngram-order=3 \
   --names=data/counts_20k_3/names \
   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) > data/optimize_20k_3/0.metaparams

# validate_metaparameters.py \
#    --ngram-order=3 \
#   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) \
#    data/optimize_20k_3/0.metaparams

# get_objf_and_derivs.py --derivs-out=data/optimize_20k_3/0.derivs \
#   data/counts_20k_3  data/optimize_20k_3/0.{metaparams,objf} data/optimize_20k_3/work.0

# validate_metaparameter_derivs.py \
#    --ngram-order=3 \
#   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) \
#    data/optimize_20k_3/0.{metaparams,derivs}

#test_metaparameter_derivs.py \
#  data/optimize_20k_3/0.metaparams \
#  data/counts_20k_3 data/optimize_20k_3/temp

# # test_metaparameter_derivs.py: analytical and difference-method derivatives agree 98.9100764013%


optimize_metaparameters.py --gradient-tolerance=0.005 \
  data/counts_20k_3 data/optimize_20k_3

# log-prob on dev data increased from -4.42278966972 to -4.41127767165 over 6 passes of derivative estimation (perplexity: 83.3284201887->82.3746440442)


get_counts.sh data/int_20k 4 data/counts_20k_4

mkdir -p data/optimize_20k_4
get_initial_metaparameters.py \
   --ngram-order=4 \
   --names=data/counts_20k_4/names \
   --num-train-sets=$(cat data/counts_20k_4/num_train_sets) > data/optimize_20k_4/0.metaparams

optimize_metaparameters.py --gradient-tolerance=0.005 \
  data/counts_20k_4 data/optimize_20k_4

# optimize_metaparameters.py: log-prob on dev data increased from -4.4224930661
# to -4.38089709795 over 13 passes of derivative estimation (perplexity:
# 83.3037083426->79.909688077