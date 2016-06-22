#!/usr/bin/env bash

get_counts.py data/int_20k 3 data/counts_20k_3

mkdir -p data/optimize_20k_3
get_initial_metaparameters.py \
   --ngram-order=3 \
   --names=data/counts_20k_3/names \
   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) > data/optimize_20k_3/0.metaparams

validate_metaparameters.py \
    --ngram-order=3 \
   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) \
    data/optimize_20k_3/0.metaparams

get_objf_and_derivs.py --derivs-out=data/optimize_20k_3/0.derivs \
   data/counts_20k_3  data/optimize_20k_3/0.{metaparams,objf} data/optimize_20k_3/work.0

validate_metaparameter_derivs.py \
   --ngram-order=3 \
   --num-train-sets=$(cat data/counts_20k_3/num_train_sets) \
    data/optimize_20k_3/0.{metaparams,derivs}

test_metaparameter_derivs.py \
  data/optimize_20k_3/0.metaparams \
  data/counts_20k_3 data/optimize_20k_3/temp

# test_metaparameter_derivs.py: analytical and difference-method derivatives agree 98.641527436%
