#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

# you have to have the Fisher corpus from LDC.
fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"

local/fisher_data_prep.sh $fisher_dirs

get_word_counts.py data/text data/word_counts

get_unigram_weights.py data/word_counts > data/weights

# decide on the vocabulary.
get_vocab.py --num-words=40000 data/word_counts  > data/vocab_40k.txt

# local/srilm_baseline.sh

prepare_int_data.sh data/text data/vocab_40k.txt data/int_40k
get_counts.sh data/int_40k 3 data/counts_40k_3

mkdir -p data/optimize_40k_3
get_initial_metaparameters.py \
   --ngram-order=3 \
   --names=data/counts_40k_3/names \
   --num-train-sets=$(cat data/counts_40k_3/num_train_sets) > data/optimize_40k_3/0.metaparams

# validate_metaparameters.py \
#    --ngram-order=3 \
#   --num-train-sets=$(cat data/counts_40k_3/num_train_sets) \
#    data/optimize_40k_3/0.metaparams

# get_objf_and_derivs.py --derivs-out=data/optimize_40k_3/0.derivs \
#   data/counts_40k_3  data/optimize_40k_3/0.{metaparams,objf} data/optimize_40k_3/work.0

# validate_metaparameter_derivs.py \
#    --ngram-order=3 \
#   --num-train-sets=$(cat data/counts_40k_3/num_train_sets) \
#    data/optimize_40k_3/0.{metaparams,derivs}

#test_metaparameter_derivs.py \
#  data/optimize_40k_3/0.metaparams \
#  data/counts_40k_3 data/optimize_40k_3/temp

# # test_metaparameter_derivs.py: analytical and difference-method derivatives agree 98.9100764013%


optimize_metaparameters.py --gradient-tolerance=0.005 \
  data/counts_40k_3 data/optimize_40k_3

# optimize_metaparameters.py: log-prob on dev data increased from -4.42717734652 to -4.35856888596 over 21 passes of derivative estimation (perplexity: 83.6948416463->78.1452196398

get_counts.sh data/int_40k 4 data/counts_40k_4

mkdir -p data/optimize_40k_4
get_initial_metaparameters.py \
   --ngram-order=4 \
   --names=data/counts_40k_4/names \
   --num-train-sets=$(cat data/counts_40k_4/num_train_sets) > data/optimize_40k_4/0.metaparams

optimize_metaparameters.py --gradient-tolerance=0.005 \
  data/counts_40k_4 data/optimize_40k_4

# optimize_metaparameters.py: log-prob on dev data increased from -4.40066078407
# to -4.30905887536 over 45 passes of derivative estimation (perplexity:
# 81.5047078876->74.3704641182



split_count_dir.sh data/counts_40k_4 10


mkdir -p data/optimize_40k_4/split

get_initial_metaparameters.py \
   --ngram-order=4 \
   --names=data/counts_40k_4/names \
   --num-train-sets=$(cat data/counts_40k_4/num_train_sets) > data/optimize_40k_4/split/0.metaparams

# optimize metaparameters using one tenth of the history words;
# we'll use the resulting Hessian and parameters as a starting point for
# optimization on all the data.  note, the --num-splits=5 means
# we use 5 parallel jobs.

# caution, we need to use a num-splits that's coprime with the
# previous split of 10.


optimize_metaparameters.py --num-splits=3 --progress-tolerance=1.0e-04 \
   --write-inv-hessian=data/optimize_40k_4/split/inv_hessian \
    data/counts_40k_4/split10/1 data/optimize_40k_4/split

mkdir -p data/optimize_40k_4
cp data/optimize_40k_4/split/final.metaparams data/optimize_40k_4/0.metaparams

optimize_metaparameters.py --num-splits=5 --progress-tolerance=1.0e-05 \
   --read-inv-hessian=data/optimize_40k_4/split/inv_hessian \
    data/counts_40k_4 data/optimize_40k_4/split

