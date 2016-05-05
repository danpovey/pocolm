#!/usr/bin/env bash

# this script tests pocolm on its own code.

export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

# the following creates a text-data directory in data/text.
local/make_data.sh

get_word_counts.py data/text data/word_counts

get_unigram_weights.py data/word_counts > data/weights

get_vocab.py --num-words=500 --weights=data/weights data/word_counts  > data/words_500.txt

prepare_int_data.sh data/text data/words_500.txt data/int_500

# note, get_counts.sh may later be called by another script.
ngram_order=3

get_counts.sh data/int_500 $ngram_order data/counts_500

validate_count_dir.py data/counts_500

mkdir -p data/optimize
get_initial_metaparameters.py --weights=data/weights \
   --ngram-order=$ngram_order \
   --names=data/counts_500/names \
   --num-train-sets=$(cat data/counts_500/num_train_sets) > data/optimize/0.metaparams

validate_metaparameters.py \
   --ngram-order=$ngram_order \
  --num-train-sets=$(cat data/counts_500/num_train_sets) \
   data/optimize/0.metaparams




mkdir -p data/optimize


