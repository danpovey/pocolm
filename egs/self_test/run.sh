#!/usr/bin/env bash

# this script tests pocolm on its own code.

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

# the following creates a text-data directory in data/text.
local/make_data.sh

get_word_counts.py data/text data/word_counts

get_unigram_weights.py data/word_counts > data/weights

num_words=500
ngram_order=3

mkdir -p data/${num_words}_${ngram_order}

datasub=data/${num_words}_${ngram_order}
mkdir -p $datasub

get_vocab.py --num-words=$num_words --weights=data/weights data/word_counts  > $datasub/words.txt

prepare_int_data.sh data/text $datasub/words.txt $datasub/int

# note, get_counts.sh may later be called by another script.

get_counts.sh $datasub/int $ngram_order $datasub/counts

validate_count_dir.py $datasub/counts

mkdir -p $datasub/optimize

get_initial_metaparameters.py --weights=data/weights \
   --ngram-order=$ngram_order \
   --names=$datasub/counts/names \
   --num-train-sets=$(cat $datasub/counts/num_train_sets) > $datasub/optimize/0.metaparams

validate_metaparameters.py \
   --ngram-order=$ngram_order \
  --num-train-sets=$(cat $datasub/counts/num_train_sets) \
   $datasub/optimize/0.metaparams

get_objf_and_derivs.py --derivs-out=$datasub/optimize/0.derivs \
  $datasub/counts  $datasub/optimize/0.{metaparams,objf} $datasub/work.0

validate_metaparameter_derivs.py \
   --ngram-order=$ngram_order \
  --num-train-sets=$(cat $datasub/counts/num_train_sets) \
   $datasub/optimize/0.{metaparams,derivs}

# we probably wouldn't run the following on a large setup, it's slow.
test_metaparameter_derivs.py \
  $datasub/optimize/0.metaparams \
  $datasub/counts $datasub/temp


# the following script expects $datasub/optimize/0.metaparams to
# already exist.
optimize_metaparameters.py \
  $datasub/counts $datasub/optimize


#mkdir -p data/optimize


