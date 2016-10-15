#!/usr/bin/env bash

# this script tests some parts of pocolm on some very tiny files,
# just because we sometimes want a setup where things are so small
# that we can work things out by formula to compare them with
# manually-worked-out numbers.

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

mkdir -p data/text

(echo a b c; echo a c) > data/text/train.txt

# the following won't actually be used.
(echo a b c) > data/text/dev.txt

(echo a; echo b; echo c) > wordlist

wordlist_to_vocab.py wordlist > data/words.txt

ngram_order=3

prepare_int_data.py data/text data/words.txt data/int

get_counts.sh data/int $ngram_order data/counts

initialize_metaparameters.py \
   --ngram-order=$ngram_order \
   --num-train-sets=1 > data/metaparams

get_objf_and_derivs.py data/counts data/metaparams data/objf data/work

# the following is broken right now.
# local/test_float_counts.sh
