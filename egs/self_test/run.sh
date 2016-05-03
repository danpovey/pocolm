#!/usr/bin/env bash

# this script tests pocolm on its own code.

export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

# the following creates a text-data directory in data/text.
local/make_data.sh

get_counts.py data/text data/counts

get_unigram_weights.py data/counts > data/weights

get_vocab.py --num-words=500 --weights=data/weights data/counts  > data/words_500.txt

prepare_int_data.sh data/text data/words_500.txt data/int_500

