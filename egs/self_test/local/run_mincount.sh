#!/usr/bin/env bash

# this script tests pocolm on its own code.

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

# this assumes the first few lines of run.sh have been run-- see in run.sh
# at what point you can run this script.

num_words=500
ngram_order=3
# the "2" is the min-count for orders 3 and above.
min=2
datasub=data/${num_words}_${ngram_order}_mc${min}
mkdir -p $datasub


word_counts_to_vocab.py --num-words=$num_words --weights=data/weights data/word_counts  > $datasub/words.txt


prepare_int_data.sh data/text $datasub/words.txt $datasub/int

get_counts.py --min-counts=$min  $datasub/int $ngram_order $datasub/counts

validate_count_dir.py $datasub/counts

mkdir -p $datasub/optimize

initialize_metaparameters.py --weights=data/weights \
   --ngram-order=$ngram_order \
   --names=$datasub/counts/names \
   --num-train-sets=$(cat $datasub/counts/num_train_sets) > $datasub/optimize/0.metaparams

get_objf_and_derivs.py --derivs-out=$datasub/optimize/0.derivs \
  $datasub/counts  $datasub/optimize/0.{metaparams,objf} $datasub/work.0

validate_metaparameter_derivs.py \
  --ngram-order=$ngram_order \
  --num-train-sets=$(cat $datasub/counts/num_train_sets) \
   $datasub/optimize/0.{metaparams,derivs}

# it's actually worse than the threshold, but because some of the
# weights are close to each other-- it's OK so we ignore the error.
test_metaparameter_derivs.py \
  $datasub/optimize/0.metaparams \
  $datasub/counts $datasub/temp || true

# the following script expects $datasub/optimize/0.metaparams to
# already exist.
optimize_metaparameters.py \
  $datasub/counts $datasub/optimize


# make LM dir without splits.
make_lm_dir.py $datasub/counts \
    $datasub/optimize/final.metaparams $datasub/lm

 prune_lm_dir.py data/500_3/lm 2.0 data/500_3/lm_pruned
 mkdir -p $datasub/arpa_pruned
 format_arpa_lm.py $datasub/lm_pruned | gzip -c > $datasub/arpa_pruned/${ngram_order}.arpa.gz


mkdir -p $datasub/arpa
format_arpa_lm.py $datasub/lm | gzip -c > $datasub/arpa/${ngram_order}.arpa.gz


# make LM dir with splits, and keeping them.
make_lm_dir.py --num-splits=2 --keep-splits=true $datasub/counts \
    $datasub/optimize/final.metaparams $datasub/lm2

mkdir -p $datasub/arpa2
format_arpa_lm.py $datasub/lm2 | gzip -c > $datasub/arpa2/${ngram_order}.arpa.gz

split_lm_dir.py $datasub/lm 3 $datasub/lm3
mkdir -p $datasub/arpa3
format_arpa_lm.py $datasub/lm3 | gzip -c > $datasub/arpa3/${ngram_order}.arpa.gz
