#!/usr/bin/env bash

# This script is deprecated and will be deleted.

# by default, we don't process the different files into counts in parallel because
# 'sort' might take up too much memory.
parallel=false

if [ "$1" == "--parallel" ]; then
  parallel=$2
  shift; shift
fi


if [ $# != 3 ]; then
  echo "Usage:"
  echo "This script is deprecated and will be deleted; use get_counts.py."
  echo "  $0 [options] <source-int-dir> <ngram-order> <dest-count-dir>"
  echo "e.g.:  $0 data/int 3 data/counts_3"
  echo
  echo "This script computes data-counts of the specified n-gram order"
  echo "for each data-source in <source-int-dir>, and puts them all in"
  echo "<dest-counts-dir>."
  echo
  echo "Options"
  echo "   --parallel <true|false>  [default: false]"
  echo "      Setting --parallel true will enable parallel"
  echo "      processing of multiple data sources by this script."
  exit 1
fi

# exit on error.
set -e

# make sure that the 'scripts' directory and the 'src' directory are on the
# path.
rootdir=$(dirname "$0")/..
PATH=$PATH:$rootdir/scripts:$rootdir/src

if ! command -v text_to_int.py  >&/dev/null; then
  echo "$0: expected text_to_int.py to be on the path"
  exit 1
fi
if ! command -v get-text-counts >&/dev/null; then
  echo "$0: expected get-text-counts to be on the path"
  exit 1
fi

int=$1
ngram_order=$2
dir=$3

set -e

validate_int_dir.py $int

if ! [ "$ngram_order" -gt 1 ]; then
  echo "$0: ngram-order must be at least 2 (if you want a unigram LM, do it by hand)"
  exit 1
fi

mkdir -p $dir/log

num_train_sets=$(cat $int/num_train_sets)

# copy over some meta-info into the 'counts' directory.
for f in num_train_sets num_words names words.txt; do
  cp $int/$f $dir/$f
done

# save the n-gram order.
echo $ngram_order > $dir/ngram_order

rm -f $dir/.error $dir/int.*

for n in $(seq $num_train_sets) dev; do
  set -o pipefail
  # get-int-counts has an output for each order of count, but the maximum order
  # is >1, and in this case the 1-gram counts are always zero (we always output
  # the highest possible order for each word, which is normally $ngram_order,
  # but can be as low as 2 for the 1st word of the sentence). So just put the
  # output for order 1 in /dev/null.

  args="/dev/null $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.$n.$o ''; done)"

  export LC_ALL=C
  export TMPDIR=$dir
  echo "# gunzip -c $int/$n.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts $args " \
     > $dir/log/get_counts.$n.log
  ( gunzip -c $int/$n.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts $args || \
    touch $dir/.error ) 2>>$dir/log/get_counts.$n.log &

  if ! $parallel; then
    wait
    if [ -f $dir/.error ]; then
      cat $dir/log/get_counts.$n.log
      echo "$0: error detected; check the logs in $dir/log"
      exit 1
    fi
  fi
done

wait

if [ -f $dir/.error ]; then
  echo "$0: error detected; check the logs in $dir/log"
  exit 1
fi

if [ -f $int/unigram_weights ]; then
  cp $int/unigram_weights $dir
else
  rm $dir/unigram_weights 2>/dev/null || true
fi

# we also want the files $dir/int.dev i.e. the dev data not split up by n-gram
# order, which is required for evaluating dev-set log-probs.
cmd="merge-int-counts $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.dev.$o ''; done) >$dir/int.dev 2>$dir/log/merge_dev_counts.log"
# cmd="split-int-counts-by-order /dev/null $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.dev.$o ''; done) <$dir/int.dev 2>$dir/log/split_int_counts.log"
echo "# $cmd"
eval $cmd

exit 0
