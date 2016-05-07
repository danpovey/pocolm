#!/usr/bin/env bash


# by default, we don't process the different files into counts in parallel because
# 'sort' might take up too much memory.
parallel=false


if [ $# != 3 ]; then
  echo "Usage:"
  echo "  $0 [options] <source-text-dir> <source-vocab> <dest-int-dir>"
  echo "e.g.:  $0 data/text data/words_100k.txt data/int_100k"
  echo "This program uses the vocabulary file to turn the data into"
  echo "integer form, and to give it a standard format in preparation"
  echo "for language model training."
  echo
  echo "Options"
  echo "   --fold-dev-into  <train-set-name>"
  echo "   e.g.  --fold-dev-into switchboard decrees that dev.txt should"
  echo "                 be folded into switchboard.txt after estimating"
  echo "                 the meta-parameters.  Default: none."
  echo "   --parallel <true|false>"
  echo "      Setting --parallel false will disable the (default) parallel"
  echo "      processing of multiple data sources by this script."
  exit 1
fi

# exit on error.
set -e

# make sure that the 'scripts' directory and the 'src' directory are on the
# path.
rootdir=$(cd ../..; pwd -P)
PATH=$PATH:$rootdir/scripts:$rootdir/src

if ! which -s text_to_int.py; then # >&/dev/null; then
  echo "$0: expected text_to_int.py to be on the path"
  exit 1
fi
if ! which -s get-text-counts; then # >&/dev/null; then
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
for f in num_train_sets num_words fold_dev_into_train names words.txt; do
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

  if [ $n == dev ]; then
    args=$dir/int.dev  # we write all the orders as one file for the dev data.
  else
    args="/dev/null $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.$n.$o ''; done)"
  fi

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

if [ -f $dir/.error ]; then
  echo "$0: error detected; check the logs in $dir/log"
  exit 1
fi

wait
exit 0
