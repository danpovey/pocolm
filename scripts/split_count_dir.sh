#!/usr/bin/env bash

# actually this script does not honor the parallel option yet.
parallel=false

if [ "$1" == "--parallel" ]; then
  parallel=$2
  shift; shift
fi


if [ $# != 2 ]; then
  echo "Usage:"
  echo "  $0 [options] <source-count-dir> <num-splits>"
  echo "e.g.:  $0 data/counts_3 10"
  echo
  echo "This script splits the data-counts stored in <source-count-dir>"
  echo "and puts them in e.g. <source-count-dir>/split10/{1,2,3,4,...,10},"
  echo "with the same directory structure as a normal count directory."
  echo "The data-counts are split based on the most recent word in the"
  echo "history [these integer word-ids are distributed modulo the <num-splits>]"
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
rootdir=$(cd ../..; pwd -P)
PATH=$PATH:$rootdir/scripts:$rootdir/src

dir=$1
num_splits=$2

if ! validate_count_dir.py $dir; then
  echo "$0: expected input $dir to be a valid count directory."
  exit 1
fi

if ! [ "$num_splits" -gt 1 ]; then
  echo "$0: invalid num-splits (must be >1): '$num_splits'"
  exit 1
fi

mkdir -p $dir/split$num_splits

all_newer=true
for s in $(seq $num_splits); do
  if [ ! -d $dir/split$num_splits/$s ] || [ ! $dir/split$num_splits/$s/int.dev -nt $dir/int.dev ]; then
    all_newer=false
  fi
done

if $all_newer && validate_count_dir.py $dir/split$num_splits/1; then
  echo "$0: not splitting since split dir already exists in $dir/split$num_splits"
  exit 0
fi

echo "$0: creating split counts in $dir/split$num_splits"

for s in $(seq $num_splits); do
  mkdir -p $dir/split$num_splits/$s
  for f in num_train_sets num_words ngram_order names fold_dev_into_train; do
    cp $dir/$f $dir/split$num_splits/$s/$f || exit 1
  done
  # words.txt could be a fairly large file, so soft-link it.
  ln -sf ../../words.txt $dir/split$num_splits/$s/words.txt
done

num_train_sets=$(cat $dir/num_train_sets)
ngram_order=$(cat $dir/ngram_order)

files=$(echo int.dev; for s in $(seq $num_train_sets); do for o in $(seq 2 $ngram_order); do echo int.$s.$o; done; done)

for f in $files; do
  if [ ! -f $dir/$f ]; then
    echo "$0: expected $dir/$f to exist"
    exit 1
  fi
  split_files=$(for s in $(seq $num_splits); do echo $dir/split$num_splits/$s/$f; done)

  split-int-counts $split_files <$dir/$f || exit 1
done

validate_count_dir.py $dir/split$num_splits/1 || exit 1

echo "$0: Success"

exit 0
