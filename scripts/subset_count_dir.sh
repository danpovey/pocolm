#!/usr/bin/env bash

if [ $# != 3 ]; then
  echo "Usage:" 1>&2
  echo "  $0 <source-count-dir> <ratio> <dest-count-dir>" 1>&2
  echo "e.g.:  $0 data/counts_3 10 data/counts_3/subset10" 1>&2
  echo 1>&2
  echo "This script takes the fraction 1/<ratio> of the data-counts " 1>&2
  echo "stored in <source-count-dir> (divided by immediate history word)," 1>&2
  echo "and puts them in <dest-count-dir> with the same directory" 1>&2
  echo " structure as a normal count directory." 1>&2
  echo "This is the same as split_count_dir.sh, and only keeping the" 1>&2
  echo "first split.  It's typically used for when you have a large" 1>&2
  echo "setup and want to do a first pass of metaparameter optimization" 1>&2
  echo "with a small subset of the data." 1>&2
  echo "Caution: this script will exit with success if it looks like" 1>&2
  echo "the data was already split and the <dest-count-dir> is newer" 1>&2
  echo "than the <source-count-dir>, so it's a good idea to encode" 1>&2
  echo "the ratio in the dirname of the <dest-count-dir>." 1>&2
  exit 1
fi

# exit on error.
set -e

# make sure that the 'scripts' directory and the 'src' directory are on the
# path.
rootdir=$(dirname "$0")/..
PATH=$PATH:$rootdir/scripts:$rootdir/src

dir=$1
ratio=$2
destdir=$3

if ! validate_count_dir.py $dir; then
  echo "$0: expected input $dir to be a valid count directory." 1>&2
  exit 1
fi

if [ "$destdir" == "$dir" ]; then
  echo "$0: source and dest dirs cannot be the same." 1>&2
  exit 1
fi

if ! [ "$ratio" -gt 1 ]; then
  echo "$0: invalid ratio (must be >1): '$ratio'" 1>&2
  exit 1
fi

mkdir -p $destdir || exit 1

if [ $destdir/int.dev -nt $dir/int.dev ] && \
  validate_count_dir.py $destdir; then
  echo "$0: not splitting since dest dir appears to be up to date" 1>&2
  exit 0
fi

echo "$0: creating split counts in $dir/split$num_splits" 1>&2


for f in num_train_sets num_words ngram_order names words.txt; do
  cp $dir/$f $destdir/$f || exit 1
done

if [ -f $dir/unigram_weights ]; then  # this file is optional.
  cp $dir/unigram_weights $destdir || exit 1
fi

num_train_sets=$(cat $dir/num_train_sets)
ngram_order=$(cat $dir/ngram_order)

files=$(echo int.dev; for s in dev $(seq $num_train_sets); do for o in $(seq 2 $ngram_order); do echo int.$s.$o; done; done)

# Normally we'll be subsetting a directory that has not already been
# split or subsetted, so old_split_modulus will be 1.
old_split_modulus=1
[ -f $dir/split_modulus ] && old_split_modulus=$(cat $dir/split_modulus)

for f in $files; do
  if [ ! -f $dir/$f ]; then
    echo "$0: expected $dir/$f to exist" 1>&2
    exit 1
  fi
  # e.g. split_files="$destdir/int.1.1 /dev/null /dev/null /dev/null"
  split_files="$destdir/$f $(for s in $(seq 2 $ratio); do echo /dev/null; done)"

  split-int-counts -d $old_split_modulus $split_files <$dir/$f || exit 1
done

# the 'split_modulus' file will be
new_split_modulus=$[$old_split_modulus*$ratio]
echo $new_split_modulus >$destdir/split_modulus

validate_count_dir.py $destdir || exit 1

echo "$0: Success" 1>&2

exit 0
