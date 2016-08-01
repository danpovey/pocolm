#!/usr/bin/env bash

# This script is deprecated and will be deleted.

# by default, we don't process the different files into counts in parallel because
# 'sort' might take up too much memory.
dump_counts_parallel=false

num_jobs=5 # num_jobs is the number of pieces we split the data into
           # when applying the min-counts.
min_counts=
cleanup=true

for n in $(seq 4); do
  if [ "$1" == "--dump-counts-parallel" ]; then
    dump_counts_parallel=$2
    shift; shift
  fi
  if [ "$1" == "--num-jobs" ]; then
    num_jobs=$2
    shift; shift
  fi
  if [ "$1" == "--min-counts" ]; then
    min_counts=$2
    shift; shift
  fi
  if [ "$1" == "--cleanup" ]; then
    cleanup=$2
    shift; shift
  fi
done


if [ $# != 3 ]; then
  echo "Usage:"
  echo "This script is deprecated and will be deleted; use get_counts.py."
  echo "  $0 [options] <source-int-dir> <ngram-order> <dest-count-dir>"
  echo "e.g.:  $0 data/int 3 data/counts_3"
  echo
  echo "This script computes data-counts of the specified n-gram order"
  echo "for each data-source in <source-int-dir>, and puts them all in"
  echo "<dest-counts-dir>.  This version applies min-counts"
  echo
  echo "Options"
  echo "   --dump-counts-parallel <true|false>  [default: false]"
  echo "      Setting --dump-counts-parallel true will enable parallel"
  echo "      processing over the different sources of training data when"
  echo "      dumping the original counts."
  echo "   --num-jobs <n>   [default: 5]"
  echo "      The number of parallel jobs used when applying the min-counts."
  echo "   --min-counts <mincounts>  [required option; no default]"
  echo "     The min-counts for each order >= 3, e.g. '2' for trigram or '2 2'"
  echo "     for 4-gram.  You may also supply different min-counts per dataset,"
  echo "     e.g. for a 4-gram setup with 3 data-sets you could use the option"
  echo "     --min-counts '1,2,2 1,2,2'"
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

export LC_ALL=C
export TMPDIR=$dir


# the next few lines process the dev data; this doesn't involve any min-counts,
# so we do it separately from the training data.
args="/dev/null $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.dev.$o ''; done)"
echo "# gunzip -c $int/dev.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts $args " \
  > $dir/log/get_counts.dev.log
(gunzip -c $int/dev.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts $args) 2>>$dir/log/get_counts.dev.log || exit 1
cmd="merge-int-counts $(for o in $(seq 2 $ngram_order); do echo -n $dir/int.dev.$o ''; done) >$dir/int.dev"
echo "# $cmd" > $dir/log/merge_dev_counts.log
eval $cmd 2>>$dir/log/merge_dev_counts.log
eval $cmd || exit 1



for n in $(seq $num_train_sets); do
  set -o pipefail

  # When initially dumping the int-counts to disk, instead of splitting them by
  # n-gram order we split them on the most recent word in the history.  This
  # will enable us to do the application of min-counts in parallel.

  args="$(for j in $(seq $num_jobs); do echo -n $dir/int.$n.split$j ''; done)"

  echo "# gunzip -c $int/$n.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts /dev/stdout | split-int-counts $args " \
     > $dir/log/get_counts.$n.log
  ( gunzip -c $int/$n.txt.gz | get-text-counts $ngram_order | sort | uniq -c | get-int-counts /dev/stdout | split-int-counts $args || \
    touch $dir/.error ) 2>>$dir/log/get_counts.$n.log &

  if ! $dump_counts_parallel; then
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

## now apply the min-counts

for j in $(seq $num_jobs); do
  inputs=$(for n in $(seq $num_train_sets); do echo -n $dir/int.$n.split$j ''; done)
  outputs=$(for n in $(seq $num_train_sets); do for o in $(seq 2 $ngram_order); do echo -n $dir/int.$n.split$j.$o ''; done; done)
  cmd="int-counts-enforce-min-counts $ngram_order $min_counts $inputs $outputs"

  echo "# $cmd" >$dir/log/enforce_min_counts.$o.log
  $cmd >>$dir/log/enforce_min_counts.$o.log || touch $dir/.error &
done

wait

if [ -f $dir/.error ]; then
  echo "$0: error detected; check the logs in $dir/log/enforce_min_counts.*.log"
  exit 1
fi

if $cleanup; then
  for j in $(seq $num_jobs); do
    rm $dir/int.*.split$j
  done
fi


## Now merge together the counts for each data source and n-gram order
for n in $(seq $num_train_sets); do
  for o in $(seq 2 $ngram_order); do
    inputs=$(for j in $(seq $num_jobs); do echo -n $dir/int.$n.split$j.$o ''; done)
    cmd="merge-int-counts $inputs >$dir/int.$n.$o"
    echo "# $cmd" >$dir/log/merge_counts.$n.$o.log
    eval $cmd 2>> $dir/log/merge_counts.$n.$o.log || touch $dir/.error &
  done
done
wait

if [ -f $dir/.error ]; then
  echo "$0: error detected; check the logs in $dir/log/merge_counts.*.*.log"
  exit 1
fi

if $cleanup; then
  rm $dir/int.*.split*.*
fi

if [ -f $int/unigram_weights ]; then
  cp $int/unigram_weights $dir
else
  rm $dir/unigram_weights 2>/dev/null || true
fi

validate_count_dir.py $dir

exit 0
