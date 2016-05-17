#!/usr/bin/env bash


# by default, process the different files into integer form in parallel.
parallel=true
fold_dev_into=

# parse options.
for iter in 1 2; do
  if [ "$1" == "--fold-dev-into" ]; then
    fold_dev_into="$2"
    shift; shift;
  fi
  if [ "$1" == "--parallel" ]; then
    parallel="$2"
    shift; shift
  fi
done

if [ $# != 3 ]; then
  echo "Usage:"
  echo "  $0 [options] <source-text-dir> <source-vocab> <dest-int-dir>"
  echo "e.g.:  $0 data/text data/words_100k.txt data/int_100k"
  echo "This program uses the vocabulary file to turn the data into"
  echo "ASCII-integer form, and to give it a standard format in preparation"
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
scriptdir=$(cd $(dirname $0); pwd -P)
PATH=$PATH:$scriptdir

if ! command -v text_to_int.py; then # >&/dev/null; then
  echo "$0: expected text_to_int.py to be on the path"
  exit 1
fi

# do some checking.
text=$1
vocab=$2
dir=$3

validate_text_dir.py $text
validate_vocab.py $vocab

mkdir -p $dir
# remove any old *.int.gz files in $dir.
rm $dir/*.int.gz 2>/dev/null || true

$scriptdir/internal/get_names.py $text > $dir/names

rm -f $dir/.error
mkdir -p $dir/log

num_train_sets=$(cat $dir/names | wc -l)
echo $num_train_sets > $dir/num_train_sets
cp $vocab $dir/words.txt

num_words=$(tail -n 1 $vocab | awk '{print $2}') || exit 1
echo $num_words >$dir/num_words

# we can include the preparation of the dev data in the following
# by adding "0 dev" to the contents of $dir/names using cat.

echo "dev dev" | cat - $dir/names | while read int name; do
  set -o pipefail
  ( if [ -f $text/$name.txt.gz ]; then gunzip -c $text/$name.txt.gz; else cat $text/$name.txt; fi | \
    text_to_int.py $vocab | gzip -c >$dir/$int.txt.gz ) 2>$dir/log/$int.log || touch $dir/.error &
  if ! $parallel; then
    wait
    if [ -f $dir/.error ]; then
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
