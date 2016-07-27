#!/bin/bash

if [ $# != 2 ]; then
  echo "Usage: $0 <fisher-dir1> <fisher-dir2>"
  echo "e.g.: $0 /export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"
  exit 1
fi

mkdir -p data/fisher

rm -f data/fisher/text0

for dir in $1 $2; do
  if [ ! -d $dir/data/trans ]; then
    echo "$0: expected directory $dir/data/trans to exist"
    exit 1;
  fi
  cat $dir/data/trans/*/*.txt |\
    grep -v ^# | grep -v ^$ | cut -d' ' -f4- >> data/fisher/text0
done

cat data/fisher/text0 | local/fisher_map_words.pl  > data/text/fisher.txt

