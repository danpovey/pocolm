#!/usr/bin/env bash

export POCOLM_ROOT=$(cd ../../; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts:$POCOLM_ROOT/src

rm -rf data/text
mkdir -p data/text

# the pattern *-*-*-*.cc only selects  files at the moment.
# this keeps the weights more distinct and makes the derivative testing easier.a
for f in ../../src/*-*-*-*.cc; do
  dest_name=$(echo $(basename $f) | sed 's:\.:-:')
  cp $f data/text/${dest_name}.txt
done

# define the dev set.
mv data/text/discount-counts-1gram-backward-cc.txt data/text/dev.txt

validate_text_dir.py data/text
