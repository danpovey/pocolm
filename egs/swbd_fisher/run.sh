#!/bin/bash

# this script generates Pocolm-estimated language models with combined data sources
# from swbd1 and fisher. We aim to compare pocolm's method of combining data
# from multiple sources with SRILM's (check /local/pocolm_with_srilm_combination.sh).

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

# you have to have the Fisher corpus from LDC.
fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"

local/fisher_data_prep.sh $fisher_dirs

num_word=40000
lm_dir="data/lm/"
arpa_dir="data/arpa/"

for order in 3 4 5; do
  # decide on the vocabulary.
  # Note: you'd use --wordlist if you had a previously determined word-list
  # that you wanted to use.
  # Note: the following might be a more reasonable setting:
  # train_lm.py --num-word=${num_word} --num-splits=5 --warm-start-ratio=10 \
  #             --min-counts='fisher=2 swbd1=1' data/text ${order} ${lm_dir}
  train_lm.py --num-word=${num_word} --num-splits=5 --warm-start-ratio=10 \
              data/text ${order} ${lm_dir}
  unpruned_lm_dir=${lm_dir}/${num_word}_${order}.pocolm

  mkdir -p ${arpa_dir}
  format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > ${arpa_dir}/${num_word}_${order}gram_unpruned.arpa.gz

  # example of pruning.  note: the threshold can be less than or more than one.
  get_data_prob.py data/text/dev.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  for threshold in 1.0 2.0 4.0; do
    pruned_lm_dir=${lm_dir}/${num_word}_${order}_prune${threshold}.pocolm
    prune_lm_dir.py --final-threshold=${threshold} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py data/text/dev.txt ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py ${pruned_lm_dir} | gzip -c > data/arpa/${num_word}_${order}gram_prune${threshold}.arpa.gz

  done

  # example of pruning by size.
  size=250000
  pruned_lm_dir=${lm_dir}/${num_word}_${order}_prune${size}.pocolm
  prune_lm_dir.py --target-num-ngrams=${size} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 8 | head -n 6 | grep -v 'log-prob changes'
  get_data_prob.py data/text/dev.txt ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

  format_arpa_lm.py ${pruned_lm_dir} | gzip -c > data/arpa/${num_word}_${order}gram_prune${size}.arpa.gz

done

# local/srilm_baseline.sh

# the following does does some self-testing, including
# that the computed derivatives are accurate.
# local/self_test.sh

# perplexities from pocolm-estimated language models with pocolm's interpolation
# method from orders 3, 4, and 5 are:
# order 3: optimize_metaparameters.py: final perplexity without barrier function was -4.358818 (perplexity: 78.164689)
# order 4: optimize_metaparameters.py: final perplexity without barrier function was -4.309507 (perplexity: 74.403797)
# order 5: optimize_metaparameters.py: final perplexity without barrier function was -4.301741 (perplexity: 73.828181)

# note, the perplexities from pocolm-estimated language models with SRILM's
# interpolation from orders 3 and 4 are (from local/pocolm_with_srilm_combination.sh),
# 78.8449 and 75.2202 respectively.

# note, the perplexities from SRILM-estimated language models with SRILM's
# interpolation tool from orders 3 and 4 are (from local/srilm_baseline.sh),
# 78.9056 and 75.5528 respectively.
