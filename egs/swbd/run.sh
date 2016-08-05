#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

num_word=20000
lm_dir="data/lm/"
arpa_dir="data/arpa/"

for order in 3 4 5; do
  train_lm.py --num-word=${num_word} --num-splits=5 --warm-start-ratio=10 \
              data/text ${order} ${lm_dir}
  unpruned_lm_dir=${lm_dir}/${num_word}_${order}.pocolm

  mkdir -p ${arpa_dir}
  format_arpa_lm.py ${unpruned_lm_dir} | gzip -c > ${arpa_dir}/${num_word}_${order}gram_unpruned.arpa.gz

  # example of pruning.  note: the threshold can be less than or more than one.
  get_data_prob.py data/text/dev.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  for threshold in 1.0 0.25; do
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

# local/self_test.sh

# notes on SRILM baselines, from local/srilm_baseline.sh:
# 3-gram: ppl= 84.6115
# 4-gram: ppl= 82.9717

# pocolm perplexities:
# 3-gram: final perplexity without barrier function was -4.4111061032 (perplexity: 82.3605123665)
# 4-gram: final perplexity without barrier function was -4.38100312372 (perplexity: 79.9181610124)
# 5-gram: final perplexity without barrier function was -4.37786357735 (perplexity: 79.6676476949)


# below I show how I verified the perplexities above using SRILM.  I also ran the same with
# higher debug settings, e.g. -debug 4, and looked at the output to check for badly-normalized
# output files.
# the following will work if you've run local/srilm_baseline.sh and installed SRILM,
# and put it on your path (change the path to somewhere suitable where you've installed it).
# export PATH=$PATH:/home/dpovey/kaldi-trunk/tools/srilm/bin/i686-m64/
# for order in 3 4 5; do ngram -order $order -unk -lm data/arpa/20k_${order}gram_unpruned.arpa.gz -ppl data/text/dev.txt ; done
#file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
#0 zeroprobs, logprob= -245699 ppl= 82.3605 ppl1= 119.597
#file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
#0 zeroprobs, logprob= -244022 ppl= 79.9182 ppl1= 115.755
#file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
#0 zeroprobs, logprob= -243847 ppl= 79.6676 ppl1= 115.362
