#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

num_word=20000
lm_dir="data/lm/"
arpa_dir="data/arpa/"

max_memory='--max-memory=10G'
# If you do not want to set memory limitation for "sort", you can use
#max_memory=
# Choices for the max-memory can be:
# 1) integer + 'K', 'M', 'G', ... 
# 2) integer + 'b', meaning unit is byte and no multiplication
# 3) integer + '%', meaning a percentage of memory
# 4) integer, default unit is 'K'

fold_dev_opt=
# If you want to fold the dev-set in to the 'swbd1' set to produce the final
# model, un-comment the following line.  For use in the Kaldi example script for
# ASR, this isn't suitable because the 'dev' set is the first 10k lines of the
# switchboard data, which we also use as dev data for speech recognition
# purposes.

#fold_dev_opt="--fold-dev-into=swbd1"

bypass_metaparam_optim_opt=
# If you want to bypass the metaparameter optimization steps with specific metaparameters
# un-comment the following line, and change the numbers to some appropriate values.
# You can find the values from output log of train_lm.py.
# These example numbers of metaparameters is for 3-gram model running with train_lm.py.
# the dev perplexity should be close to the non-bypassed model.
#bypass_metaparam_optim_opt="--bypass-metaparameter-optimization=0.500,0.763,0.379,0.218,0.034,0.911,0.510,0.376,0.127"
# Note: to use these example parameters, you may need to remove the .done files
# to make sure the make_lm_dir.py be called and tain only 3-gram model
#for order in 3; do
#rm -f ${lm_dir}/${num_word}_${order}.pocolm/.done


for order in 3 4 5; do
  train_lm.py --num-word=${num_word} --num-splits=5 --warm-start-ratio=10 ${max_memory} \
              --keep-int-data=true ${fold_dev_opt} ${bypass_metaparam_optim_opt} \
              data/text ${order} ${lm_dir}
  unpruned_lm_dir=${lm_dir}/${num_word}_${order}.pocolm

  mkdir -p ${arpa_dir}
  format_arpa_lm.py ${max_memory} ${unpruned_lm_dir} | gzip -c > ${arpa_dir}/${num_word}_${order}gram_unpruned.arpa.gz

  # example of pruning.  note: the threshold can be less than or more than one.
  get_data_prob.py ${max_memory} data/text/dev.txt ${unpruned_lm_dir} 2>&1 | grep -F '[perplexity'
  for threshold in 1.0 0.25; do
    pruned_lm_dir=${lm_dir}/${num_word}_${order}_prune${threshold}.pocolm
    prune_lm_dir.py --final-threshold=${threshold} ${max_memory} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py ${max_memory} data/text/dev.txt ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py ${max_memory} ${pruned_lm_dir} | gzip -c > data/arpa/${num_word}_${order}gram_prune${threshold}.arpa.gz

  done

  # example of pruning by size.
  size=250000
  pruned_lm_dir=${lm_dir}/${num_word}_${order}_prune${size}.pocolm
  prune_lm_dir.py --target-num-ngrams=${size} ${max_memory} ${unpruned_lm_dir} ${pruned_lm_dir} 2>&1 | tail -n 8 | head -n 6 | grep -v 'log-prob changes'
  get_data_prob.py ${max_memory} data/text/dev.txt ${pruned_lm_dir} 2>&1 | grep -F '[perplexity'

  format_arpa_lm.py ${max_memory} ${pruned_lm_dir} | gzip -c > data/arpa/${num_word}_${order}gram_prune${size}.arpa.gz

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
