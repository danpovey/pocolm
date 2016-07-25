#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh
fold_dev_opt=
# If you want to fold the dev-set in to the 'swbd1' set to produce the final
# model, un-comment the following line.  For use in the Kaldi example script for
# ASR, this isn't suitable because the 'dev' set is the first 10k lines of the
# switchboard data, which we also use as dev data for speech recognition
# purposes.

#fold_dev_opt="--fold-dev-into=swbd1"

get_word_counts.py data/text data/text/word_counts
get_unigram_weights.py data/text/word_counts > data/text/unigram_weights


# decide on the vocabulary.
word_counts_to_vocab.py --num-words=20000 data/text/word_counts  > data/vocab_20k.txt

# local/srilm_baseline.sh

prepare_int_data.py data/text data/vocab_20k.txt data/int_20k

# local/self_test.sh

for order in 3 4 5; do

  get_counts.sh data/int_20k ${order} data/counts_20k_${order}

  ratio=10
  splits=5
  subset_count_dir.sh data/counts_20k_${order} ${ratio} data/counts_20k_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=1.0e-05 --num-splits=${splits} \
    data/counts_20k_${order}_subset${ratio} data/optimize_20k_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=data/optimize_20k_${order}_subset${ratio} \
      --progress-tolerance=1.0e-03 --gradient-tolerance=0.01 --num-splits=${splits} \
    data/counts_20k_${order} data/optimize_20k_${order}

  make_lm_dir.py $fold_dev_opt --num-splits=${splits} --keep-splits=true data/counts_20k_${order} \
     data/optimize_20k_${order}/final.metaparams data/lm_20k_${order}

  mkdir -p data/arpa
  format_arpa_lm.py data/lm_20k_${order} | gzip -c > data/arpa/20k_${order}gram_unpruned.arpa.gz

  # example of pruning.  note: the threshold can be less than or more than one.
  get_data_prob.py data/text/dev.txt data/lm_20k_${order} 2>&1 | grep -F '[perplexity'
  for threshold in 1.0 0.25; do
    prune_lm_dir.py data/lm_20k_${order} $threshold data/lm_20k_${order}_prune${threshold} 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py data/text/dev.txt data/lm_20k_${order}_prune${threshold} 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py data/lm_20k_${order}_prune${threshold} | gzip -c > data/arpa/20k_${order}gram_prune${threshold}.arpa.gz

  done

# example of pruning by size.
  total_num_ngram=`get_ngram_num.py data/lm_20k_${order} 2>/dev/null`
  threshold=0.2
  for proportion in 0.1; do
    size=$(echo $proportion $total_num_ngram | awk '{ printf "%.0f", $1 * $2 }')
    prune_lm_dir.py --target-size=${size} data/lm_20k_${order} $threshold data/lm_20k_${order}_prune${size} 2>&1 | tail -n 5 | head -n 3
    get_data_prob.py data/text/dev.txt data/lm_20k_${order}_prune${size} 2>&1 | grep -F '[perplexity'

    format_arpa_lm.py data/lm_20k_${order}_prune${size} | gzip -c > data/arpa/20k_${order}gram_prune${size}.arpa.gz

  done
done


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
