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
# Note: you'd use wordlist_to_vocab.py if you had a previously determined word-list
# that you wanted to use.
word_counts_to_vocab.py --num-words=40000 data/text/word_counts  > data/vocab_40k.txt

# local/srilm_baseline.sh

prepare_int_data.py data/text data/vocab_40k.txt data/int_40k

# the following does does some self-testing, including
# that the computed derivatives are accurate.
# local/self_test.sh

for order in 3 4 5; do

  # Note: the following might be a more reasonable setting:
  # get_counts.py --min-counts='fisher=2 swbd1=1' data/int_40k ${order} data/counts_40k_${order}
  get_counts.py  data/int_40k ${order} data/counts_40k_${order}

  ratio=10
  splits=5
  subset_count_dir.sh data/counts_40k_${order} ${ratio} data/counts_40k_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=1.0e-05 --num-splits=${splits} \
    data/counts_40k_${order}_subset${ratio} data/optimize_40k_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=data/optimize_40k_${order}_subset${ratio} \
    --progress-tolerance=1.0e-03 --gradient-tolerance=0.01 --num-splits=${splits} \
    data/counts_40k_${order} data/optimize_40k_${order}

  make_lm_dir.py --num-splits=${splits} --keep-splits=true data/counts_40k_${order} \
     data/optimize_40k_${order}/final.metaparams data/lm_40k_${order}

  mkdir -p data/arpa
  format_arpa_lm.py data/lm_40k_${order} | gzip -c > data/arpa/poco_poco_combination.${order}g.gz

  echo "Perplexity for combined ${order}-gram: "
  get_data_prob.py data/text/dev.txt data/lm_40k_${order} 2>&1 | grep -F '[perplexity'

  echo "Ngram counts: "
  gunzip -c data/arpa/poco_poco_combination.${order}g.gz | head -n 50 | grep '^ngram' | cut -d '=' -f 2 | awk '{n +=$1}END{print n}'

 for threshold in 1.0 2.0 4.0; do
    prune_lm_dir.py data/lm_40k_${order} $threshold data/lm_40k_${order}_prune${threshold}
    get_data_prob.py data/text/dev.txt data/lm_40k_${order}_prune${threshold} 2>&1 | grep -F '[perplexity'
    mkdir -p data/arpa
    format_arpa_lm.py data/lm_40k_${order}_prune${threshold} | gzip -c > data/arpa/40k_${order}gram_prune${threshold}.arpa.gz
  done
done

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
