#!/bin/bash

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

prepare_int_data.sh data/text data/vocab_40k.txt data/int_40k

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
done

# order 3: optimize_metaparameters.py: final perplexity without barrier function was -4.35843985217 (perplexity: 78.1351369165)
# order 4: optimize_metaparameters.py: final perplexity without barrier function was -4.30864952578 (perplexity: 74.3400268301)
# order 5: optimize_metaparameters.py: final perplexity without barrier function was -4.30108107162 (perplexity: 73.7795115375)

# note, the perplexities from SRILM-estimated language models from orders 3 and
# 4 are (from local/srilm_baseline.sh), 79.9856 and 76.8235 respectively.

