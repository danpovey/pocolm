#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

# you have to have the Fisher corpus from LDC.
fisher_dirs="/export/corpora3/LDC/LDC2004T19/fe_03_p1_tran/ /export/corpora3/LDC/LDC2005T19/fe_03_p2_tran/"

local/fisher_data_prep.sh $fisher_dirs

get_word_counts.py data/text data/word_counts

get_unigram_weights.py data/word_counts > data/weights

# decide on the vocabulary.
counts_to_vocab.py --num-words=40000 data/word_counts  > data/vocab_40k.txt

# local/srilm_baseline.sh

prepare_int_data.sh data/text data/vocab_40k.txt data/int_40k

# the following does does some self-testing, including
# that the computed derivatives are accurate.
# local/self_test.sh

for order in 3 4 5; do

  get_counts.sh data/int_40k ${order} data/counts_40k_${order}

  ratio=10
  splits=7  # must be coprime to 'ratio'.
  subset_count_dir.sh data/counts_40k_${order} ${ratio} data/counts_40k_${order}_subset${ratio}

  mkdir -p data/optimize_40k_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=2.0e-04 --num-splits=${splits} \
    data/counts_40k_${order}_subset${ratio} data/optimize_40k_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=data/optimize_40k_${order}_subset${ratio} \
    --progress-tolerance=1.0e-04 --num-splits=${splits} \
    data/counts_40k_${order} data/optimize_40k_${order}
done

# order 3: optimize_metaparameters.py: final perplexity without barrier function was -4.35843985217 (perplexity: 78.1351369165)
# order 4: optimize_metaparameters.py: final perplexity without barrier function was -4.30864952578 (perplexity: 74.3400268301)
# order 5: optimize_metaparameters.py: final perplexity without barrier function was -4.30108107162 (perplexity: 73.7795115375)

# note, the perplexities from SRILM-estimated language models from orders 3 and
# 4 are (from local/srilm_baseline.sh), 79.9856 and 85.4848 respectively.
