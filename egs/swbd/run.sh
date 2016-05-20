#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

local/swbd_data_prep.sh

get_word_counts.py data/text data/word_counts

# decide on the vocabulary.
counts_to_vocab.py --num-words=20000 data/word_counts  > data/vocab_20k.txt

# local/srilm_baseline.sh


prepare_int_data.sh data/text data/vocab_20k.txt data/int_20k

# local/self_test.sh

for order in 3 4 5; do

  get_counts.sh data/int_20k ${order} data/counts_20k_${order}

  ratio=10
  splits=5
  subset_count_dir.sh data/counts_20k_${order} ${ratio} data/counts_20k_${order}_subset${ratio}

  mkdir -p data/optimize_20k_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=2.0e-04 --num-splits=${splits} \
     --barrier-epsilon=1.0e-03 \
    data/counts_20k_${order}_subset${ratio} data/optimize_20k_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=data/optimize_20k_${order}_subset${ratio} \
     --barrier-epsilon=1.0e-03 \
    --progress-tolerance=1.0e-04 --num-splits=${splits} \
    data/counts_20k_${order} data/optimize_20k_${order}

  make_lm_dir.py --num-splits=${splits} data/counts_20k_${order} \
     data/optimize_20k_${order}/final.metaparams data/lm_20k_${order}
done

# notes on SRILM baselines, from local/srilm_baseline.sh:
# 3-gram: ppl= 84.6115
# 4-gram: ppl= 89.0114

# pocolm perplexities:
# 3-gram:  final perplexity without barrier function was -4.4109445077 (perplexity: 82.3472043536)
# 4-gram:  final perplexity without barrier function was -4.38097477755 (perplexity: 79.9158956707)
# 5-gram:  final perplexity without barrier function was -4.37773395894 (perplexity: 79.6573219703)
