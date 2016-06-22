#!/bin/bash

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

num_dev_sentences=15000
# num_cantab_TEDLIUM_sentences=250000
vocab_size=100000

time_start=`date +%s`
if [ ! -d cantab-TEDLIUM ]; then
    echo "Downloading \"http://cantabresearch.com/cantab-TEDLIUM.tar.bz2\". "
    wget --no-verbose --output-document=- http://cantabresearch.com/cantab-TEDLIUM.tar.bz2 | bzcat | tar --extract --file=- || exit 1
    gzip cantab-TEDLIUM/cantab-TEDLIUM-pruned.lm3
    gzip cantab-TEDLIUM/cantab-TEDLIUM-unpruned.lm4
fi

if [ ! -d data ]; then
  mkdir data
  mkdir data/text
fi

head -n $num_dev_sentences < cantab-TEDLIUM/cantab-TEDLIUM.txt | sed 's/ <\/s>//g'  > data/text/dev.txt
tail -n +$(($num_dev_sentences+1)) < cantab-TEDLIUM/cantab-TEDLIUM.txt | sed 's/ <\/s>//g'  > data/text/cantap_tedlium.txt
time_download=`date +%s`

get_word_counts.py data/text data/word_counts

# decide on the vocabulary.
word_counts_to_vocab.py --num-words=${vocab_size} data/word_counts  > data/vocab_${vocab_size}.txt

# local/srilm_baseline.sh

prepare_int_data.sh data/text data/vocab_${vocab_size}.txt data/int_${vocab_size}
time_preprocessing=`date +%s`
time_optimization[2]=`date +%s`
# local/self_test.sh

for order in 3 4 5; do

  get_counts.py data/int_${vocab_size} ${order} data/counts_${vocab_size}_${order}

  ratio=10
  splits=5
  subset_count_dir.sh data/counts_${vocab_size}_${order} ${ratio} data/counts_${vocab_size}_${order}_subset${ratio}

  mkdir -p data/optimize_${vocab_size}_${order}_subset${ratio}

  optimize_metaparameters.py --progress-tolerance=2.0e-04 --num-splits=${splits} \
    data/counts_${vocab_size}_${order}_subset${ratio} data/optimize_${vocab_size}_${order}_subset${ratio}

  optimize_metaparameters.py --warm-start-dir=data/optimize_${vocab_size}_${order}_subset${ratio} \
    --progress-tolerance=1.0e-04 --num-splits=${splits} \
    data/counts_${vocab_size}_${order} data/optimize_${vocab_size}_${order}

  make_lm_dir.py --num-splits=${splits} data/counts_${vocab_size}_${order} \
     data/optimize_${vocab_size}_${order}/final.metaparams data/lm_${vocab_size}_${order}
  time_optimization[${order}]=`date +%s`
done

time_generating_arpa[2]=`date +%s`
for order in 3 4 5; do
  mkdir -p data/arpa
  format_arpa_lm.py data/lm_${vocab_size}_${order} | gzip -c > data/arpa/${vocab_size}_${order}gram_unpruned.arpa.gz
  time_generating_arpa[${order}]=`date +%s`
done

echo "Time for downloading: `expr ${time_preprocessing} - ${time_start}` "
echo "Time for preprocessing: `expr ${time_preprocessing} - ${time_download}` "

echo "For 3-gram model: optimizing takes `expr ${time_optimization[3]} - ${time_optimization[2]}` generating arpa takes `expr ${time_generating_arpa[3]} - ${time_generating_arpa[2]}`"
echo "For 4-gram model: optimizing takes `expr ${time_optimization[4]} - ${time_optimization[3]}` generating arpa takes `expr ${time_generating_arpa[4]} - ${time_generating_arpa[3]}`"
echo "For 5-gram model: optimizing takes `expr ${time_optimization[5]} - ${time_optimization[4]}` generating arpa takes `expr ${time_generating_arpa[5]} - ${time_generating_arpa[4]}`"

echo "TOTAL time takes: `expr ${time_generating_arpa[5]} - ${time_start}`"


# With 15000 dev sentences and 250000 cantap-TEDLIUM sentences:
# notes on SRILM baselines, from local/srilm_baseline.sh (without gtnmin):
# 3-gram: ppl= 182.372 ppl1=241.456
# 4-gram: ppl= 177.407 ppl1=234.534
# 5-gram: ppl= 177.008 ppl1=233.977
# Total time takes: 223 seconds

# notes on SRILM baselines, from local/srilm_baseline.sh (with gtnmin):
# 3-gram: ppl= 180.391 ppl1=238.692
# 4-gram: ppl= 174.423 ppl1=230.377
# 5-gram: ppl= 173.693 ppl1=229.362
# Total time takes: ~223 seconds

# pocolm perplexities:
# 3-gram: ppl=177.435 ppl1=234.572
# 4-gram: ppl=170.685 ppl1=225.178
# 5-gram: ppl=169.837 ppl1=223.998

# Time for downloading: 3
# Time for preprocessing: 2
# For 3-gram model: optimizing takes 39 generating arpa takes 21
# For 4-gram model: optimizing takes 68 generating arpa takes 55
# For 5-gram model: optimizing takes 103 generating arpa takes 99
# TOTAL time takes (pocolm): 388 seconds

# With 15000 sentences for dev set and all the rest 7847461 sentences for training set:
# notes on SRILM baselines, from local/srilm_baseline.sh (with gtnmin):
# 3-gram: ppl= 121.415 ppl1=157.077
# 4-gram: ppl= 104.891 ppl1=134.639
# 5-gram: ppl= 100.313 ppl1=128.454
# Total time takes: ~1 hour

# pocolm perplexities:
# 3-gram: ppl=120.36 ppl1=155.6
# 4-gram: ppl=103.167 ppl1=132.308

# Time for downloading: 71
# Time for preprocessing: 40
# For 3-gram model: optimizing takes 4989 generating arpa takes 268
# For 4-gram model: optimizing takes 13727 generating arpa takes 993
# For 5-gram model: optimizing takes 16234 generating arpa takes 895
# TOTAL time takes (pocolm): 37177 seconds / ~10.3 hours
