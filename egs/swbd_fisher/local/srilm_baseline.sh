#!/bin/bash


# you'll have to edit the following to add SRILM to your path
# for your own setup.  This is not a core part of pocolm, it's
# just for comparison, so we don't add scripts to install SRILM
# for you.

export PATH=$PATH:/home/dpovey/kaldi-trunk/tools/srilm/bin/i686-m64/

num_word=40000
if [ ! -z $1 ]; then
  num_word=$1
fi

if ! command -v ngram-count >&/dev/null; then
  echo "$0: you need to have SRILM on your path (look at the script for guidance)"
  exit 1
fi


for f in data/text/{swbd1,fisher,dev}.txt data/lm/work/vocab_${num_word}.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


mkdir -p data/srilm

tail -n +2 data/lm/work/vocab_${num_word}.txt  | awk '{print $1}' > data/srilm/wordlist

for order in 3 4; do
  echo "$0: estimating $order-gram baselines"

  for source in swbd1 fisher; do
    ngram-count -order ${order} -text data/text/${source}.txt -limit-vocab -vocab data/srilm/wordlist \
      -unk -map-unk "<unk>" -kndiscount -gt3min 0 -gt4min 0 -interpolate -lm data/srilm/$source.${order}g.kn.gz

    echo "Perplexity for $source $order-gram LM:"
    ngram -order $order -unk -lm data/srilm/$source.${order}g.kn.gz -ppl data/text/dev.txt
    echo "Ngram counts for $source $order-gram LM before pruning:"
    gunzip -c data/srilm/$source.${order}g.kn.gz | head -n 50 | grep '^ngram' | cut -d '=' -f 2 | awk '{n +=$1}END{print n}'
    
    ngram -order $order -unk -lm data/srilm/$source.${order}g.kn.gz -ppl data/text/dev.txt -debug 2 \
      >& data/srilm/$source.${order}g.ppl2
    
  done
  compute-best-mix data/srilm/{swbd1,fisher}.${order}g.ppl2 >& data/srilm/swbd1_fisher_mix.${order}g.log

  weights_file=data/srilm/swbd1_fisher.${order}g.weights
  grep 'best lambda' data/srilm/swbd1_fisher_mix.${order}g.log | perl -e '
      $_=<>;
      s/.*\(//; s/\).*//;
      @A = split;
      die "Expecting 2 numbers; found: $_" if(@A!=2);
      print "$A[0]\n$A[1]\n";' > $weights_file

  swbd1_weight=$(head -n 1 $weights_file)
  fisher_weight=$(tail -n 1 $weights_file)
  echo "For ${order}-gram LMs, giving weight of ${swbd1_weight} to SWBD1 and ${fisher_weight} to Fisher"

  # note: we don't have to supply $fisher_weight to 'ngram', it's
  # implicit because the weights are supposed to sum to one.
  ngram -order $order \
      -lm data/srilm/swbd1.${order}g.kn.gz -lambda $swbd1_weight \
      -mix-lm data/srilm/fisher.${order}g.kn.gz \
      -unk -write-lm data/srilm/combined.${order}g.kn.gz

  echo "Perplexity for combined $order-gram LM:"
  ngram -unk -order $order -lm data/srilm/combined.${order}g.kn.gz -ppl data/text/dev.txt
  echo "Ngram counts:"
  gunzip -c data/srilm/combined.${order}g.kn.gz | head -n 50 | grep '^ngram' | cut -d '=' -f 2 | awk '{n +=$1}END{print n}'

done

# local/srilm_baseline.sh: estimating 3-gram baselines
# Perplexity for swbd1 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -248861 ppl= 87.1712 ppl1= 127.192
# Perplexity for fisher 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -259224 ppl= 104.998 ppl1= 155.632
# For 3-gram LMs, giving weight of 0.59544 to SWBD1 and 0.40456 to Fisher
# Perplexity for combined 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -243312 ppl= **78.9056** ppl1= 114.165

# local/srilm_baseline.sh: estimating 4-gram baselines
# Perplexity for swbd1 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -247874 ppl= 85.6401 ppl1= 124.77
# Perplexity for fisher 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -257773 ppl= 102.297 ppl1= 151.295
# For 4-gram LMs, giving weight of 0.578637 to SWBD1 and 0.421363 to Fisher
# Perplexity for combined 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -240893 ppl= **75.5528** ppl1= 108.914
