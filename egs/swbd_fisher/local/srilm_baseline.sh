#!/bin/bash


# you'll have to edit the following to add SRILM to your path
# for your own setup.  This is not a core part of pocolm, it's
# just for comparison, so we don't add scripts to install SRILM
# for you.

export PATH=$PATH:/home/dpovey/kaldi-trunk/tools/srilm/bin/i686-m64/

if ! command -v ngram-count >&/dev/null; then
  echo "$0: you need to have SRILM on your path (look at the script for guidance)"
  exit 1
fi


for f in data/text/{swbd1,fisher,dev}.txt data/vocab_40k.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


mkdir data/srilm

tail -n +2 data/vocab_40k.txt  | awk '{print $1}' > data/srilm/wordlist


for order in 3 4; do
  echo "$0: estimating $order-gram baselines"

  for source in swbd1 fisher; do
    ngram-count -text data/text/${source}.txt -order ${order} -limit-vocab -vocab data/srilm/wordlist \
      -unk -map-unk "<unk>" -kndiscount -interpolate -lm data/srilm/$source.${order}g.kn.gz

    echo "Perplexity for $source $order-gram LM:"
    ngram -unk -lm data/srilm/$source.${order}g.kn.gz -ppl data/text/dev.txt

    ngram -unk -lm data/srilm/$source.${order}g.kn.gz -ppl data/text/dev.txt -debug 2 \
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
  ngram -unk -lm data/srilm/combined.${order}g.kn.gz -ppl data/text/dev.txt
done

# local/srilm_baseline.sh: estimating 3-gram baselines
# Perplexity for swbd1 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -249223 ppl= 87.7404 ppl1= 128.093
# Perplexity for fisher 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -259159 ppl= 104.875 ppl1= 155.435
# For 3-gram LMs, giving weight of 0.600461 to SWBD1 and 0.399539 to Fisher
# Perplexity for combined 3-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -244069 ppl= 79.9856 ppl1= 115.861

# local/srilm_baseline.sh: estimating 4-gram baselines
# Perplexity for swbd1 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -252017 ppl= 92.254 ppl1= 135.255
# Perplexity for fisher 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -262949 ppl= 112.26 ppl1= 167.34
# For 4-gram LMs, giving weight of 0.650063 to SWBD1 and 0.349937 to Fisher
# Perplexity for combined 4-gram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -247772 ppl= 85.4848 ppl1= 124.525
