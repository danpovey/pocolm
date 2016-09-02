#!/bin/bash


# you'll have to edit the following to add SRILM to your path
# for your own setup.  This is not a core part of pocolm, it's
# just for comparison, so we don't add scripts to install SRILM
# for you.

export PATH=$PATH:/home/dpovey/kaldi-trunk/tools/srilm/bin/i686-m64/

num_word=20000
if [ ! -z $1 ]; then
  num_word=$1
fi

if ! command -v ngram-count >&/dev/null; then
  echo "$0: you need to have SRILM on your path (look at the script for guidance)"
  exit 1
fi

for f in data/text/swbd1.txt data/text/dev.txt data/lm/work/vocab_${num_word}.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


mkdir data/srilm

tail -n +2 data/lm/work/vocab_${num_word}.txt  | awk '{print $1}' > data/srilm/wordlist

ngram-count -text data/text/swbd1.txt -order 3 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm data/srilm/sw1.o3g.kn.gz

echo "Perplexity for SWBD1 trigram LM:"
ngram -order 3 -unk -lm data/srilm/sw1.o3g.kn.gz -ppl data/text/dev.txt
ngram -order 3 -unk -lm data/srilm/sw1.o3g.kn.gz -ppl data/text/dev.txt -debug 2 >& data/srilm/3gram.ppl2

# Note, it probably says '0 OOVs' because we used the -unk option.
# Perplexity for SWBD1 trigram LM:
# file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
# 0 zeroprobs, logprob= -247201 ppl= 84.6115 ppl1= 123.146


ngram-count -text data/text/swbd1.txt -order 4 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -interpolate -lm data/srilm/sw1.o4g.kn.gz

echo "Perplexity for SWBD1 4-gram LM:"
ngram -order 4 -unk -lm data/srilm/sw1.o4g.kn.gz -ppl data/text/dev.txt
ngram -order 4 -unk -lm data/srilm/sw1.o4g.kn.gz -ppl data/text/dev.txt -debug 2 >& data/srilm/4gram.ppl2

#Perplexity for SWBD1 4-gram LM:
#file data/text/dev.txt: 10000 sentences, 118254 words, 0 OOVs
#0 zeroprobs, logprob= -246110 ppl= 82.9717 ppl1= 120.56

