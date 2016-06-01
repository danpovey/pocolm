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

vocab_size=100000

for f in data/text/cantap_tedlium.txt data/text/dev.txt data/vocab_${vocab_size}.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


mkdir data/srilm

tail -n +2 data/vocab_${vocab_size}.txt  | awk '{print $1}' > data/srilm/wordlist

ngram-count -text data/text/cantap_tedlium.txt -order 3 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -interpolate -lm data/srilm/ted1.o3g.kn.gz

echo "Perplexity for cantap-TEDLIUM trigram LM:"
ngram -order 3 -unk -lm data/srilm/ted1.o3g.kn.gz -ppl data/text/dev.txt
ngram -order 3 -unk -lm data/srilm/ted1.o3g.kn.gz -ppl data/text/dev.txt -debug 2 >& data/srilm/3gram.ppl2

# Note, it probably says '0 OOVs' because we used the -unk option.
# Perplexity for TEDLIUM1 trigram LM:
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -663053 ppl= 182.372 ppl1= 241.456 (without gtnmin)
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -661662 ppl= 180.391 ppl1= 238.692 (with gtnmin)

ngram-count -text data/text/cantap_tedlium.txt -order 4 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -gt4min 1 -interpolate -lm data/srilm/ted1.o4g.kn.gz

echo "Perplexity for cantap-TEDLIUM 4-gram LM:"
ngram -order 4 -unk -lm data/srilm/ted1.o4g.kn.gz -ppl data/text/dev.txt
ngram -order 4 -unk -lm data/srilm/ted1.o4g.kn.gz -ppl data/text/dev.txt -debug 2 >& data/srilm/4gram.ppl2

# Perplexity for TEDLIUM1 4-gram lm:
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -659538 ppl= 177.407 ppl1= 234.534 (without gtnmin)
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -657377 ppl= 174.423 ppl1= 230.377 (with gtnmin)

ngram-count -text data/text/cantap_tedlium.txt -order 5 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -gt4min 1 -gt5min 1  -interpolate -lm data/srilm/ted1.o5g.kn.gz

echo "Perplexity for cantap-TEDLIUM 5-gram LM:"
ngram -order 5 -unk -lm data/srilm/ted1.o5g.kn.gz -ppl data/text/dev.txt
ngram -order 5 -unk -lm data/srilm/ted1.o5g.kn.gz -ppl data/text/dev.txt -debug 2 >& data/srilm/5gram.ppl2

# Perplexity for TEDLIUM1 5-gram lm:
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -659251 ppl= 177.008 ppl1= 233.977 (without gtnmin)
# file data/text/dev.txt: 15000 sentences, 278262 words, 0 OOVs
# 0 zeroprobs, logprob= -656843 ppl= 173.693 ppl1= 229.362 (with gtnmin)
