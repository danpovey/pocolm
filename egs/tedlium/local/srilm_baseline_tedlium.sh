#!/bin/bash


# you'll have to edit the following to add SRILM to your path
# for your own setup.  This is not a core part of pocolm, it's
# just for comparison, so we don't add scripts to install SRILM
# for you.

export PATH=$PATH:/netshr/kaldi/tools/srilm/bin/i686-m64/

if ! command -v ngram-count >&/dev/null; then
  echo "$0: you need to have SRILM on your path (look at the script for guidance)"
  exit 1
fi

# by default, no devset file input, will use the one in data/text/dev.txt.
dev_set_file="data/text/dev.txt"

# parse options.
if [ "$1" == "--dev-set-file" ]; then
  dev_set_file="$2"
    shift; shift;
fi

if [ $# != 0 ]; then
  echo "Usage:"
  echo "  $0 [options]"
  echo "e.g.:  $0"
  echo "This program build SRILM LMs on the data/text/train.txt corpus and compute perplexity on default dev set or optional file"
  echo
  echo "Options"
  echo "   --dev-set-file  <text file for dev set>"
  echo "                 Default: none."
  exit 1
fi

for f in data/text/train.txt $dev_set_file data/vocab.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


mkdir data/srilm

tail -n +2 data/vocab.txt  | awk '{print $1}' > data/srilm/wordlist

ngram-count -text data/text/train.txt -order 3 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -interpolate -lm data/srilm/lm.o3g.kn.gz

echo "Perplexity for trigram LM:"
ngram -order 3 -unk -lm data/srilm/lm.o3g.kn.gz -ppl $dev_set_file
ngram -order 3 -unk -lm data/srilm/lm.o3g.kn.gz -ppl $dev_set_file -debug 2 >& data/srilm/3gram.ppl2

ngram-count -text data/text/train.txt -order 4 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -gt4min 1 -interpolate -lm data/srilm/lm.o4g.kn.gz

echo "Perplexity for 4-gram LM:"
ngram -order 4 -unk -lm data/srilm/lm.o4g.kn.gz -ppl $dev_set_file
ngram -order 4 -unk -lm data/srilm/lm.o4g.kn.gz -ppl $dev_set_file -debug 2 >& data/srilm/4gram.ppl2

ngram-count -text data/text/train.txt -order 5 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -kndiscount -gt3min 1 -gt4min 1 -gt5min 1 -interpolate -lm data/srilm/lm.o5g.kn.gz

echo "Perplexity for 5-gram LM:"
ngram -order 5 -unk -lm data/srilm/lm.o5g.kn.gz -ppl $dev_set_file
ngram -order 5 -unk -lm data/srilm/lm.o5g.kn.gz -ppl $dev_set_file -debug 2 >& data/srilm/5gram.ppl2

# Below is baseline for cantab-TEDLIUM corpus using the text of the audio/text test set from Kaldi tedlium recipe
# consisting in 1155 sentences and 27512 words
# results based on the cantab-TEDLIUM dictionary 150000 words.
# --dev-set-file cantab-TEDLIUM/text-test
# Perplexity for trigram LM:
# file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
# 0 zeroprobs, logprob= -65622.7 ppl= 194.598 ppl1= 242.795
# Perplexity for 4-gram LM:
# file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
# 0 zeroprobs, logprob= -65059.6 ppl= 185.992 ppl1= 231.617
# Perplexity for 5-gram LM:
# file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
# 0 zeroprobs, logprob= -64946.7 ppl= 184.313 ppl1= 229.439

# SRILM Pruning impact on perplexity
# ORDER 3
# source file is 480 MB
ngram -unk -lm data/srilm/lm.o3g.kn.gz -prune 1e-7 -order 3 -write-lm data/srilm/lm.o3g.pruned1e7.gz
ngram -unk -lm data/srilm/lm.o3g.kn.gz -prune 5e-8 -order 3 -write-lm data/srilm/lm.o3g.pruned5e8.gz
ngram -unk -lm data/srilm/lm.o3g.kn.gz -prune 1e-8 -order 3 -write-lm data/srilm/lm.o3g.pruned1e8.gz
ngram -unk -lm data/srilm/lm.o3g.kn.gz -prune 1e-9 -order 3 -write-lm data/srilm/lm.o3g.pruned1e9.gz
ngram -unk -lm data/srilm/lm.o3g.kn.gz -prune 1e-10 -order 3 -write-lm data/srilm/lm.o3g.pruned1e10.gz
ngram -order 3 -unk -lm data/srilm/lm.o3g.pruned1e7.gz -ppl $dev_set_file
ngram -order 3 -unk -lm data/srilm/lm.o3g.pruned5e8.gz -ppl $dev_set_file
ngram -order 3 -unk -lm data/srilm/lm.o3g.pruned1e8.gz -ppl $dev_set_file
ngram -order 3 -unk -lm data/srilm/lm.o3g.pruned1e9.gz -ppl $dev_set_file
ngram -order 3 -unk -lm data/srilm/lm.o3g.pruned1e10.gz -ppl $dev_set_file
# pruning 1e-7 target file is 26.7 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -67368.9 ppl= 223.898 ppl1= 281.002
# pruning 5e-8 target file is 46.9 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -66739.2 ppl= 212.855 ppl1= 266.576
# pruning 1e-8 target file is 157.9 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65995.5 ppl= 200.512 ppl1= 250.489
# pruning 1e-8 target file is 368.5 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65670.6 ppl= 195.348 ppl1= 243.77
# pruning 1e-8 target file is 461.8 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65620.8 ppl= 194.568 ppl1= 242.757


# ORDER 4
# source file is 1.4 GB
ngram -unk -lm data/srilm/lm.o4g.kn.gz -prune 1e-7 -order 4 -write-lm data/srilm/lm.o4g.pruned1e7.gz
ngram -unk -lm data/srilm/lm.o4g.kn.gz -prune 1e-8 -order 4 -write-lm data/srilm/lm.o4g.pruned1e8.gz
ngram -unk -lm data/srilm/lm.o4g.kn.gz -prune 1e-9 -order 4 -write-lm data/srilm/lm.o4g.pruned1e9.gz
ngram -unk -lm data/srilm/lm.o4g.kn.gz -prune 1e-10 -order 4 -write-lm data/srilm/lm.o4g.pruned1e10.gz
ngram -order 4 -unk -lm data/srilm/lm.o4g.pruned1e7.gz -ppl $dev_set_file
ngram -order 4 -unk -lm data/srilm/lm.o4g.pruned1e8.gz -ppl $dev_set_file
ngram -order 4 -unk -lm data/srilm/lm.o4g.pruned1e9.gz -ppl $dev_set_file
ngram -order 4 -unk -lm data/srilm/lm.o4g.pruned1e10.gz -ppl $dev_set_file
# pruning 1e-7 target file is 26.8 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -67523 ppl= 226.686 ppl1= 284.649
# pruning 1e-8 target file is 183.4 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65483.1 ppl= 192.428 ppl1= 239.975
# pruning 1e-9 target file is 651.6 MB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65017.1 ppl= 185.358 ppl1= 230.795
# pruning 1e-10 target file is 1.1 GB
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -65028.4 ppl= 185.527 ppl1= 231.014

# Maxent LM model example order 4
# need for LIBLBFGS and re compile SRILM 1.7.1 - make clean first
# target file is 1.2 GB and run last 5583 seconds ...
ngram-count -text data/text/train.txt -order 4 -limit-vocab -vocab data/srilm/wordlist \
  -unk -map-unk "<unk>" -maxent -gt3min 1 -gt4min 1 -lm data/srilm/lm.o4g.hme.gz
ngram -order 4 -unk -maxent -lm data/srilm/lm.o4g.hme.gz -ppl $dev_set_file
#file cantab-TEDLIUM/text-test: 1155 sentences, 27512 words, 0 OOVs
#0 zeroprobs, logprob= -64640.3 ppl= 179.832 ppl1= 223.63

#same results for first 10k sentences of cantab text
#Perplexity for trigram LM:
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -409199 ppl= 123.354 ppl1= 159.872
#Perplexity for 4-gram LM:
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -396834 ppl= 106.65 ppl1= 137.145
#Perplexity for 5-gram LM:
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -392961 ppl= 101.899 ppl1= 130.714

# impact of pruning (same order as above)
# order 3
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -430412 ppl= 158.329 ppl1= 207.979
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -424208 ppl= 147.183 ppl1= 192.579
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -414996 ppl= 132.062 ppl1= 171.788
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -410353 ppl= 125.04 ppl1= 162.177
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -409340 ppl= 123.559 ppl1= 160.153

# order 4
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -431154 ppl= 159.718 ppl1= 209.902
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -409665 ppl= 124.033 ppl1= 160.8
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -400460 ppl= 111.3 ppl1= 143.453
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -397685 ppl= 107.724 ppl1= 138.6

# maxent order 4
#file data/text/dev.txt: 10000 sentences, 185681 words, 0 OOVs
#0 zeroprobs, logprob= -396226 ppl= 105.89 ppl1= 136.115




