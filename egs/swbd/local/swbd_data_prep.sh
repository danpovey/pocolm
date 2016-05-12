#!/bin/bash

mkdir -p data/swbd1
if [ ! -d data/swbd1/swb_ms98_transcriptions ]; then
  cd data/swbd1
  echo " *** Downloading transcriptions ***"
  wget http://www.openslr.org/resources/5/switchboard_word_alignments.tar.gz ||
  wget http://www.isip.piconepress.com/projects/switchboard/releases/switchboard_word_alignments.tar.gz
  tar -xf switchboard_word_alignments.tar.gz
  cd ../..
else
  echo "Data already exists in data/swb_ms98_transcriptions, not downloading it."
fi


# (1a) Transcriptions preparation
# make basic transcription file (add segments info)
# **NOTE: In the default Kaldi recipe, everything is made uppercase, while we
# make everything lowercase here. This is because we will be using SRILM which
# can optionally make everything lowercase (but not uppercase) when mapping
# LM vocabs.
awk '{
       name=substr($1,1,6); gsub("^sw","sw0",name); side=substr($1,7,1);
       stime=$2; etime=$3;
       printf("%s-%s_%06.0f-%06.0f",
              name, side, int(100*stime+0.5), int(100*etime+0.5));
       for(i=4;i<=NF;i++) printf(" %s", $i); printf "\n"
}' data/swbd1/swb_ms98_transcriptions/*/*/*-trans.text  > data/swbd1/transcripts1.txt

# test if trans. file is sorted
export LC_ALL=C;
sort -c data/swbd1/transcripts1.txt || exit 1; # check it's sorted.


# Remove SILENCE, <B_ASIDE> and <E_ASIDE>.

# Note: we have [NOISE], [VOCALIZED-NOISE], [LAUGHTER], [SILENCE].
# removing [SILENCE], and the <B_ASIDE> and <E_ASIDE> markers that mark
# speech to somone; we will give phones to the other three (NSN, SPN, LAU).
# There will also be a silence phone, SIL.
# **NOTE: modified the pattern matches to make them case insensitive
cat data/swbd1/transcripts1.txt \
  | perl -ane 's:\s\[SILENCE\](\s|$):$1:gi;
               s/<B_ASIDE>//gi;
               s/<E_ASIDE>//gi;
               print;' \
  | awk '{if(NF > 1) { print; } } ' > data/swbd1/transcripts2.txt


# **NOTE: swbd1_map_words.pl has been modified to make the pattern matches
# case insensitive
local/swbd1_map_words.pl -f 2- data/swbd1/transcripts2.txt  > data/swbd1/text

# there is some acronym-mapping stuff after this that we do in the Kaldi ASR
# recipe, e.g. ACLU -> a.c.l.u., but I don't bother with that here, I don't think
# it would make any difference.  We should be careful when combining with fisher,
# though.


mkdir -p data/text
set -o errexit
export LC_ALL=C

heldout_sent=10000
cut -d' ' -f2- data/swbd1/text | tail -n +$heldout_sent > data/text/swbd1.txt
cut -d' ' -f2- data/swbd1/text | head -n $heldout_sent > data/text/dev.txt


