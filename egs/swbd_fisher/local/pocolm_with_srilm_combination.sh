#!/bin/bash


# you'll have to edit the following to add SRILM to your path
# for your own setup.  This is not a core part of pocolm, it's
# just for comparison, so we don't add scripts to install SRILM
# for you.

set -e
export POCOLM_ROOT=$(cd ../..; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

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

mkdir -p data/poco_sri_combination
dir=data/poco_sri_combination

# prepare text and input data directories for swbd1 and fisher separately 
mkdir -p data/text/swbd1
mkdir -p data/text/fisher
for source in swbd1 fisher; do
  cp data/text/$source.txt data/text/$source/$source.txt
  cp data/text/dev.txt data/text/$source/dev.txt

  get_word_counts.py data/text/$source data/text/$source/word_counts
  get_unigram_weights.py data/text/$source/word_counts > data/text/$source/unigram_weights

  # decide on the vocabulary.
  # Note: you'd use wordlist_to_vocab.py if you had a previously determined word-list
  # that you wanted to use.
  
  mkdir -p $dir/$source
  prepare_int_data.py data/text/$source data/vocab_40k.txt $dir/$source/int_40k
done 

for order in 3 4; do

  for source in swbd1 fisher; do
  
    # Note: the following might be a more reasonable setting:
    # get_counts.py --min-counts='fisher=2 swbd1=1' data/int_40k ${order} data/counts_40k_${order}
    get_counts.py  $dir/$source/int_40k ${order} $dir/$source/counts_40k_${order}

    ratio=10
    splits=5
    subset_count_dir.sh $dir/$source/counts_40k_${order} ${ratio} $dir/$source/counts_40k_${order}_subset${ratio}

    optimize_metaparameters.py --progress-tolerance=1.0e-05 --num-splits=${splits} \
      $dir/$source/counts_40k_${order}_subset${ratio} $dir/$source/optimize_40k_${order}_subset${ratio}

    optimize_metaparameters.py --warm-start-dir=$dir/$source/optimize_40k_${order}_subset${ratio} \
      --progress-tolerance=1.0e-03 --gradient-tolerance=0.01 --num-splits=${splits} \
      $dir/$source/counts_40k_${order} $dir/$source/optimize_40k_${order}

    make_lm_dir.py --num-splits=${splits} --keep-splits=true $dir/$source/counts_40k_${order} \
     $dir/$source/optimize_40k_${order}/final.metaparams $dir/$source/lm_40k_${order}
    
    if [ ! -d data/arpa ]; then
      mkdir -p data/arpa
    fi 
    
    # format the individual LMs by pocolm into arpa files on swbd1 and fisher
    format_arpa_lm.py $dir/$source/lm_40k_${order} | gzip -c > data/arpa/$source.poco.${order}g.gz
  
    echo "Perplexity for pocolm $source ${order}-gram before pruning" 
    # get_data_prob.py data/text/dev.txt $dir/$source/lm_40k_${order} 2>&1 | grep -F '[perplexity' 
    ngram -order $order -unk -lm data/arpa/$source.poco.${order}g.gz -ppl data/text/dev.txt
    ngram -order $order -unk -lm data/arpa/$source.poco.${order}g.gz -ppl data/text/dev.txt -debug 2 \
      >& $dir/$source.poco.${order}g.ppl2

  done
  
  compute-best-mix $dir/{swbd1,fisher}.poco.${order}g.ppl2 >& $dir/swbd1_fisher_mix.${order}g.log

  weights_file=$dir/swbd1_fisher.${order}g.weights
  grep 'best lambda' $dir/swbd1_fisher_mix.${order}g.log | perl -e '
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
  # interpolate the languages models using pocolm on swbd1 and fisher with SRILM's interpolation tool  
  ngram -order $order \
      -lm data/arpa/swbd1.poco.${order}g.gz -lambda $swbd1_weight \
      -mix-lm data/arpa/fisher.poco.${order}g.gz \
      -unk -write-lm data/arpa/poco_sri_combination.${order}g.gz

  echo "Perplexity for poco-with-sri-combination $order-gram LM:"
  ngram -unk -order $order -lm data/arpa/poco_sri_combination.${order}g.gz -ppl data/text/dev.txt
done

rm -r data/text/swbd1
rm -r data/text/fisher
rm -r data/poco_sri_combination                  
