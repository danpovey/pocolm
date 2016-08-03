#!/usr/bin/env python

# We're using python 3.x style print but want it to work in python 2.x.
from __future__ import print_function
import re, os, argparse, sys, math, warnings, operator
from collections import defaultdict


parser = argparse.ArgumentParser(description="Creates a vocabulary file from a 'counts' directory "
                                 "as created by get_counts.py and a set of weights as created by "
                                 "get_unigram_weights.py.  A vocabulary file has the same format as "
                                 "a 'symbols' file from OpenFST, i.e. each line is 'word integer-symbol'."
                                 "However, it is necessary that the BOS, EOS and unknown-word (normally "
                                 "<s>, </s> and <unk>), be give symbols 1, 2 and 3 respecively.  You "
                                 "may use this script to generate the file, or generate it manually."
                                 "The vocabulary file is written to the standard output.",
                                 epilog="See also wordlist_to_vocab.py",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--num-words', type=int,
                    help='If specified, the maximum number of words to include '
                    'in the vocabulary.  If not specified, all words will be included.')
parser.add_argument('--weights',
                    help="File with weights for each data-source (except dev), in the "
                    "same format as from get_unigram_weights.py, i.e. each line has "
                    "'corpus-name weight'.  By default, no weighting is used.")
parser.add_argument('--fold-dev-into', type=str,
                    help='If supplied, the name of data-source into which to fold the '
                    'counts of the dev data for purposes of vocabulary estimation '
                    '(typically the same data source from which the dev data was '
                    'originally excerpted).');
parser.add_argument('--unk-symbol', type=str, default='<unk>',
                    help='Written form of the unknown-word symbol, normally <unk> '
                    'or <UNK>.  Will not normally appear in the text data.')
parser.add_argument('--bos-symbol', type=str, default='<s>',
                    help='Written form of the beginning-of-sentence marker, normally <s> '
                    'or <S>.  Appears in the ARPA file but should not appear in the '
                    'text data.')
parser.add_argument('--eos-symbol', type=str, default='</s>',
                    help='Written form of the beginning-of-sentence marker, normally </s> '
                    'or <S>.  Appears in the ARPA file but should not appear in the '
                    'text data.')
parser.add_argument('--epsilon-symbol', type=str, default='<eps>',
                    help='Written form of label used for word-index zero, normally <eps>, '
                    'for compatibility with OpenFst.  This is never used to represent an '
                    'actual word, and if this appears in your text data it will be mapped '
                    'to the unknown-word symbol.  Override this at your own risk.')
parser.add_argument('count_dir',
                    help='Directory in which to look for counts (see get_counts.py)');

args = parser.parse_args()


# read in the weights.
name_to_weight = { }
if args.weights != None:
    f = open(args.weights, 'r')
    num_weights_read = 0
    for line in f:
        try:
            [ name, weight ] = line.split();
            weight = float(weight)  # check it's a float.
            if weight < 1.0e-10:  # this will ensure we get the vocab size we
                                  # wanted, even if some weights were estimated
                                  # as zero.
                weight = 1.0e-10
                print('word_counts_to_vocab.py: warning: flooring weight for {0} to {1}'.format(
                        name, weight), file=sys.stderr)
            name_to_weight[name] = weight
        except Exception as e:
            print(str(e), file=sys.stderr)
            sys.exit('word_counts_to_vocab.py: bad line {0} in weights file {1}'.format(
                    line[0:-1], args.weights))
        num_weights_read += 1
    if num_weights_read == 0:
        sys.exit('word_counts_to_vocab.py: empty weights file ' + args.weights)
    f.close()

# map from word to weighted count.
word_to_weighted_count = defaultdict(float)

saw_counts_with_weight = False
saw_counts_without_weight = False

num_counts_files = 0

for name in os.listdir(args.count_dir):
    if name.endswith('.counts'):
        num_counts_files += 1
        # read the counts.
        name_no_suffix = name[:-7]
        if name_no_suffix == 'dev':
            if args.fold_dev_into == None:
                continue  # don't include the dev counts unless we're told to
                          # fold them into some other data source.
            else:
                name_no_suffix = args.fold_dev_into
        if name_no_suffix in name_to_weight:
            weight = name_to_weight[name_no_suffix]
            saw_counts_with_weight = True
        else:
            weight = 1.0
            saw_counts_without_weight = True
        counts_path = args.count_dir + os.sep + name
        f = open(counts_path, 'r')
        for line in f:
            try:
                [ count, word ] = line.split()
                count = int(count) # just check that it's an integer.
                word_to_weighted_count[word] += count * weight
            except Exception as e:
                print(str(e), file=sys.stderr)
                sys.exit('word_counts_to_vocab.py: bad line in counts file {0}: {1}'.format(
                        counts_path, line[:-1]))
        f.close()

# note: if weights are provided, we expect 1 more counts files than the
# number of weights, due to the 'dev.counts'.
if ((saw_counts_with_weight and saw_counts_without_weight) or
    ((args.weights != None) and (num_counts_files != len(name_to_weight) + 1))):
    sys.exit('word_counts_to_vocab.py: it looks like the names in the weights file {0} '
             'do not match the files in the counts directory {1}'.format(
            args.weights, args.count_dir))

# deal with BOS, EOS and UNK, and <eps>; we can
# ensure the correct ordering by adding counts larger than the max.
# this part prints warnings if these were present in the raw counts.

max_weighted_count = max(word_to_weighted_count.values())

if args.epsilon_symbol in word_to_weighted_count:
    print('word_counts_to_vocab.py: warning: epsilon symbol {0} appears in the text. '
          ' It will be replaced by {1} during data preparation.'.format(
            args.epsilon_symbol, args.unk_symbol), file=sys.stderr)
word_to_weighted_count[args.epsilon_symbol] = 5.0 * max_weighted_count;

if args.bos_symbol in word_to_weighted_count:
    print('word_counts_to_vocab.py: severe warning: beginning-of-sentence symbol {0}'
          ' appears in the text. It will be replaced by {1} during data '
          'preparation.'.format(args.bos_symbol, args.unk_symbol), file=sys.stderr)
word_to_weighted_count[args.bos_symbol] = 4.0 * max_weighted_count;

if args.eos_symbol in word_to_weighted_count:
    print('word_counts_to_vocab.py: severe warning: end-of-sentence symbol {0}'
          ' appears in the text. It will be replaced by {1} during data '
          'preparation.'.format(args.eos_symbol, args.unk_symbol), file=sys.stderr)
word_to_weighted_count[args.eos_symbol] = 3.0 * max_weighted_count;

if args.unk_symbol in word_to_weighted_count:
    print('word_counts_to_vocab.py: mild warning: unknown-word symbol {0} appears in the text. '
          'Make sure you know what you are doing.'.format(args.unk_symbol), file=sys.stderr)
word_to_weighted_count[args.unk_symbol] = 2.0 * max_weighted_count;


sorted_list = sorted(word_to_weighted_count.items(),
                     key=operator.itemgetter(1), reverse=True)

if args.num_words != None and len(sorted_list) > args.num_words + 1:
    print('word_counts_to_vocab.py: you specified --num-words={0} so limiting the '
          'vocabulary from {1} to {0} words based on {3}count.'.format(
            args.num_words, len(sorted_list) - 1, args.num_words,
            ("weighted " if args.weights != None else "")), file=sys.stderr)
    sorted_list = sorted_list[0:args.num_words + 1]

# Here is where we produce the output of this program; it goes to the standard
# output.
index = 0
for [word,count] in sorted_list:
    print(word, index)
    index += 1

print('word_counts_to_vocab.py: created vocabulary with {0} entries'.format(len(sorted_list) - 1),
      file=sys.stderr)
