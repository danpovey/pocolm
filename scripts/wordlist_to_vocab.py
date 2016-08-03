#!/usr/bin/env python

# We're using python 3.x style print but want it to work in python 2.x.
from __future__ import print_function
import re, os, argparse, sys, math, warnings, operator
from collections import defaultdict


parser = argparse.ArgumentParser(description="Creates a vocabulary file from a pre-existing word "
                                 "list.  The output (written to the standard output) is an OpenFst "
                                 "'symbols' file, with each line like 'word integer-symbol', e.g. "
                                 "'hello 124'.  This script ensures that certain 'special'"
                                 "symbols have the required integer id's, and that all the entries"
                                 "in the supplied word-list are assigned an integer id.",
                                 epilog="See also word_counts_to_vocab.py",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
parser.add_argument('wordlist',
                    help='File containing the word-list; each line should contain one word.')

args = parser.parse_args()

# read in the weights.
words = open(args.wordlist, "r").readlines()
if len(words) <= 1:
    sys.exit("wordlist_to_vocab.py: input word-list {0} has only {1} lines".format(
            args.wordlist, len(words)))

# remove newlines from the words, and do some validation.
for i in range(len(words)):
    word = words[i]
    a = word.split()
    if len(a) != 1:
        sys.exit("wordlist_to_vocab.py: bad line {0} in word-list file {1}".format(
                word[:-1], args.wordlist))
    words[i] = a[0]

# warn_words is words that we want to warn about even if they are not part of the
# word list.
warn_words = set(['<s>', '</s>', '<unk>', '<eps>', '<S>', '</S>', '<UNK>', '<EPS>'])

# 'output' is a list of words, where the integer id corresponds to the index.
output = [ args.epsilon_symbol, args.bos_symbol, args.eos_symbol, args.unk_symbol ]
num_special_symbols = len(output)

word_to_index = { }

for i in range(len(output)):
    special_word = output[i]
    if special_word in word_to_index:
        sys.exit("wordlist_to_vocab.py: it looks like some of the 'special words'"
                 " have the same value.  Check the command line options.")
    word_to_index[special_word] = i

for word in words:
    if word in word_to_index:
        existing_index = word_to_index[word]
        if existing_index >= num_special_symbols:
            # we silently ignore words in the word-list that are the
            # special symbols; for repeats we warn.
            print("wordlist_to_vocab.py: warning: repeated word {0} in "
                  "wordlist {1}, ignoring it.".format(word, args.wordlist),
                  file=sys.stderr)
        else:
            print("wordlist_to_vocab.py: warning: word {0} in wordlist {1} "
                  "is being treated as a special symbol, make sure this is what "
                  "you intended.".format(word, args.wordlist),
                  file=sys.stderr)
    else:
        if word in warn_words:
            print("wordlist_to_vocab.py: warning: your input wordlist {0} contains "
                  "the word '{1}'.  Make sure you know what you are doing.".format(
                    args.wordlist, word), file=sys.stderr)
        index = len(output)
        output.append(word)
        word_to_index[word] = index


# this is where the output happens.
for i in range(len(output)):
    print(output[i], i)

if len(output) < 5:
    sys.exit("wordlist_to_vocab.py: something went wrong.  The output vocab is too small.")

