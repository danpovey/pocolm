#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Transforms text data into integer form "
                                 "using a symbol table, e.g. turns line 'hello there' into "
                                 "'134 943'.  There are a couple of special cases: any "
                                 "word not in the word-list or equal to words numbered 0, 1 or 2 "
                                 "(normally <eps>, <s> and </s>) are treated as out-of-vocabulary "
                                 "words (OOV) and written as symbol 3 (normally '<unk>').",
                                 epilog="e.g. text_to_int.py words.txt < text > int_text",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("vocab_file",
                    help="Filename of vocabulary file, e.g. as produced by get_vocab.py");

args = parser.parse_args()

if not os.path.exists(args.vocab_file):
    sys.exit("validate_vocab.py: Expected file {0} to exist".format(args.text_dir))

word_to_index = {}

f = open(args.vocab_file, "r")

for line in f:
    try:
        [ word, index ] = line.split()
        word_to_index[word] = int(index)
    except:
        sys.exit("validate_vocab.py: bad line {0} in vocab file {1}".format(
                line[:-1], args.vocab_file))


num_words_total = 0
num_words_oov = 0
num_words_forbidden = 0

for line in sys.stdin:
    line_ints = []
    for word in line.split():
        num_words_total += 1
        if word in word_to_index:
            index = word_to_index[word]
            if index <= 2:
                num_words_forbidden += 1
                # the next line means that when we encounter symbols <eps>, <s>
                # or </s> in the text, we treat them the same as any
                # unknown-word.
                line_ints.append(str(3))
            else:
                line_ints.append(str(index))
        else:
            num_words_oov += 1
            line_ints.append(str(3))
    print(' '.join(line_ints))


print("text_to_int.py: converted {0} words, {1}% of which were OOV".format(
        num_words_total, (100.0*num_words_oov)/num_words_total), file=sys.stderr)

forbidden_words = []
if (num_words_forbidden != 0):
    for (word,index) in word_to_index.items():
        if index <= 2:
            forbidden_words.append(word)
        if index == 3:
            unk_word = word
    print("text_to_int.py: warning: encountered forbidden symbols ({0}) {1} times; "
          "converted them to {2}".format(",".join(forbidden_words),
                                       num_words_forbidden, unk_word),
          file=sys.stderr)

