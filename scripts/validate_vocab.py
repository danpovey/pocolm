#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Validates vocabulary file in OpenFst symbol "
                                 "table format",
                                 epilog="e.g. validate_vocab.py words.txt",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--num-words", type=int,
                    help="Number of words to expect in vocabulary (not counting epsilon), "
                    "should equal highest-numbered word.")

parser.add_argument("vocab_file",
                    help="Filename of vocabulary file")

args = parser.parse_args()

if not os.path.exists(args.vocab_file):
    sys.exit("validate_vocab.py: Expected file {0} to exist".format(args.vocab_file))

# likely_special_indexes is a set of pairs like [0, <eps>]
# these additions to the set just affect warnings being printed, it's
# only advisory.
likely_special_indexes = set()
likely_special_indexes.add((0,'<eps>'))
likely_special_indexes.add((0,'<EPS>'))
likely_special_indexes.add((1,'<s>'))
likely_special_indexes.add((1,'<S>'))
likely_special_indexes.add((2,'</s>'))
likely_special_indexes.add((2,'</S>'))
likely_special_indexes.add((3,'<unk>'))
likely_special_indexes.add((3,'<UNK>'))
likely_special_indexes.add((3,'<Unk>'))
# the following only affects a printed message:
default_special_indexes = ['<eps>', '<s>', '</s>', '<unk>']


f = open(args.vocab_file, "r")
num_lines = 0
for line in f:
    try:
        [ word, index ] = line.split()
        index = int(index)  # check that it's an integer.
        if index != num_lines:
            sys.exit("validate_vocab.py: line {0} is not in the expected "
                     "order in vocab file {1}".format(line[:-1], args.vocab_file))
        if index <= 3 and not (index, word) in likely_special_indexes:
            print("validate_vocab.py: warning: expected the word indexed {0} in {1} "
                  "to be '{2}', got '{3}'".format(index, args.vocab_file,
                                                default_special_indexes[index], word))

    except:
        sys.exit("validate_vocab.py: bad line {0} in vocab file {1}".format(
                line[:-1], args.vocab_file))
    num_lines += 1

if num_lines < 5:
    sys.exit("validate_vocab.py: file {0} is too short.".format(args.vocab_file))

if args.num_words != None and num_lines - 1 != args.num_words:
    sys.exit("validate_vocab.py: expected {0} words (--num-words={0} option, "
             "found {1} words, in {2}".format(args.num_words, num_lines - 1,
                                              args.vocab_file))

print("validate_vocab.py: validated file {0} with {1} entries.".format(
        args.vocab_file, num_lines - 1), file=sys.stderr)


