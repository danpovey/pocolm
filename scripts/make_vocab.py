#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings
from collections import defaultdict

parser = argparse.ArgumentParser(description="Creates a vocabulary file from a 'counts' directory "
                                 "as created by get_counts.py and a set of weights as created by "
                                 "get_unigram_weights.py.  A vocabulary file has the same format as "
                                 "a 'symbols' file from OpenFST, i.e. each line is 'word integer-symbol'."
                                 "However, it is necessary that the BOS, EOS and unknown-word (normally "
                                 "<s>, </s> and <unk>), be give symbols 1, 2 and 3 respecively.  You "
                                 "may use this script to generate the file, or generate it manually.",
                                 epilog="The vocabulary file is written to the standard output.")


parser.add_argument("--num-words", type=int, default=-1,
                    help="If specified, the maximum number of words to include "
                    "in the vocabulary.  If not specified, all words will be included.")
parser.add_argument("--weights",
                    help="File with weights for each data-source (except dev), in the "
                    "same format as from get_unigram_weights.py, i.e. each line has "
                    "'corpus-name weight'.  By default, ")

parser.add_argument("count_dir",
                    help="Directory in which to look for counts (see get_counts.py)");




args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if os.system("validate_text_dir.py " + args.text_dir) != 0:
    sys.exit(1)


if not os.path.exists(args.count_dir):
    os.mkdir(args.count_dir)

def ProcessFile(text_file, counts_file):
    try:
        f = open(text_file, "r");
    except:
        sys.exit("Failed to open {0} for reading".format(text_file))
    word_to_count = defaultdict(int)
    for line in f:
        for word in line.split():
            word_to_count[word] += 1
    f.close()
    try:
        cf = open(counts_file, "w");
    except:
        sys.exit("Failed to open {0} for writing".format(text_file))
    for word, count in word_to_count.items():
        print("{0} {1}".format(count, word), file=cf);
    cf.close()

num_files_processed = 0;

for f in os.listdir(args.text_dir):
    if f.endswith(".txt"):
        text_path = args.text_dir + os.sep + f
        counts_path = args.count_dir + os.sep + f[:-4] + ".counts"
        ProcessFile(text_path, counts_path)
        num_files_processed += 1

num_files_in_dest = 0;
for f in os.listdir(args.count_dir):
    if f.endswith(".counts"):
        num_files_in_dest += 1
    else:
        sys.exit("Text directory should not contain extra files: " + f)

if num_files_in_dest > num_files_processed:
    sys.exit("It looks like your destination directory " + args.count_dir +
             "contains some extra counts files.  Please clean up.");

print("Created {0} .counts files in {1}".format(num_files_processed,
                                                args.count_dir),
      file=sys.stderr);

