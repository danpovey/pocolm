#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Validates input directory containing text "
                                 "files from one or more data sources, including dev.txt.",
                                 epilog="See egs/swbd/run.sh for example.");

parser.add_argument("text_dir",
                    help="Directory in which to look for text data\n");

args = parser.parse_args()


if not os.path.exists(args.text_dir):
    sys.exit("Expected directory {0} to exist".format(args.text_dir))

if not os.path.exists("{0}/dev.txt".format(args.text_dir)):
    sys.exit("Expected file {0}/dev.txt to exist")


num_text_files = 0;

def SpotCheckTextFile(text_file):
    try:
        f = open(text_file, "r")
    except:
        sys.exit("Failed to open {0} for reading".format(text_file))
    found_nonempty_line = False
    for x in range(1,10):
        line = f.readline().strip("\n");
        if line is None:
            break
        words = line.split()
        if len(words) != 0:
            found_nonempty_line = True
            if (words[0] == "<s>" or words[0] == "<S>" or
                words[-1] == "</s>" or words[-1] == "</S>"):
                sys.exit("Found suspicious line '{0}' in file {1} (BOS and "
                         "EOS symbols are disallowed!)".format(line, text_file));
    if not found_nonempty_line:
        sys.exit("Input file {0} doesn't look right.".format(text_file));



for f in os.listdir(args.text_dir):
    if f.endswith(".txt"):
        full_path = args.text_dir + "/" + f
        if not os.path.isfile(full_path):
            sys.exit("Expected {0} to be a file.".format(full_path))
        SpotCheckTextFile(full_path)
        num_text_files += 1
    else:
        sys.exit("Text directory should not contain files with suffixes "
                 "other than .txt: " + f);

if num_text_files < 2:
    sys.exit("Directory {0} should contain at least one .txt file "
              "other than dev.txt.".format(args.text_dir));

