#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script takes an lm-dir, as produced by make_lm_dir.py, "
                                 "that should not have the counts split up into pieces, and it "
                                 "splits up the counts into a specified number of pieces. "
                                 "Output is the 'split' form of lm-dir, with float.all.{1,2,3...} and "
                                 "num_splits",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("lm_dir_in",
                    help="Source directory, for the input language model.")
parser.add_argument("num_splits", type=int,
                    help="Number of split-up pieces in the source lm-dir")
parser.add_argument("lm_dir_out",
                    help="Output directory where the language model is created.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

if args.num_splits <= 1:
    sys.exit("split_lm_dir.py: num_splits must be >1.")

if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    sys.exit("split_lm_dir.py: failed to validate input LM-dir")

if os.path.exists(args.lm_dir_in + "/num_splits"):
    sys.exit("split_lm_dir.py: input LM-dir is already split")

if not os.path.isdir(args.lm_dir_out):
    try:
        os.makedirs(args.lm_dir_out)
    except Exception as e:
        sys.exit("split_lm_dir.py: error creating directory " + args.lm_dir_out + ": " + str(e))

# copy some smallish informational files from the input to output directory.
for name in [ 'words.txt', 'ngram_order', 'num_ngrams', 'names', 'metaparameters', 'was_pruned' ]:
    src = args.lm_dir_in + "/" + name
    dest = args.lm_dir_out + "/" + name
    try:
        shutil.copy(src, dest)
    except:
        sys.exit("split_lm_dir.py: error copying {0} to {1}".format(src, dest))

command = ("split-float-counts " +
           ' '.join([ args.lm_dir_out + "/" + "float.all." + str(n)
                      for n in range(1, args.num_splits + 1) ]) +
           ' <' + args.lm_dir_in + "/float.all")

if os.system(command) != 0:
    sys.exit("split_lm_dir.py: error running command " + command)

f = open(args.lm_dir_out + "/num_splits", "w")
print(args.num_splits, file=f)
f.close()

if os.system("validate_lm_dir.py " + args.lm_dir_out) != 0:
    sys.exit("split_lm_dir.py: failed to validate output LM-dir")

print("split_lm_dir.py: split input LM-dir {0} into {1} pieces into directory {2}".format(
        args.lm_dir_in, args.num_splits, args.lm_dir_out))

