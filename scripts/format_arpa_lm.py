#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script turns a pocolm language model "
                                 "directory as created by make_lm_dir.py, into an ARPA-format "
                                 "language model.  The ARPA LM is written to the standard "
                                 "output and may be redirected as desired.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--temp-dir", type=str,
                    help="Temporary directory for use by 'sort'; if not provided, "
                    "we use the destination directory lm_dir")
parser.add_argument("lm_dir",
                    help="Directory of the source language model, as created "
                    "by make_lm_dir.py")

args = parser.parse_args()

if args.temp_dir == None:
    args.temp_dir = args.lm_dir

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");
# this will affect the program "sort" that we call.
os.environ['LC_ALL'] = 'C'
# this temporary directory will be used by "sort".
os.environ['TMPDIR'] = args.temp_dir

if os.system("validate_lm_dir.py " + args.lm_dir) != 0:
    sys.exit("format_arpa_lm.py: failed to validate input LM directory")

if not os.path.isdir(args.temp_dir):
    sys.exit("format_arpa_lm.py: expected directory {0} to exist.".format(
            args.temp_dir))

# read ngram order.
f = open(args.lm_dir + "/ngram_order");
ngram_order = int(f.readline())
f.close()

# work out num_words.  Note: this doesn't count epsilon; it's the
# same as the highest numbered word symbol.
line = subprocess.check_output([ 'tail', '-n', '1', args.lm_dir + '/words.txt' ])
try:
    [ last_word, num_words ] = line.split()
    num_words = int(num_words)
    assert num_words > 3
except:
    sys.exit("format_arpa_lm.py: error getting num-words from {0}/words.txt".format(
            args.lm_dir))

if not os.path.exists(args.lm_dir + "/num_splits"):
    # LM counts are in one file.
    command = ("float-counts-to-pre-arpa {ngram_order} {num_words} {lm_dir}/float.all | sort |"
               " pre-arpa-to-arpa {lm_dir}/words.txt".format(
            ngram_order = ngram_order, num_words = num_words, lm_dir = args.lm_dir))
else:
    # reading num_splits shouldn't fail, we validated the directory.
    num_splits = int(open(args.lm_dir + "/num_splits").readline())
    # create command line of the form:
    # sort -m <(command1) <(command2) ... <(commandN) | pre-arpa-to-arpa ...
    # we put it all inside bash -c, because the process substitution <(command)
    # won't always work in /bin/sh.
    command = ("bash -c 'sort -m " +  # sort -m merges already-sorted files.
               " ".join([ "<(float-counts-to-pre-arpa {opt} {ngram_order} {num_words} "
                          "{lm_dir}/float.all.{n} | sort)".format(
                    opt = ('--no-unigram' if n > 1 else ''),
                    ngram_order = ngram_order, num_words = num_words,
                    lm_dir = args.lm_dir, n = n) for n in range(1, num_splits + 1)]) +
               " | pre-arpa-to-arpa {lm_dir}/words.txt'".format(lm_dir = args.lm_dir))

print("format_arpa_lm.py: running " + command, file=sys.stderr)

ret = os.system(command)

if ret != 0:
    sys.exit("format_arpa_lm.py: command {0} exited with status {1}".format(
            command, ret))

print("format_arpa_lm.py: succeeded formatting ARPA lm from {0}".format(args.lm_dir),
      file=sys.stderr)


