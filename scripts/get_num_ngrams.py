#!/usr/bin/env python

# This script is inefficient, it should not be used by any other scripts under scripts/ dir
# Once we have someway to store the num-ngrams in pocolm lm dir, this script should
# only involve opening and reading a file under the lm dir.
#
# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, tempfile, threading
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script get the total number of n-grams "
                                 "in a pocolm 'lm-dir' (as validated by validate_lm_dir.py).")

parser.add_argument("lm_dir_in",
                    help="Source directory, for the input language model.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");
# this will affect the program "sort" that we call.
os.environ['LC_ALL'] = 'C'


if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    sys.exit("get_ngram_num.py: failed to validate input LM-dir")


num_splits = None

if os.path.exists(args.lm_dir_in + "/num_splits"):
    f = open(args.lm_dir_in + "/num_splits")
    num_splits = int(f.readline())
    f.close()

tot_num_gram = 0

def GetNumNgram(split_index):
    if split_index == None:
        command = "print-float-counts <{0}/float.all 2>&1 >/dev/null".format(
            args.lm_dir_in)
    else:
        command = "print-float-counts <{0}/float.all.{1} 2>&1 >/dev/null".format(
            args.lm_dir_in, split_index)
    print (command, file=sys.stderr)
    try:
        output = subprocess.check_output(command, shell = True)
    except:
        sys.exit("get_ngram_num.py: error running command: " + command)
    m = re.search('with ([0-9]+) individual n-grams', output)
    global tot_num_gram
    tot_num_gram += int(m.group(1))

if num_splits == None:
    GetNumNgram(None)
else:
    threads = []
    for split_index in range(1, num_splits + 1):
        threads.append(threading.Thread(target = GetNumNgram,
                                        args = [split_index]))
        threads[-1].start()
    for t in threads:
        t.join()


print(tot_num_gram, file=sys.stdout)
