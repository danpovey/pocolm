#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import numpy as np

import re, os, argparse, sys, math, warnings, subprocess, shutil, threading
from collections import defaultdict
from subprocess import CalledProcessError

# we need to add the ./internal/ subdirectory to the pythonpath before importing
# 'bfgs'.
sys.path = [ os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal" ] + sys.path
import bfgs
# for ExitProgram, RunCommand and GetCommandStdout
from pocolm_common import *

parser = argparse.ArgumentParser(description="This script is a top level interface "
                                "for training unpruned language models.")
parser.add_argument("--order", type=int, default=3,
                    help="Order of ngram language models.")
parser.add_argument("--ratio", type=int, default=10,
                    help="Fraction 1/ratio of data counts.")
parser.add_argument("--num-splits", type=int, default=1,
                    help="Controls the number of parallel processes used to "
                    "get objective functions and derivatives.  If >1, then "
                    "we split the counts and compute these things in parallel.")
parser.add_argument("count_dir",
                    help="Directory in which to find counts of a certain order"
                    "of ngram language models.")
parser.add_argument("output_dir",
                    help="Directory of the intermediate results and final "
                    "ARPA-format language models.")

args = parser.parse_args()

# Add the script dir to the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])));

if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit("train_lm.py: validate_count_dir.py failed")

if args.num_splits < 1:
    sys.exit("train_lm.py: --num-splits must be >0.")
if args.num_splits > 1:
    if (os.system("split_count_dir.sh {0} {1}".format(
                args.count_dir, args.num_splits))) != 0:
        sys.exit("train_lm.py: failed to create split count-dir.")

# generate the subset of data counts 
subset_count_dir = args.count_dir + "_subset{0}".format(args.ratio)
if not os.path.exists(subset_count_dir):
    os.makedirs(subset_count_dir)
command = ("subset_count_dir.sh {0} {1} {2}".format(args.count_dir, args.ratio, 
           subset_count_dir))
if os.system(command) != 0:
    sys.exit("subset_count_dir.sh: failed to generate subset of data counts with"
             "ratio 1/{0}".format(args.ratio)) 
subprocess.check_call(command, shell = True)

# first pass metaparameters estimation on a sub dataset  
subset_optimize_dir = args.output_dir + "/optimize_{0}_subset{1}".format(
    args.order, args.ratio)
if not os.path.exists(subset_optimize_dir):
    os.makedirs(subset_optimize_dir)
command = ("optimize_metaparameters.py --progress-tolerance=1.0e-05 "
           "--num-splits={0} {1} {2}".format(args.num_splits, subset_count_dir,
           subset_optimize_dir))
if os.system(command) != 0:
    sys.exit("optimize_metaparameters.py: failed to initialize parameters on "
             "subset of data counts, command was: " + command)
subprocess.check_call(command, shell = True)

# optimize metaparameters on the entire dataset
optimize_dir = args.output_dir + "/optimize_{0}".format(args.order)
if not os.path.exists(optimize_dir):
    os.makedirs(optimize_dir)
command = ("optimize_metaparameters.py --warm-start-dir={0} --progress-tolerance"
           "=1.0e-03 --gradient-tolerance=0.01 --num-splits={1} {2} {3}".format(
           subset_optimize_dir, args.num_splits, args.count_dir, optimize_dir))
if os.system(command) != 0:
    sys.exit("optimize_metaparameters.py: failed to estimate parameters on entir "
             "data counts, command was: " + command)
subprocess.check_call(command, shell = True)

# create lm directory
lm_dir = args.output_dir + "/lm_20k_{0}".format(args.order)
if not os.path.exists(lm_dir):
    os.makedirs(lm_dir)
command = ("make_lm_dir.py --num-splits={0} --keep-splits=true {1} "
           "{2}/final.metaparams {3}".format(args.num_splits, args.count_dir,
           optimize_dir, lm_dir))
if os.system(command) != 0:
    sys.exit("make_lm_dir.py: failed to create LM directory, command was: " + command)
subprocess.check_call(command, shell = True)

print("train_lm.py: successfully created lm directory {0}.".format(lm_dir))
