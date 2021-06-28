#!/usr/bin/env python3

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import sys

parser = argparse.ArgumentParser(description="Validates meta-parameter file as "
                                 "produced by initialize_metaparameters.py and "
                                 "other scripts",
                                 epilog="e.g. validate_metaparameters.py 10.metaparams")

parser.add_argument("--ngram-order", type=int,
                    help="The N-gram order of your final LM (required)")
parser.add_argument("--num-train-sets", type=int,
                    help="The number of training sets (required)")

parser.add_argument("metaparameter_file",
                    help="Filename of metaparameter file to validate")

args = parser.parse_args()


if args.ngram_order is None or args.ngram_order <= 1:
    sys.exit("validate_metaparameters.py: --ngram-order "
             "option must be supplied and >1")

if args.num_train_sets is None or args.num_train_sets <= 0:
    sys.exit("validate_metaparameters.py: --num-train-sets "
             "option must be supplied and >0")

if not os.path.exists(args.metaparameter_file):
    sys.exit("validate_metaparameters.py: Expected file {0}"
             " to exist".format(args.metaparameter_file))

try:
    f = open(args.metaparameter_file, "r", encoding="utf-8")
except:
    sys.exit("validate_metaparameters.py: error opening metaparameters file " +
             args.metaparameter_file)

for n in range(1, args.num_train_sets + 1):
    line = f.readline()
    try:
        [name, value] = line.split()
        value = float(value)
        assert name == "count_scale_{0}".format(n)
        assert value > 0.0 and value < 1.0
    except:
        sys.exit("validate_metaparameters.py: bad {0}'th line '{1}'"
                 "of metaparameters file {2}".format(n, line[0:-1],
                                                     args.metaparameter_file))

for o in range(2, args.ngram_order + 1):
    lines = []
    values = []
    for n in range(4):
        lines.append(f.readline())
    try:
        for n in range(4):
            [name, value] = lines[n].split()
            assert name == "order{0}_D{1}".format(o, n + 1)
            value = float(value)
            values.append(value)
            assert 1.0 > value and value > 0.0 and (n == 0 or value < values[n-1])
    except Exception as e:
        sys.exit("validate_metaparameters.py: bad values for {0}'th order "
                 "n-gram discounting parameters: in file {1}: {2}".format(
                     o, args.metaparameter_file, str(e)))

if f.readline() != '':
    sys.exit("validate_metaparameters.py: junk at end of "
             "metaparameters file {0}".format(args.metaparameter_file))
