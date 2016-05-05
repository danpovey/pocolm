#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Validates meta-parameter file as "
                                 "produced by get_initial_metaparameters.py and "
                                 "other scripts",
                                 epilog="e.g. validate_metaparameters.py 10.metaparams")

parser.add_argument("--ngram-order", type=int,
                    help="The N-gram order of your final LM (required)")
parser.add_argument("--num-train-sets", type=int,
                    help="The number of training sets (required)")

parser.add_argument("metaparameter_file",
                    help="Filename of metaparameter file to validate")

args = parser.parse_args()


if args.ngram_order == None or args.ngram_order <= 1:
    sys.exit("validate_metaparameters.py: --ngram-order "
             "option must be supplied and >1")

if args.num_train_sets == None or args.num_train_sets <= 0:
    sys.exit("validate_metaparameters.py: --num-train-sets "
             "option must be supplied and >0")

if not os.path.exists(args.metaparameter_file):
    sys.exit("validate_metaparameters.py: Expected file {0}"
             " to exist".format(args.metaparameter_file))

try:
    f = open(args.metaparameter_file, "r")
except:
    sys.exit("validate_metaparameters.py: error opening metaparameters file " +
             args.metaparameter_file)

for n in range(1, args.num_train_sets + 1):
    line = f.readline()
    try:
        [ name, value ] = line.split()
        value = float(value)
        assert name == "count_scale_{0}".format(n)
        assert value > 0.0 and value < 1.0
    except:
        sys.exit("validate_metaparameters.py: bad {0}'th line '{1}'"
                 "of metaparameters file {2}".format(n, line[0:-1],
                                                     args.metaparameter_file))

for o in range(2, args.ngram_order + 1):
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()
    try:
        [ name1, value1 ] = line1.split()
        [ name2, value2 ] = line2.split()
        [ name3, value3 ] = line3.split()
        value1 = float(value1)
        value2 = float(value2)
        value3 = float(value3)
        assert name1 == "order{0}_D1".format(o)
        assert name2 == "order{0}_D2".format(o)
        assert name3 == "order{0}_D3".format(o)
        assert 1.0 > value1 and value1 > value2 and value2 > value3 and value3 > 0.0
    except:
        sys.exit("validate_metaparameters.py: bad values for {0}'th order "
                 "n-gram discounting parameters: '{1}', '{2}', '{3}',"
                 " in file {4}".format(o, line1[0:-1], line2[0:-1], line3[0:-1],
                                       args.metaparameter_file))

if f.readline() != '':
    sys.exit("validate_metaparameters.py: junk at end of "
             "metaparameters file {0}".format(args.metaparameter_file))

