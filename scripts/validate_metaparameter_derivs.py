#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="Validates meta-parameter derivatives, "
                                 "as produced by get_objf_and_derivs.py-- chiefly "
                                 "checks that the derivative w.r.t. scaling all the scales "
                                 "by the same scaling factor is zero",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--ngram-order", type=int,
                    help="The N-gram order of your final LM (required)")
parser.add_argument("--num-train-sets", type=int,
                    help="The number of training sets (required)")

parser.add_argument("metaparameter_file",
                    help="Filename of metaparameters")
parser.add_argument("metaparameter_derivs",
                    help="Filename of derivatives w.r.t. metaparameters")

args = parser.parse_args()

if args.ngram_order == None or args.ngram_order <= 1:
    sys.exit("validate_metaparameter_derivs.py: --ngram-order "
             "option must be supplied and >1")

if args.num_train_sets == None or args.num_train_sets <= 0:
    sys.exit("validate_metaparameter_derivs.py: --num-train-sets "
             "option must be supplied and >0")

if not os.path.exists(args.metaparameter_file):
    sys.exit("validate_metaparameter_derivs.py: Expected file {0}"
             " to exist".format(args.metaparameter_file))

if not os.path.exists(args.metaparameter_derivs):
    sys.exit("validate_metaparameter_derivs.py: Expected file {0}"
             " to exist".format(args.metaparameter_derivs))

try:
    f = open(args.metaparameter_file, "r")
    deriv_f = open(args.metaparameter_derivs, "r")
except:
    sys.exit("validate_metaparameter_derivs.py: error opening {0} or {1}".format(
        args.metaparameter_file, args.metaparameter_derivs))

scaling_deriv = 0.0

for n in range(1, args.num_train_sets + 1):
    line = f.readline()
    deriv_line = deriv_f.readline()
    try:
        [ name, value ] = line.split()
        [ deriv_name, deriv ] = deriv_line.split()
        value = float(value)
        deriv = float(deriv)
        scaling_deriv += value * deriv
        assert name == deriv_name
        assert value > 0.0 and value < 1.0
    except:
        sys.exit("validate_metaparameter_derivs.py: bad {0}'th line '{1}' and '{2}'"
                 "of metaparameters and derivatives".format(n, line[0:-1],
                                                            deriv_line[0:-1]))

for o in range(2, args.ngram_order + 1):
    for n in range(4):
        line = f.readline()
        deriv_line = deriv_f.readline()
        try:
            [ name, value ] = line.split()
            [ deriv_name, deriv ] = deriv_line.split()
            value = float(value)
            deriv = float(deriv)
            assert name == deriv_name
            assert value > 0.0 and value < 1.0
        except:
            sys.exit("validate_metaparameter_derivs.py: bad lines '{0}' and '{1}'"
                     "of metaparameters and derivatives".format(line[0:-1],
                                                                deriv_line[0:-1]))

assert f.readline() == ''
assert deriv_f.readline() == ''
f.close()
deriv_f.close()

print("validate_metaparameter_derivs.py: deriv w.r.t. scaling "
      "is {0} (should be close to zero)".format(scaling_deriv),
      file=sys.stderr)

if abs(scaling_deriv) > 0.01:
    sys.exit("validate_metaparameter_derivs.py: excessively large deriv "
    "w.r.t. scaling: {0} ".format(scaling_deriv))
