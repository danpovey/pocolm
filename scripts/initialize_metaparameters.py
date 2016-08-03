#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="This script prints to its standard output "
                                 "an initial version of the file of meta-parameters, "
                                 "containing either default weight values or values supplied "
                                 "via the --weights option.",
                                 epilog="Prints its output to the stdout",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--weights", type=str,
                    help="If supplied, the filename for weights from unigram "
                    "LM estimation (e.g. as obtained from get_unigram_weights.py). "
                    "In this case you must also supply the --names option to "
                    "help interpret the weights.");
parser.add_argument("--names", type=str,
                    help="Required if the --weights option is used, to help "
                    "interpret the weights (as we want them expressed in terms of "
                    "the integer names of the training sets)");
parser.add_argument("--ngram-order", type=int,
                    help="The N-gram order of your final LM (required)")
parser.add_argument("--num-train-sets", type=int,
                    help="The number of training sets (required)")

args = parser.parse_args()


# this reads the 'names' file (which has lines like "1 switchboard", "2 fisher"
# and so on), and returns a dictionary from integer id to name.
def ReadNames(names_file):
    try:
        f = open(names_file, "r");
    except:
        sys.exit("initialize_metaparameters.py: failed to open --names={0}"
                 " for reading".format(names_file))
    number_to_name = { }
    for line in f:
        try:
            [ number, name ] = line.split();
            number = int(number)
        except:
            sys.exit("initialize_metaparameters.py: Bad line '{0}' in names file {1}".format(
                    line[0:-1], names_file))
        if number in number_to_name:
            sys.exit("initialize_metaparameters.py: duplicate number {0} in names file {1}".format(
                    number, names_file))
        number_to_name[number] = name
    f.close()
    return number_to_name


# this reads the 'weights' file (which has lines like "switchboard 0.9552",
# "fisher 0.0423" and so on), and returns a dictionary from name to
# floating-point weight.
def ReadWeights(weights_file):
    try:
        f = open(weights_file, "r");
    except:
        sys.exit("initialize_metaparameters.py: failed to open --weights={0}"
                 " for reading".format(weights_file))
    name_to_weight = { }
    for line in f:
        try:
            [ name, weight ] = line.split();
            weight = float(weight)
        except:
            sys.exit("initialize_metaparameters.py: Bad line '{0}' in weights file {1}".format(
                    line[0:-1], weights_file))
        if name in name_to_weight:
            sys.exit("initialize_metaparameters.py: duplicate name {0} in weights file {1}".format(
                    name, weights_file))
        name_to_weight[name] = weight
    f.close()
    return name_to_weight

if args.num_train_sets == None or  args.num_train_sets <= 0:
    sys.exit("initialize_metaparameters.py: --num-train-sets must be supplied, and >0.")

if args.ngram_order == None or args.ngram_order <= 1:
    sys.exit("initialize_metaparameters.py: --num-train-sets must be supplied, and >1.")


# set all the weights to 0.5, it will give them room to grow (since
# we'll constrain them to not exceed 1.0).
weights = [ 0.5 ] * args.num_train_sets

if args.weights != None:
    if args.names == None:
        sys.exit("initialize_metaparameters.py: if --weights is supplied, "
                 "--names must also be supplied.")
    number_to_name = ReadNames(args.names)
    name_to_weight = ReadWeights(args.weights)
    for n in range(args.num_train_sets):
        try:
            weights[n] = name_to_weight[number_to_name[n + 1]]
        except:
            sys.exit("initialize_metaparameters.py: it looks like there is a mismatch between "
                     "the --names, --weights, and/or --num-train-sets options; check that the "
                     "{0}'th dataset has a name in the 'names' file and that that name has a "
                     "weight in the 'weights' file.".format(n+1))

# OK, now we need to make sure that all the weights are unique (otherwise the
# derivative backpropagation gets very arbitrary, due to ties).  Note, we want
# this delta to be larger than the one used in test_metaparameter_derivs.py.
delta = 0.01
weights_set = set()
for n in range(args.num_train_sets):
    if weights[n] in weights_set:
        this_delta = delta
        while weights[n] + this_delta in weights_set:
            this_delta += delta
        weights[n] += this_delta
    weights_set.add(weights[n])

# Now make sure that all weights are strictly >0 and strictly <1.
# we'll use a barrier function in the optimization, so we do need
# to make sure that none of the weights are exactly 0 or 1.
# First shift the weights up so none exceed 0.01.
weight_lower_limit = 0.01
if min(weights) < weight_lower_limit:
    shift = weight_lower_limit - min(weights)
    weights = [ x + shift for x in weights ]

# Next, in order to come reasonably close to placing the weights
# in the 'middle' of the range [0, 1], and keeping the barrier
# function happy, scale the weights by a value such that the
# smallest and largest weight are equal distances from [0,1]
# respectively.  Note: scaling all the weights by the same
# amount does not affect the actual objective function, only
# the barrier function.

min_weight = min(weights)
max_weight = max(weights)
# we want to solve for scale, such that
# (min_weight * scale - 0) == (1 - max_weight * scale).
# That means: (min_weight + max_weight) * scale == 1,
# or scale = 1 / (min_weight + max_weight)

scale = 1.0 / (min_weight + max_weight)
weights = [ x * scale for x in weights ]

# At this point we print out the weights.
# The general format of the meta-parameters file is:
# description-of-number number,
# e.g.
# count_scale_1  0.023
# count_scale_2  0.543
# ..
# order2_D1 0.8
# order2_D2 0.4
# order2_D3 0.2

for n in range(args.num_train_sets):
    print("count_scale_{0} {1}".format(n + 1, weights[n]))

for o in range(2, args.ngram_order + 1):
    print("order{0}_D1 0.8".format(o))
    print("order{0}_D2 0.4".format(o))
    print("order{0}_D3 0.2".format(o))
    print("order{0}_D4 0.1".format(o))

