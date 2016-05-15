#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import numpy as np
from scipy.optimize import minimize

import re, os, argparse, sys, math, warnings, shutil
from math import log

parser = argparse.ArgumentParser(description="Optimizes metaparameters for LM estimation; "
                                 "this utility uses derivatives from get_objf_and_derivs.py or "
                                 "get_objf_and_derivs_split.py");

parser.add_argument("--barrier-epsilon", type=float, default=1.0e-04,
                    help="Scaling factor on logarithmic barrier function to "
                    "enforce parameter constraints (should make very little "
                    "difference as long as it is quite small)")
parser.add_argument("--gradient-tolerance", type=float, default=1.0e-07,
                    help="Norm of gradient w.r.t. metaparameters, at which we "
                    "terminate optimization.  Larger->faster, smaller->more accurate.")
parser.add_argument("--num-splits", type=int, default=1,
                    help="Controls the number of parallel processes used to "
                    "get objective functions and derivatives.  If >1, then "
                    "we split the counts and compute these things in parallel.");
parser.add_argument("count_dir",
                    help="Directory in which to find counts")
parser.add_argument("optimize_dir",
                    help="Directory to store temporary files for optimization, including "
                    "metaparameters; should contain the file 0.metaparams at the start.")

args = parser.parse_args()

# Add the script dir to the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])));


if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit("optimize_metaparameters.py: validate_count_dir.py failed")

if args.num_splits < 1:
    sys.exit("optimize_metaparameters.py: --num-splits must be >0.")
if args.num_splits > 1:
    if (os.system("split_count_dir.sh {0} {1}".format(
                args.count_dir, args.num_splits))) != 0:
        sys.exit("optimize_metaparameters.py: failed to create split count-dir.")

if not os.path.exists(args.optimize_dir + "/0.metaparams"):
    sys.exit("optimize_metaparameters.py: expected file {0}/0.metaparams "
             "to exist".format(args.optimize_dir))

if not os.path.exists(args.optimize_dir + "/work"):
    os.makedirs(args.optimize_dir + "/work")

# you can set this to true for testing, to verify that the objf and derivative
# from the barrier functions do not mismatch.
use_zero_objf = False

# read the variables 'ngram_order' and 'num_train_sets'
# from the corresponding files in count_dir.
for name in [ 'ngram_order', 'num_train_sets' ]:
    f = open(args.count_dir + os.sep + name)
    globals()[name] = int(f.readline())
    f.close()


def ReadObjf(file):
    f = open(file, "r")
    line = f.readline()
    assert  len(line.split()) == 1
    assert f.readline() == ''
    f.close()
    return float(line)

metaparameter_names = None

# this function reads metaparameters or derivatives from 'file', and returns
# them as a vector (in the form of a numpy array).
def ReadMetaparametersOrDerivs(file):
    global metaparameter_names
    f = open(file, "r")
    a = f.readlines()
    if metaparameter_names == None:
        metaparameter_names = [ line.split()[0] for line in a ]
    else:
        if not metaparameter_names == [ line.split()[0] for line in a ]:
            sys.exit("optimize_metaparameters.py: mismatch in metaparameter names; "
                     "try cleaning directory " + args.optimize_dir)
    f.close()
    return np.array([ float(line.split()[1]) for line in a ])


# this function writes the metaparameters in the numpy array 'array'
# to file 'file'; it returns true if the file was newly created and/or
# had different contents than its previous contents.
def WriteMetaparameters(file, array):
    f = open(file + ".tmp", "w")
    assert len(array) == len(metaparameter_names)
    for i in range(len(array)):
        print(metaparameter_names[i], array[i], file=f)
    f.close()
    if os.system("cmp -s {0} {0}.tmp".format(file)) == 0:
        os.unlink(file + ".tmp")
        return False  # no change.
    else:
        os.rename(file + ".tmp", file)
        return True  # it changed or is new

# This function returns true if the metaparameters are in the allowed region,
# i.e. the scales are strictly between 0 and 1 and for each order, 1 > d1 > d2 >
# d3 > 0.  Otherwise it returns false.
def MetaparametersAreAllowed(x):
    global num_train_sets, ngram_order
    assert len(x) == num_train_sets + 4 * (ngram_order - 1)
    for i in range(num_train_sets):
        if x[i] <= 0.0 or x[i] >= 1.0:
            return False
    for o in range(2, ngram_order + 1):
        dim_offset = num_train_sets + 4 * (o-2)
        d1 = x[dim_offset]
        d2 = x[dim_offset + 1]
        d3 = x[dim_offset + 2]
        d4 = x[dim_offset + 3]
        if not (1.0 > d1 and d1 > d2 and d2 > d3 and d3 > d4 and d4 > 0.0):
            return False
    return True


# this function, which requires that x be in the 'allowed' region, computes
# the modified objective function taking into account the barrier function,
# and returns the modified pair (objf, derivs).
# note, this happens before the objf is negated, while we're still maximizing.
# if we have a constraint of the form x > 0, we're just adding
# log(x) to the objective function.  This becomes very negative (i.e. bad),
# as x approaches 0.
def ModifyWithBarrierFunction(x, objf, derivs):
    epsilon = args.barrier_epsilon
    derivs = derivs.copy() # don't overwrite the object.
    global num_train_sets, ngram_order
    assert len(x) == num_train_sets + 4 * (ngram_order - 1)
    for i in range(num_train_sets):
        xi = x[i]
        # the constraints are: xi > 0.0, and 1.0 - xi > 0.0
        objf += epsilon * (log(xi - 0.0) + log(1.0 - xi))
        derivs[i] += epsilon * ((1.0 / xi) + (-1.0 / (1.0 - xi)))

    for o in range(2, ngram_order + 1):
        dim_offset = num_train_sets + 4 * (o-2)
        d1 = x[dim_offset]
        d2 = x[dim_offset + 1]
        d3 = x[dim_offset + 2]
        d4 = x[dim_offset + 3]
        # the constraints are:
        # 1.0 - d1 > 0.0
        # d1 - d2 > 0.0
        # d2 - d3 > 0.0
        # d3 - d4 > 0.0
        #      d4 > 0.0
        objf += epsilon * (log(1.0 - d1) + log(d1 - d2) + log(d2 - d3) +
                           log(d3 - d4) + log(d4))
        # deriv for d1
        derivs[dim_offset] += epsilon * (-1.0 / (1.0 - d1) + 1.0 / (d1 - d2))
        # deriv for d2
        derivs[dim_offset + 1] += epsilon * (-1.0 / (d1 - d2) + 1.0 / (d2 - d3))
        # deriv for d3
        derivs[dim_offset + 2] += epsilon * (-1.0 / (d2 - d3) + 1.0 / (d3 - d4))
        # deriv for d4
        derivs[dim_offset + 3] += epsilon * (-1.0 / (d3 - d4) + 1.0 / d4)
    return (objf, derivs)


# this will return a 2-tuple (objf, deriv).  note, the objective function and
# derivative are both negated because scipy only supports minimization.
def GetObjfAndDeriv(x):
    global iteration
    if not MetaparametersAreAllowed(x):
        # return negative infinity, and a zero derivative.
        print("Metaparameters not allowed: ", x)
        return (1.0e+10, np.array([0]*len(x)))

    metaparameter_file = "{0}/{1}.metaparams".format(args.optimize_dir, iteration)
    deriv_file = "{0}/{1}.derivs".format(args.optimize_dir, iteration)
    objf_file = "{0}/{1}.objf".format(args.optimize_dir, iteration)
    log_file = "{0}/{1}.log".format(args.optimize_dir, iteration)

    if use_zero_objf:  # only for testing.
        objf = 0.0
        derivs = x * 0.0
    else:
        changed_or_new = WriteMetaparameters(metaparameter_file, x)
        prev_metaparameter_file = "{0}/{1}.metaparams".format(args.optimize_dir, iteration - 1)
        enable_caching = True # if true, enable re-use of files from a previous run.
        enable_remembering = True  # if true, enable re-use of objf,derivs from
                                   # previous iteration (if metaparameters are
                                   # the same).
        if enable_caching and (not changed_or_new and os.path.exists(deriv_file) and
                               os.path.exists(objf_file) and
                               os.path.getmtime(deriv_file) >
                               os.path.getmtime(metaparameter_file)):
            print("optimize_metaparameters.py: using previously computed objf and deriv "
                  "info from {0} and {1} (presumably you are rerunning after a partially "
                  "finished run)".format(deriv_file, objf_file), file=sys.stderr)
        elif (enable_remembering and iteration > 0 and
            os.system("cmp -s {0} {1}".format(metaparameter_file, prev_metaparameter_file))==0):
            print("optimize_metaparameters.py: metaparameters are the same as previous iteration, so "
                  "re-using objf and derivs and copying log.", file=sys.stderr);
            shutil.copyfile("{0}/{1}.derivs".format(args.optimize_dir, iteration - 1),
                            deriv_file)
            shutil.copyfile("{0}/{1}.objf".format(args.optimize_dir, iteration - 1),
                            objf_file)
            try:
                shutil.copyfile("{0}/{1}.log".format(args.optimize_dir, iteration - 1),
                                log_file)
            except:
                pass
        else:
            # we need to call get_objf_and_derivs.py
            command = ("get_objf_and_derivs{maybe_split}.py {split_opt} --derivs-out={derivs} {counts} {metaparams} "
                       "{objf} {work} 2>{log}".format(derivs = deriv_file, counts = args.count_dir,
                                                      metaparams = metaparameter_file,
                                                      maybe_split = "_split" if args.num_splits > 1 else "",
                                                      split_opt= ("--num-splits={0}".format(args.num_splits) if
                                                                  args.num_splits > 1 else ""),
                                                      objf = objf_file, log = log_file,
                                                      work = args.optimize_dir + "/work"))
            print("optimize_metaparameters.py: getting objf and derivs on "
                  "iteration {0}, command is: {1}".format(iteration, command),
                  file=sys.stderr)
            if os.system(command) != 0:
                sys.exit("optimize_metaparameters.py: error running command.")
        derivs = ReadMetaparametersOrDerivs(deriv_file)
        objf = ReadObjf(objf_file)
        iteration += 1


    # Add something for the barrier functions to enforce constraints.  Note, the
    # derivatives couldn't easily be computed if any of the scales were to
    # become exactly zero, so it's easier to use a barrier function to enforce
    # the constraints, so we don't have to deal with that issue.
    (objf, derivs) = ModifyWithBarrierFunction(x, objf, derivs)
    print("Iteration {0}: objf={1}, deriv-magnitude={2} (with barrier function)".format(
            iteration, objf, math.sqrt(np.vdot(derivs, derivs))), file=sys.stderr)

    # we need to negate the objective function and derivatives, since
    # scipy only supports minimizing functions and we want to maximize.
    # we actually apply a negative scale not equal to -1.0, as otherwise
    # the first step is too small.
    scale = -1.0
    return (objf * scale, derivs * scale)



x0 = ReadMetaparametersOrDerivs(args.optimize_dir + "/0.metaparams")
# 'iteration' will affect the filenames used to write the metaparameters
# and derivatives.
iteration = 0

result = minimize(GetObjfAndDeriv, x0, method='BFGS', jac=True,
                  options={'disp': True, 'gtol': args.gradient_tolerance, 'mls':50})

print("result is ", result, file=sys.stderr)

WriteMetaparameters("{0}/final.metaparams".format(args.optimize_dir),
                    result.x)

old_objf = ReadObjf("{0}/0.objf".format(args.optimize_dir))
new_objf = ReadObjf("{0}/{1}.objf".format(args.optimize_dir, iteration - 1))

print("optimize_metaparameters.py: log-prob on dev data increased "
      "from {0} to {1} over {2} passes of derivative estimation (perplexity: {3}->{4}".format(
                old_objf, new_objf, iteration, math.exp(-old_objf), math.exp(-new_objf)),
      file=sys.stderr)

print("optimize_metaparameters.py: do `diff -y {0}/{{0,final}}.metaparams` "
      "to see change in metaparameters.".format(args.optimize_dir),
      file=sys.stderr)

print("optimize_metaparameters.py: Wrote final metaparameters to "
      "{0}/final.metaparams".format(args.optimize_dir),
      file=sys.stderr)
