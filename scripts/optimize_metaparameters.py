#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import numpy as np

import re, os, argparse, sys, math, warnings, shutil
from math import log

# we need to add the ./internal/ subdirectory to the pythonpath before importing
# 'bfgs'.
sys.path = [ os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal" ] + sys.path
import bfgs
# for ExitProgram, RunCommand and GetCommandStdout
from pocolm_common import *

parser = argparse.ArgumentParser(description="Optimizes metaparameters for LM estimation; "
                                 "this utility uses derivatives from get_objf_and_derivs.py or "
                                 "get_objf_and_derivs_split.py")
parser.add_argument("--barrier-epsilon", type=float, default=1.0e-04,
                    help="Scaling factor on logarithmic barrier function to "
                    "enforce parameter constraints (should make very little "
                    "difference as long as it is quite small)")
parser.add_argument("--gradient-tolerance", type=float, default=0.0005,
                    help="Norm of gradient w.r.t. metaparameters, at which we "
                    "terminate optimization.  Larger->faster, smaller->more accurate.")
parser.add_argument("--progress-tolerance", type=float, default=1.0e-06,
                    help="Tolerance for amount of objective function progress, amortized over "
                    "3 iterations, to be used as a termination condition for BFGS.");
parser.add_argument("--num-splits", type=int, default=1,
                    help="Controls the number of parallel processes used to "
                    "get objective functions and derivatives.  If >1, then "
                    "we split the counts and compute these things in parallel.")
parser.add_argument("--read-inv-hessian", type=str,
                    help="Filename from which to write the inverse Hessian for "
                    "BFGS optimization.")
parser.add_argument("--initial-metaparameters", type=str,
                    help="If supplied, the initial metaparameters will be taken from  "
                    "here.  If not supplied, initialize_metaparameters.py will be "
                    "called to initialize them.")
parser.add_argument("--warm-start-dir", type=str,
                    help="The name of a directory where optimize_metaparameters.py was "
                    "run on a subset of data.  Setting --subset-optimize-dir=X is "
                    "equivalent to setting --read-inv-hessian=X/final.inv_hessian and "
                    "--initial-metaparameters=X/final.metaparams")
parser.add_argument("--clean-up", type=str, default='true', choices=['true','false'],
                    help="If true, remove the data that won't be used in the future to "
                    "save space (use 'false' for debug purpose). ")
parser.add_argument("count_dir",
                    help="Directory in which to find counts")
parser.add_argument("optimize_dir",
                    help="Directory to store temporary files for optimization; final "
                    "metaparameters are written to final.metaparams in this directory.")

args = parser.parse_args()

# Add the script dir to the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])));

if args.warm_start_dir != None:
    if args.initial_metaparameters != None or args.read_inv_hessian != None:
        sys.exit("optimize_metaparameters.py: if you set --subset-optimize-dir "
                 "you should not set --initial-metaparameters or "
                 "--read-inv-hessian.")
    args.initial_metaparameters = args.warm_start_dir + "/final.metaparams"
    args.read_inv_hessian = args.warm_start_dir + "/final.inv_hessian"

if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit("optimize_metaparameters.py: validate_count_dir.py failed")

if args.num_splits < 1:
    sys.exit("optimize_metaparameters.py: --num-splits must be >0.")
if args.num_splits > 1:
    if (os.system("split_count_dir.sh {0} {1}".format(
                args.count_dir, args.num_splits))) != 0:
        sys.exit("optimize_metaparameters.py: failed to create split count-dir.")

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

if args.initial_metaparameters != None:
    # the reason we do the cmp before copying, is that
    # if we copy even if it's the same as before, it messes with the caching.
    if (os.system("cmp -s {0} {1}/0.metaparams || cp {0} {1}/0.metaparams".format(
                args.initial_metaparameters, args.optimize_dir)) != 0 or
        os.system("validate_metaparameters.py --ngram-order={0} --num-train-sets={1} "
                  "{2}/0.metaparams".format(ngram_order, num_train_sets,
                                            args.optimize_dir)) != 0):
        sys.exit("optimize_metaparameters.py: error copying or validating initial "
                 "metaparameters from {0}".format(args.initial_metaparameters))
else:
    if os.path.exists(args.count_dir + "/unigram_weights"):
        # initialize the corpus weights from the weights optimized for a
        # unigram model; this is a better starting point than all-equal.
        weight_opts = "--weights={0}/unigram_weights --names={0}/names".format(
            args.count_dir)
    else:
        weight_opts = ""

    command = ("initialize_metaparameters.py {0} --ngram-order={1} --num-train-sets={2} "
               ">{3}/0.metaparams".format(weight_opts, ngram_order,
                                          num_train_sets, args.optimize_dir));
    if os.system(command) != 0:
        sys.exit("optimize_metaparameters.py: failed to initialize parameters, command was: " +
                 command)

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


# this function, which requires that x be in the 'allowed' region,
# returns the barrier-function component of the objective function,
# and its derivative w.r.t. x, as a 2-tuple.
# this barrier function approaches negative infinity as we get
# close to the edges of the region.  [note: we negate before
# minimizing.
def BarrierFunctionAndDeriv(x):
    epsilon = args.barrier_epsilon
    barrier = 0.0
    derivs = np.array([0.0] * len(x))
    global num_train_sets, ngram_order
    assert len(x) == num_train_sets + 4 * (ngram_order - 1)
    for i in range(num_train_sets):
        xi = x[i]
        # the constraints are: xi > 0.0, and 1.0 - xi > 0.0
        barrier += epsilon * (log(xi - 0.0) + log(1.0 - xi))
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
        barrier += epsilon * (log(1.0 - d1) + log(d1 - d2) + log(d2 - d3) +
                           log(d3 - d4) + log(d4))
        # deriv for d1
        derivs[dim_offset] += epsilon * (-1.0 / (1.0 - d1) + 1.0 / (d1 - d2))
        # deriv for d2
        derivs[dim_offset + 1] += epsilon * (-1.0 / (d1 - d2) + 1.0 / (d2 - d3))
        # deriv for d3
        derivs[dim_offset + 2] += epsilon * (-1.0 / (d2 - d3) + 1.0 / (d3 - d4))
        # deriv for d4
        derivs[dim_offset + 3] += epsilon * (-1.0 / (d3 - d4) + 1.0 / d4)
    return (barrier, derivs)


# this will return a 2-tuple (objf, deriv).  note, the objective function and
# derivative are both negated because conventionally optimization problems are
# framed as minimization problems.
def GetObjfAndDeriv(x):
    global iteration
    if not MetaparametersAreAllowed(x):
        # return negative infinity, and a zero derivative.
        print("Metaparameters not allowed: ", x)
        return (1.0e+10, np.array([0.0]*len(x)))

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
        if enable_caching and (not changed_or_new and os.path.exists(deriv_file) and
                               os.path.exists(objf_file) and
                               os.path.getmtime(deriv_file) >
                               os.path.getmtime(metaparameter_file)):
            print("optimize_metaparameters.py: using previously computed objf and deriv "
                  "info from {0} and {1} (presumably you are rerunning after a partially "
                  "finished run)".format(deriv_file, objf_file), file=sys.stderr)
        else:
            # we need to call get_objf_and_derivs.py
            clean_up_opt = '--clean-up=false' if args.clean_up == 'false' else ''
            command = ("get_objf_and_derivs{maybe_split}.py {split_opt} --derivs-out={derivs} {clean_up_opt} {counts} {metaparams} "
                       "{objf} {work}".format(derivs = deriv_file, counts = args.count_dir,
                                              metaparams = metaparameter_file,
                                              clean_up_opt = clean_up_opt,
                                                      maybe_split = "_split" if args.num_splits > 1 else "",
                                                      split_opt= ("--num-splits={0}".format(args.num_splits) if
                                                                  args.num_splits > 1 else ""),
                                              objf = objf_file, work = args.optimize_dir + "/work"))
            RunCommand(command, log_file, verbose = True)
        derivs = ReadMetaparametersOrDerivs(deriv_file)
        objf = ReadObjf(objf_file)
        iteration += 1


    # Add something for the barrier functions to enforce constraints.  Note, the
    # derivatives couldn't easily be computed if any of the scales were to
    # become exactly zero, so it's easier to use a barrier function to enforce
    # the constraints, so we don't have to deal with that issue.
    (barrier_objf, barrier_derivs) = BarrierFunctionAndDeriv(x)

    barrier_free_magnitude = math.sqrt(np.vdot(derivs, derivs))
    objf += barrier_objf
    derivs += barrier_derivs
    print("Evaluation %d: objf=%.6f, deriv-magnitude=%.6f (with barrier function; without = %.6f)" %
          (iteration, objf, math.sqrt(np.vdot(derivs, derivs)), barrier_free_magnitude),
           file=sys.stderr)

    # we need to negate the objective function and derivatives, since we are
    # minimizing.
    scale = -1.0
    global value0
    if value0 == None:
        value0 = objf * scale
    return (objf * scale, derivs * scale)


x0 = ReadMetaparametersOrDerivs(args.optimize_dir + "/0.metaparams")
# 'iteration' will affect the filenames used to write the metaparameters
# and derivatives.
iteration = 0
# value0 will store the first evaluated objective function.a
value0 = None

inv_hessian = None
if not args.read_inv_hessian is None:
    print("optimize_metaparameters.py: reading inverse Hessian from {0}".format(
            args.read_inv_hessian), file=sys.stderr)
    inv_hessian = np.loadtxt(args.read_inv_hessian)
    if inv_hessian.shape != (len(x0), len(x0)):
        sys.exit("optimize_metaparameters.py: inverse Hessian from {0} "
                 "has wrong shape.".format(args.read_inv_hessian),
                 file=sys.stderr)

(x, value, deriv, inv_hessian) = bfgs.Bfgs(x0, GetObjfAndDeriv, MetaparametersAreAllowed,
                                           init_inv_hessian = inv_hessian,
                                           gradient_tolerance = args.gradient_tolerance,
                                           progress_tolerance = args.progress_tolerance)

print("optimize_metaparameters: final metaparameters are ", x, file=sys.stderr)

WriteMetaparameters("{0}/final.metaparams".format(args.optimize_dir), x)

old_objf = -1.0 * value0
new_objf = -1.0 * value

print("optimize_metaparameters.py: log-prob on dev data (with barrier function) increased "
      "from %.6f to %.6f over %d passes of derivative estimation (penalized perplexity: %.6f->%.6f" %
      (old_objf, new_objf, iteration, math.exp(-old_objf), math.exp(-new_objf)),
      file=sys.stderr)

print("optimize_metaparameters.py: final perplexity without barrier function was %.6f "
      "(perplexity: %.6f)" % (new_objf - BarrierFunctionAndDeriv(x)[0],
                               math.exp(-(new_objf - BarrierFunctionAndDeriv(x)[0]))),
       file=sys.stderr)

print("optimize_metaparameters.py: do `diff -y {0}/{{0,final}}.metaparams` "
      "to see change in metaparameters.".format(args.optimize_dir),
      file=sys.stderr)


print("optimize_metaparameters.py: Wrote final metaparameters to "
      "{0}/final.metaparams".format(args.optimize_dir),
      file=sys.stderr)

# save the inverse Hessian, in case we want to use it to initialize a later
# round of optimization.
np.savetxt("{0}/final.inv_hessian".format(args.optimize_dir), inv_hessian)
