#!/usr/bin/env python3

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import numpy as np
import os
import argparse
import sys
import math
from math import log

# If the encoding of the default sys.stdout is not utf-8,
# force it to be utf-8. See PR #95.
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != "utf-8":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

# we need to add the ./internal/ subdirectory to the pythonpath before importing
# 'bfgs'.
sys.path = [os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal"
            ] + sys.path
import bfgs
# for GetCommandStdout
from pocolm_common import RunCommand

parser = argparse.ArgumentParser(
    description="Optimizes metaparameters for LM estimation; "
    "this utility uses derivatives from get_objf_and_derivs.py or "
    "get_objf_and_derivs_split.py",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    "--gradient-tolerance",
    type=float,
    default=0.000125,
    help="Norm of gradient w.r.t. metaparameters, at which we "
    "terminate optimization.  Larger->faster, smaller->more accurate.")
parser.add_argument(
    "--progress-tolerance",
    type=float,
    default=1.0e-06,
    help="Tolerance for amount of objective function progress, amortized over "
    "3 iterations, to be used as a termination condition for BFGS.")
parser.add_argument(
    "--num-splits",
    type=int,
    default=1,
    help="Controls the number of parallel processes used to "
    "get objective functions and derivatives.  If >1, then "
    "we split the counts and compute these things in parallel.")
parser.add_argument(
    "--read-inv-hessian",
    type=str,
    help="Filename from which to write the inverse Hessian for "
    "BFGS optimization.")
parser.add_argument(
    "--initial-metaparameters",
    type=str,
    help="If supplied, the initial metaparameters will be taken from  "
    "here.  If not supplied, initialize_metaparameters.py will be "
    "called to initialize them.")
parser.add_argument(
    "--warm-start-dir",
    type=str,
    help="The name of a directory where optimize_metaparameters.py was "
    "run on a subset of data.  Setting --subset-optimize-dir=X is "
    "equivalent to setting --read-inv-hessian=X/final.inv_hessian and "
    "--initial-metaparameters=X/final.metaparams")
parser.add_argument("--cleanup",
                    type=str,
                    default="true",
                    choices=["true", "false"],
                    help="If true, remove intermediate files in work_dir "
                    "that won't be used in future")
parser.add_argument("count_dir", help="Directory in which to find counts")
parser.add_argument(
    "optimize_dir",
    help="Directory to store temporary files for optimization; final "
    "metaparameters are written to final.metaparams in this directory.")

# echo command line to stderr for logging.
print(' '.join(sys.argv), file=sys.stderr)

args = parser.parse_args()

# Add the script dir to the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if args.warm_start_dir is not None:
    if args.initial_metaparameters is not None or args.read_inv_hessian is not None:
        sys.exit(
            "optimize_metaparameters.py: if you set --subset-optimize-dir "
            "you should not set --initial-metaparameters or "
            "--read-inv-hessian.")
    args.initial_metaparameters = args.warm_start_dir + "/final.metaparams"
    args.read_inv_hessian = args.warm_start_dir + "/final.inv_hessian"

if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit("optimize_metaparameters.py: validate_count_dir.py failed")

if args.num_splits < 1:
    sys.exit("optimize_metaparameters.py: --num-splits must be >0.")
if args.num_splits > 1:
    if (os.system("split_count_dir.sh {0} {1}".format(args.count_dir,
                                                      args.num_splits))) != 0:
        sys.exit(
            "optimize_metaparameters.py: failed to create split count-dir.")

if not os.path.exists(args.optimize_dir + "/work"):
    os.makedirs(args.optimize_dir + "/work")

# read the variables 'ngram_order' and 'num_train_sets'
# from the corresponding files in count_dir.
for name in ['ngram_order', 'num_train_sets']:
    f = open(args.count_dir + os.sep + name, encoding="utf-8")
    globals()[name] = int(f.readline())
    f.close()

if args.initial_metaparameters is not None:
    # the reason we do the cmp before copying, is that
    # if we copy even if it's the same as before, it messes with the caching.
    if (os.system(
            "cmp -s {0} {1}/0.metaparams || cp {0} {1}/0.metaparams".format(
                args.initial_metaparameters, args.optimize_dir)
    ) != 0 or os.system(
            "validate_metaparameters.py --ngram-order={0} --num-train-sets={1} "
            "{2}/0.metaparams".format(ngram_order, num_train_sets,
                                      args.optimize_dir)) != 0):
        sys.exit(
            "optimize_metaparameters.py: error copying or validating initial "
            "metaparameters from {0}".format(args.initial_metaparameters))
else:
    if os.path.exists(args.count_dir + "/unigram_weights"):
        # initialize the corpus weights from the weights optimized for a
        # unigram model; this is a better starting point than all-equal.
        weight_opts = "--weights={0}/unigram_weights --names={0}/names".format(
            args.count_dir)
    else:
        weight_opts = ""

    command = (
        "initialize_metaparameters.py {0} --ngram-order={1} --num-train-sets={2} "
        ">{3}/0.metaparams".format(weight_opts, ngram_order, num_train_sets,
                                   args.optimize_dir))
    if os.system(command) != 0:
        sys.exit(
            "optimize_metaparameters.py: failed to initialize parameters, command was: "
            + command)


def ReadObjf(file):
    f = open(file, "r", encoding="utf-8")
    line = f.readline()
    assert len(line.split()) == 1
    assert f.readline() == ''
    f.close()
    return float(line)


metaparameter_names = None
# this function reads metaparameters or derivatives from 'file', and returns
# them as a vector (in the form of a numpy array).


def ReadMetaparametersOrDerivs(file):
    global metaparameter_names
    f = open(file, "r", encoding="utf-8")
    a = f.readlines()
    if metaparameter_names is None:
        metaparameter_names = [line.split()[0] for line in a]
    else:
        if not metaparameter_names == [line.split()[0] for line in a]:
            sys.exit(
                "optimize_metaparameters.py: mismatch in metaparameter names; "
                "try cleaning directory " + args.optimize_dir)
    f.close()
    return np.array([float(line.split()[1]) for line in a])


# this function writes the metaparameters in the numpy array 'array'
# to file 'file'; it returns true if the file was newly created and/or
# had different contents than its previous contents.
def WriteMetaparameters(file, array):
    f = open(file + ".tmp", "w", encoding="utf-8")
    assert len(array) == len(metaparameter_names)

    # Even though mathematically none of the values can be <= 0 or >= 1, they
    # might be so to machine precision or in the printed format, so we impose
    # maxima and minima very close to 1 to ensure this does not happen.
    assert len(array) == num_train_sets + 4 * (ngram_order - 1)
    floors = []
    ceilings = []
    for i in range(num_train_sets):
        floors.append(1.0e-10)
        # want a number that's distinct from 1 in single precision.
        ceilings.append(1.0 - 1.0e-6)
    for i in range(ngram_order - 1):
        floors += [1.0e-10, 1.0e-11, 1.0e-12, 1.0e-13]
        # want ceilings that get farther from 1 and that are distinct from 1 and
        # from each other in single precision.
        for m in [0.25e-05, 0.5e-05, 0.75e-05, 1.0e-05]:
            ceilings.append(1.0 - m)

    for i in range(len(array)):
        value = array[i]
        if value < floors[i]:
            value = floors[i]
        if value > ceilings[i]:
            value = ceilings[i]
        printed_form = '%.15f' % value
        print(metaparameter_names[i], printed_form, file=f)

    f.close()
    if os.system("cmp -s {0} {0}.tmp".format(file)) == 0:
        os.unlink(file + ".tmp")
        return False  # no change.
    else:
        os.rename(file + ".tmp", file)
        return True  # it changed or is new


def Sigmoid(x):
    if x > 0:
        return 1.0 / (1 + math.exp(-x))
    else:
        e = math.exp(x)
        return e / (e + 1.0)


# this is the inverse of the sigmoid function.
def Logit(x):
    if (x == 0.0):
        print("optimize_metaparameters.py: warning: x == 0.")
        return -100.0
    elif (x == 1.0):
        print("optimize_metaparameters.py: warning: x == 1.")
        return 100.0
    else:
        return log(x / (1.0 - x))


# this takes a set of metaparameters x (and optionally a set of derivatives) in
# the unconstrained space, and reparameterizes it to the constrained space
# (the constrained space, e.g. between 0 and 1 for the corpus weights, is
# the most natural representation of the parameters).
# the 'derivs' are the derivatives of an objective function w.r.t. x (df/dx).
def UnconstrainedToConstrained(x):
    global num_train_sets, ngram_order
    assert len(x) == num_train_sets + 4 * (ngram_order - 1)
    y = np.array([0.0] * len(x))

    for i in range(num_train_sets):
        y[i] = Sigmoid(x[i])

    for o in range(2, ngram_order + 1):
        dim_offset = num_train_sets + 4 * (o - 2)
        x1 = x[dim_offset]
        x2 = x[dim_offset + 1]
        x3 = x[dim_offset + 2]
        x4 = x[dim_offset + 3]
        s1 = Sigmoid(x1)
        s2 = Sigmoid(x2)
        s3 = Sigmoid(x3)
        s4 = Sigmoid(x4)
        d1 = s1
        d2 = s1 * s2
        d3 = s1 * s2 * s3
        d4 = s1 * s2 * s3 * s4
        y[dim_offset] = d1
        y[dim_offset + 1] = d2
        y[dim_offset + 2] = d3
        y[dim_offset + 3] = d4

    return y


# this takes a set of metaparameters y (and optionally a set of derivatives) in
# the constrained space (the space where the parameters are in their most
# natural form), and reparameterizes them into to the unconstrained space;
# it also transforms the derivatives df_dy.
# if df_dy is supplied, then returns (x, df_dx), else returns just x.


def ConstrainedToUnconstrained(y, df_dy=None):
    global num_train_sets, ngram_order
    assert len(y) == num_train_sets + 4 * (ngram_order - 1)
    x = np.array([0.0] * len(y))
    df_dx = np.array([0.0] * len(y))
    if df_dy is None:
        df_dy = np.array([0.0] * len(y))
        need_deriv = False
    else:
        need_deriv = True

    for i in range(num_train_sets):
        x[i] = Logit(y[i])
        # dy/dx of y = Sigmoid(x) is y*(1-y).
        # this is like backprop.
        df_dx[i] = df_dy[i] * y[i] * (1.0 - y[i])

    for o in range(2, ngram_order + 1):
        dim_offset = num_train_sets + 4 * (o - 2)
        d1 = y[dim_offset]
        d2 = y[dim_offset + 1]
        d3 = y[dim_offset + 2]
        d4 = y[dim_offset + 3]
        df_dd1 = df_dy[dim_offset]
        df_dd2 = df_dy[dim_offset + 1]
        df_dd3 = df_dy[dim_offset + 2]
        df_dd4 = df_dy[dim_offset + 3]
        s1 = d1
        s2 = d2 / d1
        s3 = d3 / d2
        s4 = d4 / d3
        # the following expressions could be made more compact, but in the
        # following form it's probably easiest to see how they derive (via
        # backprop) from the expressions for d1 through d4 in the
        # UnconstrainedToConstrained() function.
        df_ds1 = df_dd1 + df_dd2 * s2 + df_dd3 * s2 * s3 + df_dd4 * s2 * s3 * s4
        df_ds2 = df_dd2 * s1 + df_dd3 * s1 * s3 + df_dd4 * s1 * s3 * s4
        df_ds3 = df_dd3 * s1 * s2 + df_dd4 * s1 * s2 * s4
        df_ds4 = df_dd4 * s1 * s2 * s3
        x1 = Logit(s1)
        x2 = Logit(s2)
        x3 = Logit(s3)
        x4 = Logit(s4)
        x[dim_offset] = Logit(s1)
        x[dim_offset + 1] = Logit(s2)
        x[dim_offset + 2] = Logit(s3)
        x[dim_offset + 3] = Logit(s4)
        # ds/dx of s = Sigmoid(x) is s*(1-s).
        df_dx[dim_offset] = df_ds1 * s1 * (1.0 - s1)
        df_dx[dim_offset + 1] = df_ds2 * s2 * (1.0 - s2)
        df_dx[dim_offset + 2] = df_ds3 * s3 * (1.0 - s4)
        df_dx[dim_offset + 3] = df_ds4 * s4 * (1.0 - s4)

    return (x, df_dx) if need_deriv else x


def TestConstraints():
    dim = num_train_sets + 4 * (ngram_order - 1)
    x0 = np.random.rand(dim)
    df_dy = np.random.rand(dim)
    x_delta = np.random.rand(dim) * 0.0001
    x1 = x0 + x_delta
    y0 = UnconstrainedToConstrained(x0)
    y1 = UnconstrainedToConstrained(x1)
    f0 = np.dot(df_dy, y0)
    f1 = np.dot(df_dy, y1)
    delta_f = f1 - f0
    (x0_check, df_dx) = ConstrainedToUnconstrained(y0, df_dy)
    delta_f_check = np.dot(df_dx, x_delta)
    x_diff = np.dot(x0 - x0_check, x0 - x0_check)
    print(
        "optimize_metaparameters.py: checking constraints: x_diff = {0} [should be small], "
        "delta_f = {1}, delta_f_check = {2} [should be similar]".format(
            x_diff, delta_f, delta_f_check),
        file=sys.stderr)


# this will return a 2-tuple (objf, deriv).  note, the objective function and
# derivative are both negated because conventionally optimization problems are
# framed as minimization problems.
def GetObjfAndDeriv(x):
    global iteration

    y = UnconstrainedToConstrained(x)

    metaparameter_file = "{0}/{1}.metaparams".format(args.optimize_dir,
                                                     iteration)
    deriv_file = "{0}/{1}.derivs".format(args.optimize_dir, iteration)
    objf_file = "{0}/{1}.objf".format(args.optimize_dir, iteration)
    log_file = "{0}/{1}.log".format(args.optimize_dir, iteration)

    changed_or_new = WriteMetaparameters(metaparameter_file, y)
    prev_metaparameter_file = "{0}/{1}.metaparams".format(
        args.optimize_dir, iteration - 1)
    enable_caching = True  # if true, enable re-use of files from a previous run.
    if enable_caching and (not changed_or_new and os.path.exists(deriv_file)
                           and os.path.exists(objf_file)
                           and os.path.getmtime(deriv_file) >
                           os.path.getmtime(metaparameter_file)):
        print(
            "optimize_metaparameters.py: using previously computed objf and deriv "
            "info from {0} and {1} (presumably you are rerunning after a partially "
            "finished run)".format(deriv_file, objf_file),
            file=sys.stderr)
    else:
        # we need to call get_objf_and_derivs.py
        command = (
            "get_objf_and_derivs{maybe_split}.py {split_opt} --cleanup={cleanup} --derivs-out={derivs} {counts} {metaparams} "
            "{objf} {work}".format(
                derivs=deriv_file,
                counts=args.count_dir,
                metaparams=metaparameter_file,
                maybe_split="_split" if args.num_splits > 1 else "",
                split_opt=("--num-splits={0}".format(args.num_splits)
                           if args.num_splits > 1 else ""),
                cleanup=args.cleanup,
                objf=objf_file,
                work=args.optimize_dir + "/work"))
        RunCommand(command, log_file, verbose=True)
    df_dy = ReadMetaparametersOrDerivs(deriv_file)
    objf = ReadObjf(objf_file)
    iteration += 1

    (x2, df_dx) = ConstrainedToUnconstrained(y, df_dy)

    # check that x == x2, we just changed variables back and forth so it should
    # be the same.
    if math.sqrt(np.dot(x - x2, x - x2)) > 0.001:
        print(
            "optimize_metaparameters.py: warning: difference {0} versus {1}\n".
            format(x, x2))

    print("Evaluation %d: objf=%.6f, deriv-magnitude=%.6f " %
          (iteration, objf, math.sqrt(np.vdot(df_dx, df_dx))),
          file=sys.stderr)

    # we need to negate the objective function and derivatives, since we are
    # minimizing.
    scale = -1.0
    global value0
    if value0 is None:
        value0 = objf * scale
    return (objf * scale, df_dx * scale)


# Uncomment the following to check that the derivatives are correct (w.r.t.
# the change of variables); you have to look at the output, it doesn't
# fail automatically.
# TestConstraints()

y0 = ReadMetaparametersOrDerivs(args.optimize_dir + "/0.metaparams")
x0 = ConstrainedToUnconstrained(y0)  # change of variables to make it an
# unconstrained optimization problem.
# 'iteration' will affect the filenames used to write the metaparameters
# and derivatives.
iteration = 0
# value0 will store the first evaluated objective function.
value0 = None

inv_hessian = None
if args.read_inv_hessian is not None:
    print(
        "optimize_metaparameters.py: reading inverse Hessian from {0}".format(
            args.read_inv_hessian),
        file=sys.stderr)
    inv_hessian = np.loadtxt(args.read_inv_hessian)
    if inv_hessian.shape != (len(x0), len(x0)):
        sys.exit("optimize_metaparameters.py: inverse Hessian from {0} "
                 "has wrong shape.".format(args.read_inv_hessian),
                 file=sys.stderr)

(x, value, deriv,
 inv_hessian) = bfgs.Bfgs(x0,
                          GetObjfAndDeriv, (lambda x: True),
                          init_inv_hessian=inv_hessian,
                          gradient_tolerance=args.gradient_tolerance,
                          progress_tolerance=args.progress_tolerance,
                          verbose=True)

y = UnconstrainedToConstrained(x)
print("optimize_metaparameters: final metaparameters are ", y, file=sys.stderr)
if y[-4] < 0.1:
    sys.exit(
        "your dev set is probably in your training set; this is not advisable")

WriteMetaparameters("{0}/final.metaparams".format(args.optimize_dir), y)

old_objf = -1.0 * value0
new_objf = -1.0 * value

print(
    "optimize_metaparameters.py: log-prob on dev data increased "
    "from %.6f to %.6f over %d passes of derivative estimation (perplexity: %.6f->%.6f"
    %
    (old_objf, new_objf, iteration, math.exp(-old_objf), math.exp(-new_objf)),
    file=sys.stderr)

print("optimize_metaparameters.py: final objf was %.6f (perplexity: %.6f)" %
      (new_objf, math.exp(-new_objf)),
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
