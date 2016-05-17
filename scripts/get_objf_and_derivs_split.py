#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, threading

parser = argparse.ArgumentParser(description="This does the same as get_objf_and_derivs.py "
                                 "(i.e. it computes the log-prob per word and the derivatives), "
                                 "except it works with split-up counts directories.  For example, "
                                 "if you call split_count_dir.sh data/counts 10, then you could "
                                 "call this script with that counts directory and the option "
                                 "--num-splits 10, and it should give the same output as if you "
                                 "had called get_objf_and_derivs.py with the same counts and "
                                 "metaparameters, and without the --num-splits option.  "
                                 "The point is that this program does things in parallel, so "
                                 "it's faster.");

parser.add_argument("--num-splits", type=int,
                    help="Number of splits in count directory.  You must previously have "
                    "split the counts directory with this number of splits.  This option is "
                    "required (if you're not using splitting, then use get_objf_and_deriv.py)");
parser.add_argument("--fold-dev-into-int", type=int,
                    help="Integer identifier of dataset into which the dev data "
                    "should be folded (not compatible with the --derivs-out option)")
parser.add_argument("--derivs-out", type=str,
                    help="Filename to which to write derivatives (if supplied)");
parser.add_argument("count_dir",
                    help="Directory from which to obtain counts files\n");
parser.add_argument("metaparameters",
                    help="Filename from which to read metaparameters");
parser.add_argument("objf_out",
                    help="Filename to which to write objective function");
parser.add_argument("work_dir",
                    help="Directory used to temporarily store files and for logs");

args = parser.parse_args()

if args.num_splits == None or not args.num_splits > 1:
    sys.exit("get_objf_and_derivs_split.py: --num-splits must be supplied and >1.");

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");


if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit(1)

split_count_dir = "{0}/split{1}".format(args.count_dir, args.num_splits)
if not os.path.isdir(split_count_dir):
    sys.exit("get_objf_and_derivs_split.py: expected directory {0} to exist.".format(
            split_count_dir))
if os.system("validate_count_dir.py {0}/1".format(split_count_dir)) != 0:
    sys.exit(1)


for split_dir in [ "{0}/split{1}/{2}".format(args.work_dir, args.num_splits, n)
                   for n in range(1, args.num_splits + 1) ]:
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)
split_work_dir = "{0}/split{1}".format(args.work_dir, args.num_splits)

# read the variables 'ngram_order', 'num_train_sets' and 'num_words'
# from the corresponding files in count_dir.  (this should be the
# same as in the split directories; if not, you have a big problem).
for name in [ 'ngram_order', 'num_train_sets', 'num_words' ]:
    f = open(args.count_dir + os.sep + name)
    globals()[name] = int(f.readline())
    f.close()

if args.fold_dev_into_int != None:
    if args.derivs_out != None:
        sys.exit("get_objf_and_derivs.py: --fold-dev-into-int and --derivs-out "
                 "options are not compatible.")
    if not args.fold_dev_into_int >= 1 and args.fold_dev_into_int <= num_train_sets:
        sys.exit("get_objf_and_derivs.py: --fold-dev-into-int={0} is out of range".format(
                args.fold_dev_into_int))

if os.system("validate_metaparameters.py --ngram-order={ngram_order} "
             "--num-train-sets={num_train_sets} {metaparameters}".format(
        ngram_order=ngram_order, num_train_sets=num_train_sets,
        metaparameters = args.metaparameters)) != 0:
    sys.exit(1)


# read the metaparameters as dicts.
# train_set_scale will be a map from integer
# training-set number to floating-point scale.  Note: there is no checking
# because we already called validate_metaparameters.py.
f = open(args.metaparameters, "r")
train_set_scale = {}
for n in range(1, num_train_sets + 1):
    train_set_scale[n] = float(f.readline().split()[1])
# the discounting constants will be stored as maps d1,d2,d3,d4 from integer order
# to discounting constant.
d1 = {}
d2 = {}
d3 = {}
d4 = {}
for o in range(2, ngram_order + 1):
    d1[o] = float(f.readline().split()[1])
    d2[o] = float(f.readline().split()[1])
    d3[o] = float(f.readline().split()[1])
    d4[o] = float(f.readline().split()[1])
f.close()


def ExitProgram(message):
    print(message, file=sys.stderr)
    os._exit(1)

def RunCommand(command):
    # print the command for logging
    print(command, file=sys.stderr)
    if os.system(command) != 0:
        ExitProgram("get_objf_and_derivs_split.py: error running command: " + command)

def GetCommandStdout(command):
    # print the command for logging
    print(command, file=sys.stderr)
    try:
        output = subprocess.check_output(command, shell = True)
    except:
        ExitProgram("get_objf_and_derivs_split.py: error running command: " + command)
    return output


# This function does the count merging for the specified
# n-gram order, writing to $work_dir/merged.$order
# For the highest order we merge count_dir/int.*.order,
# each with its appropriate scalign factor; for orders
# strictly between the highest order and 1 we merge those
# but also the discounted counts from work_dir/discounted.order;
# for order 1, no merging is done (-> this function shouldn't be
# called).
def MergeCounts(split_index, order):
    # merge counts of the specified order > 1.
    assert order > 1
    command = "merge-counts";
    for n in range(1, num_train_sets + 1):
        command += " {split_counts}/{split_index}/int.{train_set}.{order},{scale}".format(
            split_counts = split_count_dir, split_index = split_index,
            train_set = n, order = order, scale = train_set_scale[n])
    if args.fold_dev_into_int != None:
        command += " {counts}/int.dev.{order},{scale}".format(
            counts = args.count_dir, order = order,
            scale = train_set_scale[args.fold_dev_into_int])
    # for orders less than the highest order, we also have to include the
    # discounted counts from the one-higher order.  there is no scale here, so
    # the program will expect general-counts, not int-counts.
    if order < ngram_order:
        command += " {0}/{1}/discounted.{2}".format(split_work_dir, split_index, order)

    # the output gets redirected to the output file.
    command += " >{0}/{1}/merged.{2}".format(split_work_dir, split_index, order)
    RunCommand(command)

def MergeCountsBackward(split_index, order):
    global scale_derivs
    # merge counts of the specified order > 1; the backprop phase.
    assert order > 1

    command = "merge-counts-backward {swork}/{s}/merged.{order} {swork}/{s}/merged_derivs.{order} ".format(
        swork = split_work_dir, s = split_index, order = order)

    for n in range(1, num_train_sets + 1):
        command += " {split_counts}/{s}/int.{train_set}.{order} {scale}".format(
            split_counts = split_count_dir, s = split_index, train_set = n,
            order = order, scale = train_set_scale[n])
    # for orders less than the highest order, we also have to include the
    # discounted counts from the one-higher order, and provide a filename
    # for it to output the derivatives w.r.t. that file.
    if order < ngram_order:
        command += " {swork}/{s}/discounted.{order} {swork}/{s}/discounted_derivs.{order}".format(
            swork = split_work_dir, s = split_index, order = order)
    output = GetCommandStdout(command)
    try:
        this_scale_derivs = [ float(n) / num_dev_set_words_total for n in output.split() ]
        assert len(scale_derivs) == num_train_sets
        # the scaling factors are applied for each order > 1, and the
        # derivatives will be a sum over the derivatives for each of these
        # orders (and also a sum over the different split-directories).
        for n in range(num_train_sets):
            scale_derivs[n] += this_scale_derivs[n]
    except:
        ExitProgram("get_objf_and_derivs_split.py: unexpected output from command:" + output)


def DiscountCounts(split_index, order):
    # discount counts of the specified order > 1.
    assert order > 1
    this_split_work = "{0}/{1}".format(split_work_dir, split_index)
    command = "discount-counts {d1} {d2} {d3} {d4} {sdir}/merged.{order} {sdir}/float.{order} {sdir}/discounted.{orderm1} ".format(
        d1 = d1[order], d2 = d2[order], d3 = d3[order], d4 = d4[order],
        sdir = this_split_work, order = order, orderm1 = order - 1)
    RunCommand(command)

def DiscountCountsBackward(split_index, order):
    # discount counts of the specified order > 1; backprop version.
    assert order > 1
    this_split_work = "{0}/{1}".format(split_work_dir, split_index)
    command = ("discount-counts-backward {d1} {d2} {d3} {d4} {sdir}/merged.{order} {sdir}/float.{order} "
               "{sdir}/float_derivs.{order} {sdir}/discounted.{orderm1} {sdir}/discounted_derivs.{orderm1} "
               "{sdir}/merged_derivs.{order}".format(
            d1 = d1[order], d2 = d2[order], d3 = d3[order], d4 = d4[order],
            sdir = this_split_work, order = order, orderm1 = order - 1))
    output = GetCommandStdout(command);
    try:
        [ deriv1, deriv2, deriv3, deriv4 ] = output.split()
    except:
        ExitProgram("get_objf_and_derivs_split.py: could not parse output of command: " + output)
    d1_deriv[order] += float(deriv1) / num_dev_set_words_total
    d2_deriv[order] += float(deriv2) / num_dev_set_words_total
    d3_deriv[order] += float(deriv3) / num_dev_set_words_total
    d4_deriv[order] += float(deriv4) / num_dev_set_words_total


def MergeCountsOrder1():
    # This function merges the order-1 discounted counts across all splits.
    command = ("merge-counts " +
               " ".join([ "{0}/{1}/discounted.1".format(split_work_dir, s)
                          for s in range(1, args.num_splits + 1) ]) +
               " >{0}/discounted.1".format(args.work_dir))
    RunCommand(command)

def MergeCountsOrder1Backward():
    # This function merges the order-1 discounted counts across all splits.
    # we pipe it to /dev/null because it writes a newline to stdout (this is
    # to terimate the derivs w.r.t. the scaling factors, which are written to
    # stdout but in this case are empty.
    command = ("merge-counts-backward {0}/discounted.1 {0}/discounted_derivs.1 ".format(
                 args.work_dir) +
               " ".join([ "{0}/{1}/discounted.1 {0}/{1}/discounted_derivs.1".format(split_work_dir, s)
                          for s in range(1, args.num_splits + 1) ]) +
               ">/dev/null")
    RunCommand(command)

def DiscountCountsOrder1():
    command = "discount-counts-1gram {num_words} <{work}/discounted.1 >{work}/float.1".format(
        num_words = num_words, work = args.work_dir)
    RunCommand(command)

def SumFloatDerivsOrder1():
    # this has to be called before DiscountCountsOrder1Backward, to sum up the
    # different parts of the final float-count derivatives w.r.t. the unigram counts, from all the
    # individual split directories.
    command = ("sum-float-derivs {0}/float.1 ".format(args.work_dir) +
               " ".join([ "{0}/{1}/float_derivs.1".format(split_work_dir, s)
                          for s in range(1, args.num_splits + 1) ]) +
               " >{0}/float_derivs.1".format(args.work_dir))
    RunCommand(command)

def DiscountCountsOrder1Backward():
    command = ("discount-counts-1gram-backward {work}/discounted.1 {work}/float.1 "
               "{work}/float_derivs.1 {work}/discounted_derivs.1".format(work = args.work_dir))
    RunCommand(command)

def MergeAllOrders(split_index):
    this_split_work = "{0}/{1}".format(split_work_dir, split_index)
    # this merges all the orders of float-counts in each of the split
    # directories.  Note that for unigram, it takes the merged-across-all-splits
    # counts from the top-level work-dir, not the split work-dir.
    command = ("merge-float-counts " +
               " ".join([ "{0}/float.{1}".format(this_split_work if n > 1 else args.work_dir, n)
                          for n in range(1, ngram_order + 1) ])
               + ">{0}/float.all".format(this_split_work))
    RunCommand(command)

def ComputeObjfAndFinalDerivs(split_index, need_derivs):
    global num_dev_set_words_total, loglike_total
    command = "compute-probs {swork}/{s}/float.all {scount}/{s}/int.dev ".format(
        swork = split_work_dir, s = split_index, scount = split_count_dir)
    if need_derivs:
        command += " ".join([ "{swork}/{s}/float_derivs.{order}".format(swork = split_work_dir,
                                                                        s = split_index, order = o)
                              for o in range(1, ngram_order + 1) ])
    output = GetCommandStdout(command)
    try:
        [ num_dev_set_words, tot_objf ] = output.split()
        num_dev_set_words_total += int(num_dev_set_words)
        loglike_total += float(tot_objf)
    except:
        ExitProgram("get_objf_and_derivs_split.py: error interpreting the output of compute-probs: "
                 "output was: " + output)


def WriteObjectiveFunction():
    objf = loglike_total / num_dev_set_words_total
    print("get_objf_and_derivs_split.py: objf is {0} over {1} "
          "words".format(objf, num_dev_set_words_total), file=sys.stderr)
    # Write the objective function.
    try:
        f = open(args.objf_out, "w")
        print(str(objf), file=f)
        f.close()
    except:
        ExitProgram("get_objf_and_derivs_split.py: error writing objective function to: " +
                    args.objf_out)

def WriteDerivs():
    try:
        f = open(args.derivs_out, "w")
    except:
        ExitProgram("get_objf_and_derivs_split.py: error opening --derivs-out={0} for writing".format(
                 args.derivs_out))
    for n in range(num_train_sets):
        print("count_scale_{0} {1}".format(n + 1, scale_derivs[n]), file=f)
    for o in range(2, ngram_order + 1):
        print("order{0}_D1 {1}".format(o, d1_deriv[o]), file=f)
        print("order{0}_D2 {1}".format(o, d2_deriv[o]), file=f)
        print("order{0}_D3 {1}".format(o, d3_deriv[o]), file=f)
        print("order{0}_D4 {1}".format(o, d4_deriv[o]), file=f)
    f.close()


def ForwardAllButFirstOrder(split_index):
    # for n-gram orders down to 2, do the merging and discounting.
    for o in range(ngram_order, 1, -1):
        MergeCounts(split_index, o)
        DiscountCounts(split_index, o)


def BackwardAllButFirstOrder(split_index):
    # for n-gram orders 2 and greater, do the backwards discounting and merging.
    # this is the backward version of ForwardAllButFirstOrder().
    for o in range(2, ngram_order + 1):
        DiscountCountsBackward(split_index, o)
        MergeCountsBackward(split_index, o)


def MergeAndComputeObjfForSplit(split_index):
    MergeAllOrders(split_index)
    ComputeObjfAndFinalDerivs(split_index, args.derivs_out != None)

# do the 'forward' computation (merging and discounting) for all the orders from
# the highest down to order 2.
threads = []
for split_index in range(1, args.num_splits + 1):
    threads.append(threading.Thread(target = ForwardAllButFirstOrder,
                                    args=[split_index]))
    threads[-1].start()
for t in threads:
    t.join()

MergeCountsOrder1()
DiscountCountsOrder1()

num_dev_set_words_total = 0
loglike_total = 0.0

threads = []
for split_index in range(1, args.num_splits + 1):
    threads.append(threading.Thread(target = MergeAndComputeObjfForSplit,
                                    args=[split_index]))
    threads[-1].start()
for t in threads:
    t.join()

WriteObjectiveFunction()

if args.derivs_out == None:
    sys.exit(0)

# scale_derivs will be an array of the derivatives of the objective function
# w.r.t. the scaling factors of the training sets.
scale_derivs = [ 0 ] * num_train_sets
# the following dicts will be indexed by the order.
d1_deriv = {}
d2_deriv = {}
d3_deriv = {}
d4_deriv = {}
for o in range(2, ngram_order + 1):
    d1_deriv[o] = 0.0;
    d2_deriv[o] = 0.0;
    d3_deriv[o] = 0.0;
    d4_deriv[o] = 0.0;

# Now comes the backprop code.

# Note: there is no need for a call like MergeAllOrdersBackward(), because that
# merging was just aggregating different histories of distinct orders, and to
# avoid the need for a backprop version of this program, the program
# 'compute-probs' writes the derivatives for history-states of distinct orders,
# to distinct files.


SumFloatDerivsOrder1()
DiscountCountsOrder1Backward()
MergeCountsOrder1Backward()

# do the 'backward' computation for orders 2 and greater,
# in parallel.
threads = []
for split_index in range(1, args.num_splits + 1):
    threads.append(threading.Thread(target = BackwardAllButFirstOrder,
                                    args=[split_index]))
    threads[-1].start()
for t in threads:
    t.join()

WriteDerivs()
