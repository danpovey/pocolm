#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import sys
import subprocess
import shutil
import threading
# from collections import defaultdict
# from subprocess import CalledProcessError

# make sure scripts/internal is on the pythonpath.
sys.path = [os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal"] + sys.path
from prune_size_model import PruneSizeModel

# for ExitProgram and RunCommand
from pocolm_common import ExitProgram
from pocolm_common import RunCommand
from pocolm_common import GetCommandStdout
from pocolm_common import LogMessage

parser = argparse.ArgumentParser(description="This script takes an lm-dir, as produced by make_lm_dir.py, "
                                 "that should not have the counts split up into pieces, and it prunes "
                                 "the counts and writes out to a new lm-dir.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--steps", type=str,
                    default='prune*0.25 EM EM EM prune*0.5 EM EM EM prune*1.0 EM EM EM prune*1.0 EM EM EM',
                    help='This string specifies a sequence of steps in the pruning sequence.'
                    'prune*X, with X <= 1.0, tells it to prune with X times the threshold '
                    'specified with the --final-threshold option.  EM specifies one iteration of '
                    'E-M on the model. ')
parser.add_argument("--final-threshold", type=float,
                    help="Threshold for pruning, e.g. 0.5, 1.0, 2.0, 4.0.... "
                    "larger threshold will give you more highly-pruned models."
                    "Threshold is interpreted as entropy-change times overall "
                    "weighted data count, for each parameter.  It should be "
                    "larger if you have more data, assuming you want the "
                    "same-sized models. "
                    "This is only relevant if --target-num-ngrams is not specified.")
parser.add_argument("--target-num-ngrams", type=int, default=0,
                    help="Target num-ngrams of final LM after pruning. "
                    "If setting this to a positive value, the --steps would be "
                    "ignored and a few steps may be worked out util the num-ngrams "
                    "of pruned LM match the target-num-ngrams.")
parser.add_argument("--target-lower-threshold", type=int,
                    help="lower tolerance of target num-ngrams. Default value is"
                    "5% relativly less than target num-ngrams. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--target-upper-threshold", type=int,
                    help="upper tolerance of target num-ngrams. Default value is"
                    "5% relativly larger than target num_ngrams. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--initial-threshold", type=float, default=0.25,
                    help="Initial threshold for the pruning steps starting from. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--max-iter", type=int, default=20,
                    help="Max iterations allowed to find the threshold for target-num-ngrams LM. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--verbose", type=str, default='false',
                    choices=['true', 'false'],
                    help="If true, print commands as we execute them.")
parser.add_argument("--cleanup",  type=str, choices=['true', 'false'],
                    default='true', help='Set this to false to disable clean up of the '
                    'work directory.')
parser.add_argument("--remove-zeros", type=str, choices=['true', 'false'],
                    default='true', help='Set this to false to disable an optimization. '
                    'Only useful for debugging purposes.')
parser.add_argument("--check-exact-divergence", type=str, choices=['true', 'false'],
                    default='true', help='')
parser.add_argument("--max-memory", type=str, default='',
                    help="Memory limitation for sort.")
parser.add_argument("lm_dir_in",
                    help="Source directory, for the input language model.")
parser.add_argument("lm_dir_out",
                    help="Output directory where the language model is created.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src")

if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    ExitProgram("failed to validate input LM-dir")

# verify the input string max_memory
if args.max_memory != '':
    # valid string max_memory must have at least two items
    if len(args.max_memory) >= 2:
        s = args.max_memory
        # valid string max_memory can be formatted as:
        # "a positive integer + a letter or a '%'" or "a positive integer"
        # the unit of memory size can also be 'T', 'P', 'E', 'Z', or 'Y'. They
        # are not included here considering their rare use in practice
        if s[-1] in ['b', 'B', '%', 'k', 'K', 'm', 'M', 'g', 'G'] or s[-1].isdigit():
            for x in s[:-1]:
                if not x.isdigit():
                    sys.exit("prune_lm_dir.py: --max-memory should be formatted as "
                             "'a positive integer' or 'a positive integer appended "
                             "with 'b', 'K', 'M','G', or '%''.")
            # max memory size must be larger than zero
            if int(s[:-1]) == 0:
                sys.exit("prune_lm_dir.py: --max-memory must be > 0 {unit}.".format(
                         unit=s[-1]))
        else:
            sys.exit("prune_lm_dir.py: the format of string --max-memory is not correct.")
    else:
        sys.exit("prune_lm_dir.py: the lenght of string --max-memory must >= 2.")
    if args.max_memory[-1] == 'B':  # sort seems not recognize 'B'
        args.max_memory[-1] = 'b'

num_splits = None
if os.path.exists(args.lm_dir_in + "/num_splits"):
    f = open(args.lm_dir_in + "/num_splits")
    num_splits = int(f.readline())
    f.close()

work_dir = args.lm_dir_out + "/work"

if args.target_num_ngrams > 0:
    if args.target_lower_threshold is not None:
        if args.target_lower_threshold >= args.target_num_ngrams:
            ExitProgram("--target-lower-threshold[{0}] should be less than "
                        "--target-num-ngrams[{1}].".format(
                          args.target_lower_threshold, args.target_num_ngrams))
    else:
        args.target_lower_threshold = int(0.95 * args.target_num_ngrams)

    if args.target_upper_threshold is not None:
        if args.target_upper_threshold <= args.target_num_ngrams:
            ExitProgram("--target-upper-threshold[{0}] should be larger than "
                        "--target-num-ngrams[{1}].".format(
                          args.target_upper_threshold, args.target_num_ngrams))
    else:
        args.target_upper_threshold = int(1.05 * args.target_num_ngrams)

    if args.max_iter <= 1:
        ExitProgram("--max-iter must be bigger than 1, got: " + str(args.max_iter))

    steps = []
else:
    if args.final_threshold <= 0.0:
        ExitProgram("--final-threshold must be positive, got: " + str(args.final_threshold))

    steps = args.steps.split()

    if len(steps) == 0:
        ExitProgram("'steps' cannot be empty.")

# set the memory restriction for "sort"
sort_mem_opt = ''
if args.max_memory != '':
    sort_mem_opt = ("--buffer-size={0} ".format(args.max_memory))

# returns num-words in this lm-dir.


def GetNumWords(lm_dir_in):
    command = "tail -n 1 {0}/words.txt".format(lm_dir_in)
    line = subprocess.check_output(command, shell=True, universal_newlines=True)
    try:
        a = line.split()
        assert len(a) == 2
        ans = int(a[1])
    except:
        ExitProgram("error: unexpected output '{0}' from command {1}".format(
                line, command))
    return ans


def GetNgramOrder(lm_dir_in):
    f = open(lm_dir_in + "/ngram_order")
    return int(f.readline())


def GetNumGrams(lm_dir_in):
    num_unigrams = 0
    # we generally use num_xgrams to refer to num_ngrams - num_unigrams
    tot_num_xgrams = 0
    f = open(lm_dir_in + "/num_ngrams")
    for order, line in enumerate(f):
        if order == 0:
            num_unigrams = int(line.split()[1])
            continue
        tot_num_xgrams += int(line.split()[1])
    return (num_unigrams, tot_num_xgrams)


# This script creates work/protected.all (listing protected
# counts which may not be removed); it requires work/float.all
# to exist.
def CreateProtectedCounts(work):
    command = ("bash -c 'float-counts-to-histories <{0}/float.all | LC_ALL=C sort {1}|"
               " histories-to-null-counts >{0}/protected.all'".format(work, sort_mem_opt))
    log_file = work + "/log/create_protected_counts.log"
    RunCommand(command, log_file, args.verbose == 'true')


def SoftLink(src, dest):
    if os.path.lexists(dest):
        os.remove(dest)
    try:
        os.symlink(os.path.abspath(src), dest)
    except:
        ExitProgram("error linking {0} to {1}".format(os.path.abspath(src), dest))


def CreateInitialWorkDir():
    # Creates float.all, stats.all, and protected.all in work_dir/step
    work0dir = work_dir + "/step0"
    # create float.all
    if not os.path.isdir(work0dir + "/log"):
        os.makedirs(work0dir + "/log")
    SoftLink(args.lm_dir_in + "/num_ngrams", work0dir + "/num_ngrams")
    if num_splits is None:
        SoftLink(args.lm_dir_in + "/float.all", work0dir + "/float.all")
    else:
        splits_star = ' '.join([args.lm_dir_in + "/float.all." + str(n)
                               for n in range(1, num_splits + 1)])
        command = "merge-float-counts " + splits_star + " >{0}/float.all".format(work0dir)
        log_file = work0dir + "/log/merge_initial_float_counts.log"
        RunCommand(command, log_file, args.verbose == 'true')

    # create protected.all
    CreateProtectedCounts(work0dir)

    stats_star = ' '.join(["{0}/stats.{1}".format(work0dir, n)
                          for n in range(1, ngram_order + 1)])

    # create stats.{1,2,3..}
    # e.g. command = 'float-counts-to-float-stats 20000 foo/work/step0/stats.1 '
    #                'foo/work/step0/stats.2 <foo/work/step0/float.all'
    command = ("float-counts-to-float-stats {0} ".format(num_words) +
               stats_star +
               " <{0}/float.all".format(work0dir))
    log_file = work0dir + "/log/float_counts_to_float_stats.log"
    RunCommand(command, log_file, args.verbose == 'true')
    command = "merge-float-counts {0} > {1}/stats.all".format(
        stats_star, work0dir)
    log_file = work0dir + "/log/merge_float_counts.log"
    RunCommand(command, log_file, args.verbose == 'true')
    for f in stats_star.split():
        os.remove(f)


# sets initial_logprob_per_word.
def GetInitialLogprob():
    work0dir = work_dir + "/step0"
    float_star = ' '.join(['/dev/null' for n in range(1, ngram_order + 1)])
    command = ('float-counts-estimate {num_words} {work0dir}/float.all '
               '{work0dir}/stats.all {float_star} '.format(
                   num_words=num_words, work0dir=work0dir,
                   float_star=float_star))
    try:
        print(command, file=sys.stderr)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        # the stdout of this program will be something like:
        # 1.63388e+06 -7.39182e+06 10.5411 41.237 49.6758
        # representing: total-count, total-like, and for each order, the like-change
        # for that order.
        line = p.stdout.readline()
        print(line, file=sys.stderr)
        a = line.split()
        tot_count = float(a[0])
        tot_like = float(a[1])
        like_change = 0.0
        logprob_per_word = tot_like / tot_count
        for i in range(2, len(a)):  # for each n-gram order
            like_change += float(a[i])
        like_change_per_word = like_change / tot_count
        assert like_change_per_word < 0.0001  # should be exactly zero.
    except Exception as e:
        ExitProgram("error running command '{0}', error is '{1}'".format(
                command, repr(e)))
    global initial_logprob_per_word
    initial_logprob_per_word = logprob_per_word


def WriteNumNgrams(out_dir, num_ngrams):
    out_file = out_dir + "/num_ngrams"
    try:
        f = open(out_file, "w")
        for order, num in enumerate(num_ngrams):
            print(str(order + 1) + ' ' + str(num), file=f)
        f.close()
    except:
        ExitProgram("error writing num-ngrams to: " + out_file)


def RunPruneStep(work_in, work_out, threshold):
    # set float_star = 'work_out/float.1 work_out/float.2 ...'
    float_star = " ".join(['{0}/float.{1}'.format(work_out, n)
                          for n in range(1, ngram_order + 1)])
    # create work_out/float.{1,2,..}
    log_file = work_out + '/log/float_counts_prune.log'
    command = ("float-counts-prune {threshold} {num_words} {work_in}/float.all "
               "{work_in}/protected.all {float_star} 2>>{log_file}".format(
                  threshold=threshold, num_words=num_words,
                  work_in=work_in, float_star=float_star, log_file=log_file))
    with open(log_file, 'w') as f:
        print("# " + command, file=f)
    try:
        print(command, file=sys.stderr)
        p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True, universal_newlines=True)
        [word_count, like_change] = p.stdout.readline().split()
        like_change_per_word = float(like_change) / float(word_count)
        [tot_xgrams, shadowed, protected, pruned] = p.stdout.readline().split()
        num_ngrams = p.stdout.readline().split()

        assert p.stdout.readline() == ''
        ret = p.wait()
        assert ret == 0
        global current_num_xgrams

        current_num_xgrams = int(tot_xgrams) - int(pruned)
    except Exception as e:
        ExitProgram("error running command '{0}', error is '{1}'".format(
                command, repr(e)))

    WriteNumNgrams(work_out, num_ngrams)

    if args.remove_zeros == 'false':
        # create work_out/float.all.
        command = 'merge-float-counts {0} >{1}/float.all'.format(float_star,
                                                                 work_out)
        log_file = work_out + '/log/merge_float_counts.log'
        RunCommand(command, log_file, args.verbose == 'true')
        for f in float_star.split():
            os.remove(f)
        # soft-link work_out/stats.all to work_in/stats.all
        SoftLink(work_in + "/stats.all",
                 work_out + "/stats.all")
    else:
        # in this case we pipe the output of merge-float-counts into
        # float-counts-stats-remove-zeros.
        # set stats_star = 'work_out/stats.1 work_out/stats.2 ..'
        stats_star = " ".join(['{0}/stats.{1}'.format(work_out, n)
                              for n in range(1, ngram_order + 1)])

        command = ('merge-float-counts {float_star} | float-counts-stats-remove-zeros '
                   '{num_words} /dev/stdin {work_in}/stats.all {work_out}/float.all '
                   '{stats_star}'.format(
                       num_words=num_words, float_star=float_star,
                       work_in=work_in, work_out=work_out,
                       stats_star=stats_star))
        log_file = work_out + '/log/remove_zeros.log'
        RunCommand(command, log_file, args.verbose == 'true')
        # create work_out/stats.all
        command = 'merge-float-counts {0} >{1}/stats.all'.format(stats_star, work_out)
        log_file = work_out + '/log/merge_float_counts.log'
        RunCommand(command, log_file, args.verbose == 'true')
        for f in float_star.split() + stats_star.split():
            os.remove(f)

    # create work_out/protected.all
    CreateProtectedCounts(work_out)
    return like_change_per_word


def RunEmStep(work_in, work_out):
    # set float_star = 'work_out/float.1 work_out/float.2 ...'
    float_star = " ".join(['{0}/float.{1}'.format(work_out, n)
                          for n in range(1, ngram_order + 1)])

    command = ('float-counts-estimate {num_words} {work_in}/float.all {work_in}/stats.all '
               '{float_star}'.format(num_words=num_words, work_in=work_in,
                                     float_star=float_star))
    log_file = work_out + "/log/float_counts_estimate.log"
    try:
        output = GetCommandStdout(command, log_file, args.verbose == 'true')
        # the stdout of this program will be something like:
        # 1.63388e+06 -7.39182e+06 10.5411 41.237 49.6758
        # representing: total-count, total-like, and for each order, the like-change
        # for that order.
        a = output.split()
        tot_count = float(a[0])
        tot_like = float(a[1])
        like_change = 0.0
        global final_logprob_per_word
        final_logprob_per_word = tot_like / tot_count
        for i in range(2, len(a)):  # for each n-gram order
            like_change += float(a[i])
        like_change_per_word = like_change / tot_count
    except Exception as e:
        ExitProgram("error running command '{0}', error is '{1}'".format(
                command, repr(e)))

    command = 'merge-float-counts {0} >{1}/float.all'.format(float_star, work_out)
    log_file = work_out + '/log/merge_float_counts.log'
    RunCommand(command, log_file, args.verbose == 'true')
    for f in float_star.split():
        os.remove(f)
    # soft-link work_out/stats.all to work_in/stats.all
    SoftLink(work_in + "/stats.all",
             work_out + "/stats.all")
    # soft-link work_out/protected.all to work_in/protected.all
    SoftLink(work_in + "/protected.all",
             work_out + "/protected.all")
    SoftLink(work_in + "/num_ngrams",
             work_out + "/num_ngrams")
    return like_change_per_word


# runs one of the numbered steps.  step_number >= 0 is the number of the work
# directory we'll get the input from (the output will be that plus one).
# returns the expected log-prob change (on data generated from the model
# itself.. this will be negative for pruning steps and positive for E-M steps.
def RunStep(step_number, threshold, **kwargs):
    if 'in_step' in kwargs:
        work_in = work_dir + "/step" + str(kwargs['in_step'])
    else:
        work_in = work_dir + "/step" + str(step_number)
    work_out = work_dir + "/step" + str(step_number + 1)
    if not os.path.isdir(work_out + "/log"):
        os.makedirs(work_out + "/log")
    step_text = steps[step_number]
    if step_text[0:6] == 'prune*':
        try:
            scale = float(step_text[6:])
            assert scale != 0.0
        except:
            ExitProgram("invalid step (wrong --steps "
                        "option): '{0}'".format(step_text))
        return RunPruneStep(work_in, work_out, threshold * scale)

    elif step_text == 'EM':
        return RunEmStep(work_in, work_out)
    else:
        ExitProgram("invalid step (wrong --steps "
                    "option): '{0}'".format(step_text))


def FinalizeOutput(final_work_out):
    try:
        shutil.move(final_work_out + "/float.all",
                    args.lm_dir_out + "/float.all")
    except:
        ExitProgram("error moving {0}/float.all to {1}/float.all".format(
                final_work_out, args.lm_dir_out))
    try:
        shutil.copy(final_work_out + "/num_ngrams",
                    args.lm_dir_out + "/num_ngrams")
    except:
        ExitProgram("error copying {0}/num_ngrams to {1}/num_ngrams".format(
                final_work_out, args.lm_dir_out))
    f = open(args.lm_dir_out + "/was_pruned", "w")
    print("true", file=f)
    f.close()
    for f in ['names', 'words.txt', 'ngram_order', 'metaparameters']:
        try:
            shutil.copy(args.lm_dir_in + "/" + f,
                        args.lm_dir_out + "/" + f)
        except:
            ExitProgram("error copying {0}/{1} to {2}/{1}".format(
                    args.lm_dir_in, f, args.lm_dir_out))
    if os.path.exists(args.lm_dir_out + "/num_splits"):
        os.remove(args.lm_dir_out + "/num_splits")


# find threshold in order to match the target-num-ngrams with final LM
# using PruneSizeModel
# this will return a tuple of (threshold, num_iterations), if we overshot with
# the initial_threshold, it will return (0.0, 0)
def FindThreshold(initial_threshold):
    global initial_num_xgrams, current_num_xgrams, num_unigrams, steps
    global logprob_changes, effective_logprob_changes

    model = PruneSizeModel(num_unigrams, args.target_num_ngrams,
                           args.target_lower_threshold, args.target_upper_threshold)
#    model.SetDebug(True)

    model.SetInitialThreshold(initial_threshold, initial_num_xgrams)

    cur_threshold = initial_threshold
    backtrack_iter = 0
    step = 0
    iter2step = [0]  # This maps a iter-index to the step-index of the last step of that iteration
    while True:
        steps += ['prune*1.0']
        logprob_change = RunStep(step, cur_threshold, in_step=iter2step[backtrack_iter])
        logprob_changes.append(logprob_change)
        effective_logprob_changes.append(logprob_change)
        thresholds.append(cur_threshold)
        step += 1

        (action, arguments) = model.GetNextAction(current_num_xgrams)
        if action == 'overshoot':
            return (0.0, 0)

        if action == 'backtrack':
            (cur_threshold, backtrack_iter) = arguments
            assert(iter2step[backtrack_iter] > 0)
            del effective_logprob_changes[iter2step[backtrack_iter]:]
            iter2step.append(-1)
            continue

        # EM steps
        steps += 'EM EM'.split()
        while step < len(steps):
            logprob_change = RunStep(step, 0.0)
            logprob_changes.append(logprob_change)
            effective_logprob_changes.append(logprob_change)
            step += 1

        iter2step.append(step)

        if action == 'success':
            return (cur_threshold, model.iter)

        # action == 'continue':
        if model.iter > args.max_iter:
            ExitProgram("Too many iterations, please set a higher --initial-threshold and rerun.")

        cur_threshold = arguments
        backtrack_iter = model.iter


if not os.path.isdir(work_dir):
    try:
        os.makedirs(work_dir)
    except:
        ExitProgram("error creating directory " + work_dir)


num_words = GetNumWords(args.lm_dir_in)
ngram_order = GetNgramOrder(args.lm_dir_in)
(num_unigrams, initial_num_xgrams) = GetNumGrams(args.lm_dir_in)
current_num_xgrams = None
initial_logprob_per_word = None
final_logprob_per_word = None
waiting_thread = None
logprob_changes = []
effective_logprob_changes = []
thresholds = []

CreateInitialWorkDir()

if args.check_exact_divergence == 'true':
    if args.target_num_ngrams <= 0 and steps[-1] != 'EM':
        LogMessage("--check-exact-divergence=true won't give you the "
                   "exact divergence because the last step is not 'EM'.")
    waiting_thread = threading.Thread(target=GetInitialLogprob)
    waiting_thread.start()

if args.target_num_ngrams > 0:
    # For PruneSizeModel.MatchTargetNumNgrams() and PruneSizeModel.NumXgrams2NumNgrams()
    model = PruneSizeModel(num_unigrams, args.target_num_ngrams,
                           args.target_lower_threshold, args.target_upper_threshold)

    if model.MatchTargetNumNgrams(initial_num_xgrams):
        LogMessage("the input LM is already match the size with target-num-ngrams, do not need any pruning")
        sys.exit(0)

    if args.target_num_ngrams > model.NumXgrams2NumNgrams(initial_num_xgrams):
        ExitProgram("the num-ngrams({0}) of input LM is less than the target-num-ngrams({1}), "
                    "can not do any pruning.".format(
                        model.NumXgrams2NumNgrams(initial_num_xgrams), args.target_num_ngrams))

    threshold = 0.0
    initial_threshold = args.initial_threshold
    while threshold == 0.0:
        (threshold, iter) = FindThreshold(initial_threshold)
        if threshold > 0.0:
            break
        logprob_changes = []
        effective_logprob_changes = []
        thresholds = []
        steps = []
        initial_threshold /= 4.0
        LogMessage("Reduce --initial-threshold to {0}, and retry.".format(
                    initial_threshold))

    LogMessage("Find the threshold {0} in {1} iteration(s)".format(threshold, iter))
    LogMessage("thresholds per iter were " + str(thresholds))
else:
    for step in range(len(steps)):
        logprob_change = RunStep(step, args.final_threshold)
        logprob_changes.append(logprob_change)
        effective_logprob_changes.append(logprob_change)

FinalizeOutput(work_dir + "/step" + str(len(steps)))

if waiting_thread is not None:
    waiting_thread.join()

LogMessage("log-prob changes per step were " + str(logprob_changes))

initial_num_ngrams = initial_num_xgrams + num_unigrams
current_num_ngrams = current_num_xgrams + num_unigrams
LogMessage("reduced number of n-grams from {0} to {1}, "
           "i.e. by {2}%".format(initial_num_ngrams, current_num_ngrams,
                                 100.0 * (initial_num_ngrams - current_num_ngrams) / initial_num_ngrams))

# The following prints the K-L divergence; it breaks out the parts by sign so
# you can see the effect of the E-M separately (it's usually quite small).
LogMessage("approximate K-L divergence was {0} + {1} = {2}".format(
        -sum([max(0.0, x) for x in effective_logprob_changes]),
        -sum([min(0.0, x) for x in effective_logprob_changes]),
        -sum(effective_logprob_changes)))

if initial_logprob_per_word is not None and steps[-1] == 'EM':
    LogMessage("exact K-L divergence was {0}".format(
            initial_logprob_per_word - final_logprob_per_word))

# clean up the work directory.
if args.cleanup == 'true':
    shutil.rmtree(work_dir)

if os.system("validate_lm_dir.py " + args.lm_dir_out) != 0:
    ExitProgram("failed to validate output LM-dir")
