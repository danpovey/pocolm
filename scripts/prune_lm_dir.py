#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, threading
from collections import defaultdict
from subprocess import CalledProcessError

# make sure scripts/internal is on the pythonpath.
sys.path = [ os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal" ] + sys.path

# for ExitProgram and RunCommand
from pocolm_common import *

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
                    "of pruned LM match the target-num-ngrams. This value excludes the number of unigrams.")
parser.add_argument("--initial-threshold", type=float, default=0.25,
                    help="Initial threshold for the pruning steps starting from. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--tolerance", type=float, default=0.05,
                    help="Tolerance of actual num-ngrams of final LM and the target-num-ngrams. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--max-iter", type=int, default=20,
                    help="Max iterations allowed to find the threshold for target-num-ngrams LM. "
                    "This is only relevant if --target-num-ngrams is specified.")
parser.add_argument("--verbose", type=str, default='false',
                    choices=['true','false'],
                    help="If true, print commands as we execute them.")
parser.add_argument("--cleanup",  type=str, choices=['true','false'],
                    default='true', help='Set this to false to disable clean up of the '
                    'work directory.')
parser.add_argument("--remove-zeros", type=str, choices=['true','false'],
                    default='true', help='Set this to false to disable an optimization. '
                    'Only useful for debugging purposes.')
parser.add_argument("--check-exact-divergence", type=str, choices=['true','false'],
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
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    sys.exit("prune_lm_dir.py: failed to validate input LM-dir")

# verify the input string max_memory
if args.max_memory != '':
    # valid string max_memory must have at least two items 
    if len(args.max_memory) >= 2:
        s = args.max_memory
        # valid string max_memory can be formatted as:
        # "a positive integer + a letter or a '%'" or "a positive integer"
        # the unit of memory size can also be 'T', 'P', 'E', 'Z', or 'Y'. They
        # are not included here considering their rare use in practice
        if s[-1] in ['b', '%', 'K', 'M', 'G'] or s[-1].isdigit():
            for x in s[:-1]:
                if not x.isdigit():
                    sys.exit("prune_lm_dir.py: --max-memory should be formatted as "
                             "'a positive integer' or 'a positive integer appended "
                             "with 'b', 'K', 'M','G', or '%''.")
            # max memory size must be larger than zero
            if int(s[:-1]) == 0:
                sys.exit("prune_lm_dir.py: --max-memory must be > 0 {unit}.".format(
                         unit = s[-1]))    
        else:
            sys.exit("prune_lm_dir.py: the format of string --max-memory is not correct.")
    else:
         sys.exit("prune_lm_dir.py: the lenght of string --max-memory must >= 2.")

num_splits = None
if os.path.exists(args.lm_dir_in + "/num_splits"):
    f = open(args.lm_dir_in + "/num_splits")
    num_splits = int(f.readline())
    f.close()

work_dir = args.lm_dir_out + "/work"

if args.target_num_ngrams > 0:
    if args.tolerance <= 0.0 or args.tolerance >= 0.5:
        sys.exit("prune_lm_dir.py: illegal tolerance: " + str(args.tolerance))
    if args.max_iter <= 1:
        sys.exit("prune_lm_dir.py: --max-iter must be bigger than 1, got : " + str(args.max_iter))

    steps = 'prune*1.0 EM EM EM'.split()
else:
    if args.final_threshold <= 0.0:
        sys.exit("prune_lm_dir.py: --final-threshold must be positive: got " + str(args.final_threshold))

    steps = args.steps.split()

    if len(steps) == 0:
        sys.exit("prune_lm_dir.py: 'steps' cannot be empty.")

# set the memory restriction for "sort"
sort_mem_opt = ''
if args.max_memory != '':
  sort_mem_opt = ("--buffer-size={0} ".format(args.max_memory))

# returns num-words in this lm-dir.
def GetNumWords(lm_dir_in):
    command = "tail -n 1 {0}/words.txt".format(lm_dir_in)
    line = subprocess.check_output(command, shell = True)
    try:
        a = line.split()
        assert len(a) == 2
        ans = int(a[1])
    except:
        sys.exit("prune_lm_dir: error: unexpected output '{0}' from command {1}".format(
                line, command))
    return ans

def GetNgramOrder(lm_dir_in):
    f = open(lm_dir_in + "/ngram_order");
    return int(f.readline())

def GetTotalNumNgrams(lm_dir_in):
    tot_num_ngrams = 0
    f = open(lm_dir_in + "/num_ngrams");
    for order, line in enumerate(f):
        if order == 0: continue # output of float-counts-prune excludes num-unigrams
        tot_num_ngrams += int(line.split()[1])
    return tot_num_ngrams

# This script creates work/protected.all (listing protected
# counts which may not be removed); it requires work/float.all
# to exist.
def CreateProtectedCounts(work):
    command = ("bash -c 'float-counts-to-histories <{0}/float.all | LC_ALL=C sort {1}|"
               " histories-to-null-counts >{0}/protected.all'".format(work, sort_mem_opt))
    log_file = work + "/log/create_protected_counts.log"
    RunCommand(command, log_file, args.verbose == 'true')

def SoftLink(src, dest):
    if os.path.exists(dest):
        os.remove(dest)
    try:
        os.symlink(os.path.abspath(src), dest)
    except:
        sys.exit("prune_lm_dir.py: error linking {0} to {1}".format(src, dest))

def CreateInitialWorkDir():
    # Creates float.all, stats.all, and protected.all in work_dir/step
    work0dir = work_dir + "/step0"
    # create float.all
    if not os.path.isdir(work0dir + "/log"):
        os.makedirs(work0dir + "/log")
    SoftLink(args.lm_dir_in + "/num_ngrams", work0dir + "/num_ngrams")
    if num_splits == None:
        SoftLink(args.lm_dir_in + "/float.all", work0dir + "/float.all")
    else:
        splits_star = ' '.join([ args.lm_dir_in + "/float.all." + str(n)
                                 for n in range(1, num_splits + 1) ])
        command = "merge-float-counts " + splits_star + " >{0}/float.all".format(work0dir)
        log_file = work0dir + "/log/merge_initial_float_counts.log"
        RunCommand(command, log_file, args.verbose == 'true')

    # create protected.all
    CreateProtectedCounts(work0dir)

    stats_star = ' '.join([ "{0}/stats.{1}".format(work0dir, n)
                            for n in range(1, ngram_order + 1) ])

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
        os.remove(f);

# sets initial_logprob_per_word.
def GetInitialLogprob():
    work0dir = work_dir + "/step0"
    float_star = ' '.join([ '/dev/null' for n in range(1, ngram_order + 1) ])
    command = ('float-counts-estimate {num_words} {work0dir}/float.all '
               '{work0dir}/stats.all {float_star} '.format(
            num_words = num_words, work0dir = work0dir,
            float_star = float_star))
    try:
        print(command, file=sys.stderr)
        p = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
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
        sys.exit("prune_lm_dir.py: error running command '{0}', error is '{1}'".format(
                command, str(e)))
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
        ExitProgram("prune_lm_dir.py: error writing num-ngrams to: " + out_file)

def RunPruneStep(work_in, work_out, threshold):
    # set float_star = 'work_out/float.1 work_out/float.2 ...'
    float_star = " ".join([ '{0}/float.{1}'.format(work_out, n)
                            for n in range(1, ngram_order + 1) ])
    # create work_out/float.{1,2,..}
    command = ("float-counts-prune {threshold} {num_words} {work_in}/float.all "
               "{work_in}/protected.all ".format(threshold = threshold,
                                                 num_words = num_words,
                                                 work_in = work_in) +
               float_star)
    try:
        print(command, file=sys.stderr)
        p = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
        [ word_count, like_change ] = p.stdout.readline().split()
        like_change_per_word = float(like_change) / float(word_count)
        [ tot_ngrams, shadowed, protected, pruned ] = p.stdout.readline().split()
        num_ngrams = p.stdout.readline().split()

        assert p.stdout.readline() == ''
        ret = p.wait()
        assert ret == 0
        global current_num_ngrams, initial_num_ngrams, input_num_ngrams
        if initial_num_ngrams == None:
            initial_num_ngrams = int(tot_ngrams)
            if initial_num_ngrams != input_num_ngrams:
                sys.exit("prune_lm_dir.py: total num-ngrams are not match. "
                    "The num_ngrams file says it is '{0}', "
                    "but float-counts-prune outputs '{1}'".format(input_num_ngrams, initial_num_ngrams))

        current_num_ngrams = int(tot_ngrams) - int(pruned)
    except Exception as e:
        sys.exit("prune_lm_dir.py: error running command '{0}', error is '{1}'".format(
                command, str(e)))

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
        stats_star = " ".join([ '{0}/stats.{1}'.format(work_out, n)
                                for n in range(1, ngram_order + 1) ])

        command = ('merge-float-counts {float_star} | float-counts-stats-remove-zeros '
                   '{num_words} /dev/stdin {work_in}/stats.all {work_out}/float.all '
                   '{stats_star}'.format(
                num_words = num_words, float_star = float_star,
                work_in = work_in, work_out = work_out,
                stats_star = stats_star))
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
    float_star = " ".join([ '{0}/float.{1}'.format(work_out, n)
                            for n in range(1, ngram_order + 1) ])

    command = ('float-counts-estimate {num_words} {work_in}/float.all {work_in}/stats.all '
               '{float_star}'.format(num_words = num_words, work_in = work_in,
                                     float_star = float_star))
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
        sys.exit("prune_lm_dir.py: error running command '{0}', error is '{1}'".format(
                command, str(e)))

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
            sys.exit("prune_lm_dir.py: invalid step (wrong --steps "
                     "option): '{0}'".format(step_text))
        return RunPruneStep(work_in, work_out, threshold * scale)

    elif step_text == 'EM':
        return RunEmStep(work_in, work_out)
    else:
        sys.exit("prune_lm_dir.py: invalid step (wrong --steps "
                 "option): '{0}'".format(step_text))


def FinalizeOutput(final_work_out):
    try:
        shutil.move(final_work_out + "/float.all",
                    args.lm_dir_out + "/float.all")
    except:
        sys.exit("prune_lm_dir.py: error moving {0}/float.all to {1}/float.all".format(
                final_work_out, args.lm_dir_out))
    try:
        shutil.copy(final_work_out + "/num_ngrams",
                    args.lm_dir_out + "/num_ngrams")
    except:
        sys.exit("prune_lm_dir.py: error copying {0}/num_ngrams to {1}/num_ngrams".format(
                final_work_out, args.lm_dir_out))
    f = open(args.lm_dir_out + "/was_pruned", "w")
    print("true", file=f)
    f.close()
    for f in [ 'names', 'words.txt', 'ngram_order', 'metaparameters' ]:
        try:
            shutil.copy(args.lm_dir_in + "/" + f,
                        args.lm_dir_out + "/" + f)
        except:
            sys.exit("prune_lm_dir.py: error copying {0}/{1} to {2}/{1}".format(
                    args.lm_dir_in, f, args.lm_dir_out))
    if os.path.exists(args.lm_dir_out + "/num_splits"):
        os.remove(args.lm_dir_out + "/num_splits")


def MatchTargetSize(num_ngrams):
    assert(args.target_num_ngrams > 0)
    return abs(num_ngrams - args.target_num_ngrams) / float(args.target_num_ngrams) < args.tolerance

def IterateOnce(threshold, step, iter, recovery_step, recovery_size):
    """Prune with a specific threshold and check whether the resulting LM matches the target-num-ngrams

    Args:
        threshold: the threshold used to prune LM
        step: the current step-index (each individual operation, such as pruning or E-M, counts as one step).
        iter: index of current iteration. We only increase the iteration index when we do an actual pruning step,
            so it will be <= the step index.'
        recovery_step, recovery_size: the step-index and num-grams
            for LM that is most recent and smallest but larger than target-num-ngrams.
            This is used as the starting point for the iteration done by this function.
    Returns:
        A tuple with the first element is a bool variable indicating whether we already got the right threshold.
        And the remaining are updated value for (step, iter, recovery_step, recovery_size)
    Example Usage:

        step = 0
        recovery_step = step
        recovery_size = 0
        iter = 0

        threshold = init_threshold()
        while True:
            threshold = EstimateThreshold(args.target_num_ngrams)

            (success, step, iter, recovery_step, recovery_size) = \
                    IterateOnce(threshold, step, iter, recovery_step, recovery_size)
            if success:
                break

    """
    global logprob_changes, logprob_changes_without_dropped_steps, steps, current_num_ngrams

    if step > 0:
        steps += 'prune*1.0 EM EM EM'.split()

    thresholds.append(threshold)
    # Prune step
    logprob_change = RunStep(step, threshold, in_step=recovery_step)
    logprob_changes.append(logprob_change)
    step += 1

    step_without_dropped_steps = -1
    if len(logprob_changes_without_dropped_steps) > recovery_step:
        step_without_dropped_steps = recovery_step

    if step_without_dropped_steps < 0:
        logprob_changes_without_dropped_steps.append(logprob_change)
    else:
        logprob_changes_without_dropped_steps[step_without_dropped_steps] = logprob_change
        step_without_dropped_steps += 1

    while step < len(steps): # EM steps
        logprob_change = RunStep(step, threshold)
        logprob_changes.append(logprob_change)
        step += 1

        if step_without_dropped_steps < 0:
            logprob_changes_without_dropped_steps.append(logprob_change)
        else:
            logprob_changes_without_dropped_steps[step_without_dropped_steps] = logprob_change
            step_without_dropped_steps += 1
    iter += 1

    if MatchTargetSize(current_num_ngrams):
        return (True, step, iter, recovery_step, recovery_size)

    if iter > args.max_iter:
        sys.exit("prune_lm_dir.py: Too many iterations, please set a higher --initial-threshold and rerun.")

    # always prune from LM larger than target_num_ngrams
    if current_num_ngrams > args.target_num_ngrams and (recovery_size <= 0 or current_num_ngrams < recovery_size):
        recovery_step = step
        recovery_size = current_num_ngrams

    return (False, step, iter, recovery_step, recovery_size)

class LinearEstimator(object):
    """Estimate the coeffients of a line using two most recent points

    Example Usage:

        ls = LinearEstimator()
        ls.AddPoint(x0, y0)
        ls.AddPoint(x1, y1)

        while True:
            xn = Update()
            yn = ls.Estimate(xn)

            if Match(yn):
                break

            ls.AddPoint(xn, yn)
    """
    def __init__(self):
        self.switch = True
        self.x0 = 0.0
        self.x1 = 0.0
        self.y0 = 0.0
        self.y1 = 0.0
        self.n = 0

    def AddPoint(self, x, y):
        if self.switch:
            self.x0 = x
            self.y0 = y
        else:
            self.x1 = x
            self.y1 = y
        self.switch = not self.switch
        self.n += 1

    def Estimate(self, x):
        assert self.n >= 2
        alpha = (self.y1 - self.y0) / (self.x1 - self.x0)
        beta = self.y0 - alpha * self.x0

        return alpha * x + beta

# find threshold in order to match the target-num-ngrams with final LM
#
# Here we fit a linear of log(num-ngrams) versus log(threshold)
# with the latest two points, and approach the target-num-ngrams gradually.
def FindThreshold():
    global current_num_ngrams

    step = 0
    # recovery_step is the index of the most recent step of pruning that resulted in
    # a model with num-ngrams > target_num_ngrams.
    recovery_step = step
    # recovery_size will be set to the number of ngrams in the model at step 'recovery_step'
    recovery_size = 0
    iter = 0
    cur_threshold = args.initial_threshold

    ls = LinearEstimator()

    # get initial two points
    (success, step, iter, recovery_step, recovery_size) = \
        IterateOnce(cur_threshold, step, iter, recovery_step, recovery_size)
    if success:
        return (cur_threshold, iter)

    if current_num_ngrams < args.target_num_ngrams:
        sys.exit("prune_lm_dir.py: --initial-threshold={0} is too big, "
                 "please reduce this value and rerun. "
                 "Number of n-grams pruning with this threshold is {1} "
                 "versus --target-num-ngrams={2}".format(args.initial_threshold,
                     current_num_ngrams, args.target_num_ngrams))

    ls.AddPoint(math.log(current_num_ngrams), math.log(cur_threshold))

    # To get an initial estimate of the log-log linear regression between num-ngrams and
    # pruning threshold, we need at least two data points.  The initial data point is
    # obtained by pruning to the user-specified threshold 'args.initial_threshold'.
    # We get the next data point by multiplying 'cur_threshold' by a value that can be
    # as much as 4.0, but will be less than that if we're relatively close to the
    # final desired model size (because we don't want to overshoot).
    threshold_increase_ratio = min(4.0, (float(current_num_ngrams) / args.target_num_ngrams) ** 0.5)
    cur_threshold = cur_threshold * threshold_increase_ratio

    (success, step, iter, recovery_step, recovery_size) = \
        IterateOnce(cur_threshold, step, iter, recovery_step, recovery_size)
    if success:
        return (cur_threshold, iter)

    ls.AddPoint(math.log(current_num_ngrams), math.log(cur_threshold))

    while True:
        log_threshold = ls.Estimate(math.log(args.target_num_ngrams))
        cur_threshold = math.exp(log_threshold)

        (success, step, iter, recovery_step, recovery_size) = \
            IterateOnce(cur_threshold, step, iter, recovery_step, recovery_size)
        if success:
            return (cur_threshold, iter)

        ls.AddPoint(math.log(current_num_ngrams), log_threshold)

if not os.path.isdir(work_dir):
    try:
        os.makedirs(work_dir)
    except:
        sys.exit("prune_lm_dir.py: error creating directory " + work_dir)


num_words = GetNumWords(args.lm_dir_in)
ngram_order = GetNgramOrder(args.lm_dir_in)
input_num_ngrams = GetTotalNumNgrams(args.lm_dir_in)
initial_num_ngrams = None
current_num_ngrams = None
initial_logprob_per_word = None
final_logprob_per_word = None
waiting_thread = None
logprob_changes = []
logprob_changes_without_dropped_steps = []
thresholds = []

CreateInitialWorkDir()

if args.check_exact_divergence == 'true':
    if steps[-1] != 'EM':
        print("prune_lm_dir.py: --check-exact-divergence=true won't give you the "
              "exact divergence because the last step is not 'EM'.", file=sys.stderr)
    waiting_thread = threading.Thread(target=GetInitialLogprob)
    waiting_thread.start()

if args.target_num_ngrams > 0:
    if MatchTargetSize(input_num_ngrams):
        print ("prune_lm_dir.py:  the input LM is already match the size with target-num-ngrams, do not need any pruning",
            file=sys.stderr)
        sys.exit(0)

    if args.target_num_ngrams > input_num_ngrams:
        sys.exit ("prune_lm_dir.py:  the num-ngrams({0}) of input LM is less than the target-num-ngrams({1}), "
                  "can not do any pruning.".format(input_num_ngrams, args.target_num_ngrams))

    (threshold, iter) = FindThreshold()
    print ("prune_lm_dir.py: Find the threshold "
        + str(threshold) + " in " + str(iter) + " iteration(s)",
        file=sys.stderr)
    print ("prune_lm_dir.py: thresholds per iter were "
       + str(thresholds), file=sys.stderr)
else:
    for step in range(len(steps)):
        logprob_change = RunStep(step, args.final_threshold)
        logprob_changes.append(logprob_change)
        logprob_changes_without_dropped_steps.append(logprob_change)

FinalizeOutput(work_dir + "/step" + str(len(steps)))

if waiting_thread != None:
    waiting_thread.join()

print ("prune_lm_dir.py: log-prob changes per step were "
       + str(logprob_changes), file=sys.stderr)

print ("prune_lm_dir.py: reduced number of n-grams from {0} to {1}, i.e. by {2}%".format(
        initial_num_ngrams, current_num_ngrams,
        100.0 * (initial_num_ngrams - current_num_ngrams) / initial_num_ngrams),
       file=sys.stderr)

print ("prune_lm_dir.py: approximate K-L divergence was {0}".format(-sum(logprob_changes_without_dropped_steps)),
       file=sys.stderr)

if initial_logprob_per_word != None and steps[-1] == 'EM':
    print ("prune_lm_dir.py: exact K-L divergence was {0}".format(
            initial_logprob_per_word - final_logprob_per_word),
           file=sys.stderr)

# clean up the work directory.
if args.cleanup == 'true':
    shutil.rmtree(work_dir)

if os.system("validate_lm_dir.py " + args.lm_dir_out) != 0:
    sys.exit("prune_lm_dir.py: failed to validate output LM-dir")
