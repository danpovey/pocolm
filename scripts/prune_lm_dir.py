#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, threading
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script takes an lm-dir, as produced by make_lm_dir.py, "
                                 "that should not have the counts split up into pieces, and it prunes "
                                 "the counts and writes out to a new lm-dir.")

parser.add_argument("--steps", type=str,
                    default='prune*0.25 EM EM EM prune*0.5 EM EM EM prune*1.0 EM EM EM prune*1.0 EM EM EM',
                    help='This string specifies a sequence of steps in the pruning sequence.'
                    'prune*X, with X <= 1.0, tells it to prune with X times the threshold '
                    'specified with the --threshold option.  EM specifies one iteration of '
                    'E-M on the model. ')
parser.add_argument("--remove-zeros", type=str, choices=['true','false'],
                    default='true', help='Set this to false to disable an optimization. '
                    'Only useful for debugging purposes.')
parser.add_argument("--check-exact-divergence", type=str, choices=['true','false'],
                    default='true', help='')
parser.add_argument("lm_dir_in",
                    help="Source directory, for the input language model.")
parser.add_argument("threshold", type=float,
                    help="Threshold for pruning, e.g. 0.5, 1.0, 2.0, 4.0.... "
                    "larger threshold will give you more highly-pruned models."
                    "Threshold is interpreted as entropy-change times overall "
                    "weighted data count, for each parameter.  It should be "
                    "larger if you have more data, assuming you want the "
                    "same-sized models.")
parser.add_argument("lm_dir_out",
                    help="Output directory where the language model is created.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    sys.exit("prune_lm_dir.py: failed to validate input LM-dir")

num_splits = None
if os.path.exists(args.lm_dir_in + "/num_splits"):
    f = open(args.lm_dir_in + "/num_splits")
    num_splits = int(f.readline())
    f.close()

if args.threshold <= 0.0:
    sys.exit("prune_lm_dir.py: --threshold must be positive: got " + str(args.threshold))

work_dir = args.lm_dir_out + "/work"

steps = args.steps.split()

if len(steps) == 0:
    sys.exit("prune_lm_dir.py: 'steps' cannot be empty.")

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

def RunCommand(command):
    # print the command for logging
    print(command, file=sys.stderr)
    if os.system(command) != 0:
        sys.exit("get_objf_and_derivs_split.py: error running command: " + command)




# This script creates work/protected.all (listing protected
# counts which may not be removed); it requires work/float.all
# to exist.
def CreateProtectedCounts(work):
    command = ('float-counts-to-histories <{0}/float.all | LC_ALL=C sort |'
               ' histories-to-null-counts >{0}/protected.all'.format(work))
    RunCommand(command)


def SoftLink(src, dest):
    if os.path.exists(dest):
        os.remove(dest)
    try:
        os.symlink(os.path.abspath(src), dest)
    except:
        sys.exit("prune_lm_dir.py: error linking {0} to {1}".format(src, dest))

def CreateInitialWorkDir():
    # Creates float.all, stats.all, and protected.all in work_dir/iter
    work0dir = work_dir + "/iter0"
    # create float.all
    if not os.path.isdir(work0dir):
        os.makedirs(work0dir)
    if num_splits == None:
        SoftLink(args.lm_dir_in + "/float.all", work0dir + "/float.all")
    else:
        splits_star = ' '.join([ args.lm_dir_in + "/float.all." + str(n)
                                 for n in range(1, num_splits + 1) ])
        command = "merge-float-counts " + splits_star + " >{0}/float.all".format(work0dir)
        RunCommand(command)

    # create protected.all
    CreateProtectedCounts(work0dir)

    stats_star = ' '.join([ "{0}/stats.{1}".format(work0dir, n)
                            for n in range(1, ngram_order + 1) ])

    # create stats.{1,2,3..}
    # e.g. command = 'float-counts-to-float-stats 20000 foo/work/iter0/stats.1 '
    #                'foo/work/iter0/stats.2 <foo/work/iter0/float.all'
    command = ("float-counts-to-float-stats {0} ".format(num_words) +
               stats_star +
               " <{0}/float.all".format(work0dir))
    RunCommand(command)
    command = "merge-float-counts {0} > {1}/stats.all".format(
        stats_star, work0dir)
    RunCommand(command)
    command = "rm " + stats_star
    RunCommand(command)

# sets initial_logprob_per_word.
def GetInitialLogprob():
    work0dir = work_dir + "/iter0"
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
        assert p.stdout.readline() == ''
        ret = p.wait()
        assert ret == 0
        global final_num_ngrams, initial_num_ngrams
        if initial_num_ngrams == None:
            initial_num_ngrams = int(tot_ngrams)
        final_num_ngrams = int(tot_ngrams) - int(pruned)
    except Exception as e:
        sys.exit("prune_lm_dir.py: error running command '{0}', error is '{1}'".format(
                command, str(e)))

    if args.remove_zeros == 'false':
        # create work_out/float.all.
        command = 'merge-float-counts {0} >{1}/float.all'.format(float_star, work_out)
        RunCommand(command)
        command = 'rm ' + float_star
        RunCommand(command)
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
        RunCommand(command)

        # create work_out/stats.all
        command = 'merge-float-counts {0} >{1}/stats.all'.format(stats_star, work_out)
        RunCommand(command)
        command = 'rm ' + float_star + ' ' + stats_star
        RunCommand(command)

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
        global final_logprob_per_word
        final_logprob_per_word = tot_like / tot_count
        for i in range(2, len(a)):  # for each n-gram order
            like_change += float(a[i])
        like_change_per_word = like_change / tot_count
    except Exception as e:
        sys.exit("prune_lm_dir.py: error running command '{0}', error is '{1}'".format(
                command, str(e)))

    command = 'merge-float-counts {0} >{1}/float.all'.format(float_star, work_out)
    RunCommand(command)
    command = 'rm ' + float_star
    RunCommand(command)
    # soft-link work_out/stats.all to work_in/stats.all
    SoftLink(work_in + "/stats.all",
             work_out + "/stats.all")
    # soft-link work_out/protected.all to work_in/protected.all
    SoftLink(work_in + "/protected.all",
             work_out + "/protected.all")
    return like_change_per_word


# runs one of the numbered steps.  step_number >= 0 is the number of the work
# directory we'll get the input from (the output will be that plus one).
# returns the expected log-prob change (on data generated from the model
# itself.. this will be negative for pruning steps and positive for E-M steps.
def RunStep(step_number):
    work_in = work_dir + "/iter" + str(step_number)
    work_out = work_dir + "/iter" + str(step_number + 1)
    if not os.path.isdir(work_out):
        os.makedirs(work_out)
    step_text = steps[step_number]
    if step_text[0:6] == 'prune*':
        try:
            scale = float(step_text[6:])
            assert scale <= 1.0
        except:
            sys.exit("prune_lm_dir.py: invalid step (wrong --steps "
                     "option): '{0}'".format(step_text))
        return RunPruneStep(work_in, work_out, args.threshold * scale)

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


if not os.path.isdir(work_dir):
    try:
        os.makedirs(work_dir)
    except:
        sys.exit("prune_lm_dir.py: error creating directory " + work_dir)


num_words = GetNumWords(args.lm_dir_in)
ngram_order = GetNgramOrder(args.lm_dir_in)
initial_num_ngrams = None
final_num_ngrams = None
initial_logprob_per_word = None
final_logprob_per_word = None
waiting_thread = None
logprob_changes = []

CreateInitialWorkDir()

if args.check_exact_divergence == 'true':
    if steps[-1] != 'EM':
        print("prune_lm_dir.py: --check-exact-divergence=true won't give you the "
              "exact divergence because the last step is not 'EM'.", file=sys.stderr)
    waiting_thread = threading.Thread(target=GetInitialLogprob)
    waiting_thread.start()

for step in range(len(steps)):
    logprob_changes.append(RunStep(step))

FinalizeOutput(work_dir + "/iter" + str(len(steps)))

if waiting_thread != None:
    waiting_thread.join()

print ("prune_lm_dir.py: log-prob changes per iter were "
       + str(logprob_changes), file=sys.stderr)

print ("prune_lm_dir.py: reduced number of n-grams from {0} to {1}, i.e. by {2}%".format(
        initial_num_ngrams, final_num_ngrams,
        100.0 * (initial_num_ngrams - final_num_ngrams) / initial_num_ngrams),
       file=sys.stderr)

print ("prune_lm_dir.py: approximate K-L divergence was {0}".format(-sum(logprob_changes)),
       file=sys.stderr)

if initial_logprob_per_word != None and steps[-1] == 'EM':
    print ("prune_lm_dir.py: exact K-L divergence was {0}".format(
            initial_logprob_per_word - final_logprob_per_word),
           file=sys.stderr)

# clean up the work directory.
shutil.rmtree(work_dir)

if os.system("validate_lm_dir.py " + args.lm_dir_out) != 0:
    sys.exit("split_lm_dir.py: failed to validate output LM-dir")
