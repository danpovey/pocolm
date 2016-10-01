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

parser = argparse.ArgumentParser(description="This script trains an n-gram language model with <order> "
                                 "from <text-dir> and writes out the model to <lm-dir>. "
                                 "The output model dir is in pocolm-format, user can call "
                                 "format_arpa_lm.py with <lm-dir> to get a ARPA-format model. "
                                 "Pruning a model could be achieve by call prune_lm_dir.py with <lm-dir>."
                                 "If --lm-dir is not specified, the model would be written into a subdirectory of <work_dir>, "
                                 "see help of --lm-dir for details. "
                                 "Example usage: "
                                 "  'train_lm.py --num-words=20000 --num-splits=5 --warm-start-ratio=10 "
                                 "               --max-memory=10G data/text 3 data/work data/lm/20000_3.pocolm'"
                                 " or "
                                 "  'train_lm.py --wordlist=foo.txt --num-splits=5 --warm-start-ratio=10"
                                 "               --max-memory=10G data/text 3 data/work data/lm/foo_3.pocolm'",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--wordlist", type=str, default=None,
                    help="A file contains the list of words one wish to use during the training. "
                    "If not specified, it will generate vocab from training text.")
parser.add_argument("--num-words", type=int, default=0,
                    help="The maximum size of vocab. if this is non-positive, the vocab is "
                    "unlimited.")
parser.add_argument("--num-splits", type=int, default=1,
                    help="Number of parallel processes would be used during training.")
parser.add_argument("--warm-start-ratio", type=int, default=10,
                    help="number by which it divide the data to get the warm start for "
                    "metaparameters optimization  If <= 1, we skip the warm-start"
                    "and do the optimization with all the data from scratch.")
parser.add_argument("--min-counts", type=str, default='',
                    help="If specified, apply min-count when we get the ngram counts from training text. "
                         "run 'get_counts.py -h' to see the details on how to set this option.")
parser.add_argument("--limit-unk-history", type=str, default='false',
                    choices=['true','false'],
                    help="If true, the left words of <unk> in history of a n-gram will be truncated. "
                    "run 'get_counts.py -h' to see the details on how to set this option.")
parser.add_argument("--fold-dev-into", type=str,
                    help="If supplied, the name of data-source into which to fold the "
                    "counts of the dev data when building the model (typically the "
                    "same data source from which the dev data was originally excerpted). "
                    "run 'make_lm_dir.py -h' to see the details on how to set this option.")
parser.add_argument("--bypass-metaparameter-optimization", type=str, default=None,
                    help="This option accepts a string encoding the metaparameters as "
                    "a comma separated list. If this is specified, the stages of metaparameter optimization "
                    "would be completely bypassed. One can get the approaviate numbers after "
                    "running one time of train_lm.py.  Caution: if you change the data "
                    "or the options, the values are no longer valid and you should "
                    "remove this option.")
parser.add_argument("--verbose", type=str, default='false',
                    choices=['true','false'],
                    help="If true, print commands as we execute them.")
parser.add_argument("--cleanup",  type=str, choices=['true','false'],
                    default='true', help='Set this to false to disable clean up of the '
                    'work directory.')
parser.add_argument("--keep-int-data",  type=str, choices=['true','false'],
                    default='false', help='whether to avoid the int-dir being cleanuped. '
                    'This is useful when user trains different orders of model with the same int-data. '
                    'It is valid only when --cleanup=true')
parser.add_argument("--max-memory", type=str, default='',
                    help="Memory limitation for sort called by get_counts.py.")
parser.add_argument("text_dir",
                    help="Directory containing the training text.")
parser.add_argument("order",
                    help="Order of N-gram model to be trained.")
parser.add_argument("work_dir",
                    help="Working directory for building the language model.")
parser.add_argument("lm_dir", type=str, default='', nargs='?',
                    help="Output directory where the language model is created."
                    "If this is not specified, the output directory would be a subdirectory under <work-dir>, "
                    "with name '<vocab_name>_<order>.pocolm', where the <vocab_name> will be the name of wordlist "
                    "if --wordlist is specified otherwise the size of vocabulary, and <order> is the ngram order of model.")


args = parser.parse_args()
# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

if args.num_words < 0:
    sys.exit("train_lm.py: --num-words must be >=0.")

if args.num_splits < 1:
    sys.exit("train_lm.py: --num-splits must be >=1.")

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
                    sys.exit("train_lm.py: --max-memory should be formatted as "
                             "'a positive integer' or 'a positive integer appended "
                             "with 'b', 'K', 'M','G', or '%''.")
            # max memory size must be larger than zero
            if int(s[:-1]) == 0:
                sys.exit("train_lm.py: --max-memory must be > 0 {unit}.".format(
                         unit = s[-1]))
        else:
            sys.exit("train_lm.py: the format of string --max-memory is not correct.")
    else:
         sys.exit("train_lm.py: the lenght of string --max-memory must >= 2.")

if args.lm_dir != '':
  work_dir = args.work_dir
else:
  work_dir = os.path.join(args.work_dir, 'work')

log_dir = os.path.join(work_dir, "log")
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)

def GetNumNgrams(lm_dir_in):
    tot_num_ngrams = 0
    num_ngrams = []
    f = open(os.path.join(lm_dir_in, "num_ngrams"));
    for line in f:
        n = int(line.split()[1])
        num_ngrams.append(n)
        tot_num_ngrams += n
    f.close()
    num_ngrams.append(tot_num_ngrams)

    return num_ngrams

def ReadMetaparameters(metaparameter_file):
    metaparameters = []
    f = open(metaparameter_file);
    for line in f:
        metaparameters.append(float(line.split()[1]))
    f.close()

    return metaparameters

def WriteMetaparameters(metaparameters, ngram_order, num_train_sets, out_file):
    assert(len(metaparameters) == (ngram_order - 1) * 4 + num_train_sets)
    f = open(out_file, "w")
    i = 0
    for n in range(1, num_train_sets + 1):
        print("count_scale_{0}".format(n), metaparameters[i], file=f)
        i += 1

    for o in range(2, ngram_order + 1):
        for n in range(4):
            print("order{0}_D{1}".format(o, n + 1), metaparameters[i], file=f)
            i += 1
    f.close()

def FormatMetaparameters(metaparameters):
    out = []
    for param in metaparameters:
        x = '{:.3f}'.format(param)
        if x == '0.00':
            x = '{:.3g}'.format(param)
        out.append(x)

    return ','.join(out)

def ParseMetaparameters(encoded_str, ngram_order, num_train_sets):
    metaparameters = encoded_str.split(',')
    assert(len(metaparameters) == (ngram_order - 1) * 4 + num_train_sets)
    map(lambda x: float(x), metaparameters)

    return metaparameters

def CheckFreshness(tgt_file, ref_files):
    if not os.path.exists(tgt_file):
      return True

    for f in ref_files:
      if os.path.getmtime(tgt_file) < os.path.getmtime(f):
        return True

    return False

# get word counts
word_counts_dir = os.path.join(work_dir, 'word_counts')
if os.system("validate_text_dir.py " + args.text_dir) != 0:
    sys.exit(1)
last_done_files = []
for f in os.listdir(args.text_dir):
    if f.endswith(".txt") or f.endswith(".txt.gz"):
        last_done_files.append(os.path.join(args.text_dir, f))
done_file = os.path.join(word_counts_dir, '.done')
if not CheckFreshness(done_file, last_done_files):
    LogMessage("Skip getting word counts")
else:
    log_file = os.path.join(log_dir, 'get_word_counts.log')
    LogMessage("Getting word counts... log in " + log_file)
    command = "get_word_counts.py {0} {1}".format(args.text_dir, word_counts_dir)
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# get unigram weights
unigram_weights = os.path.join(args.text_dir, 'unigram_weights')
last_done_files = [done_file]
done_file = os.path.join(work_dir, '.unigram_weights.done')
if not CheckFreshness(done_file, last_done_files):
    LogMessage("Skip getting unigram weights")
else:
    log_file = os.path.join(log_dir, 'get_unigram_weights.log')
    LogMessage("Getting unigram weights... log in " + log_file)
    command = "get_unigram_weights.py {0} > {1}".format(word_counts_dir, unigram_weights)
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# generate vocab
vocab_name = ''
vocab = ''
if args.wordlist == None:
    if args.num_words > 0:
        vocab_name = str(args.num_words)
        log_dir = os.path.join(work_dir, 'log', vocab_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        vocab = os.path.join(work_dir, 'vocab_' + vocab_name + '.txt')
        last_done_files = [done_file]
        done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
        if not CheckFreshness(done_file, last_done_files):
            LogMessage("Skip generating vocab")
        else:
            log_file = os.path.join(log_dir, 'word_counts_to_vocab.log')
            LogMessage("Generating vocab with num-words={0} ... log in {1}".format(
                args.num_words, log_file))
            command = "word_counts_to_vocab.py --num-words={0} {1} > {2}".format(args.num_words, \
                    word_counts_dir,  vocab)
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)
    else:
        vocab_name = 'unlimited'
        log_dir = os.path.join(work_dir,'log', vocab_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        vocab = os.path.join(work_dir, 'vocab_' + vocab_name + '.txt')
        last_done_files = [done_file]
        done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
        if not CheckFreshness(done_file, last_done_files):
            LogMessage("Skip generating vocab")
        else:
            log_file = os.path.join(log_dir, 'word_counts_to_vocab.log')
            LogMessage("Generating vocab with unlmited num-words ... log in " + log_file)
            command = "word_counts_to_vocab.py {0} > {1}".format(word_counts_dir,  vocab)
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)
else:
    if args.num_words < 0:
        LogMessage("Ignoring --num-words because --wordlist is specified")

    vocab_name = os.path.basename(args.wordlist)
    log_dir = os.path.join(work_dir, 'log', vocab_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    vocab = os.path.join(work_dir, 'vocab_' + vocab_name + '.txt')
    last_done_files = [done_file]
    done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
    if not CheckFreshness(done_file, last_done_files):
        LogMessage("Skip generating vocab")
    else:
        log_file = os.path.join(log_dir, 'wordlist_to_vocab.log')
        LogMessage("Generating vocab with wordlist[{0}]... log in {1}".format(
            args.wordlist, log_file))
        command = "wordlist_to_vocab.py {0} > {1}".format(args.wordlist, vocab)
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

# preparing int data
int_dir = os.path.join(work_dir, 'int_' + vocab_name)
last_done_files = [done_file]
done_file = os.path.join(int_dir, '.done')
if not CheckFreshness(done_file, last_done_files):
    LogMessage("Skip preparing int data")
else:
    log_file = os.path.join(log_dir, 'prepare_int_data.log')
    LogMessage("Preparing int data... log in " + log_file)
    command = "prepare_int_data.py {0} {1} {2}".format(args.text_dir, vocab, int_dir)
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# get ngram counts
lm_name = vocab_name + '_' + str(args.order)
if args.min_counts != '':
    # replace '=' to '-', since '=' need to be escaped in shell
    lm_name += '_' + '_'.join(args.min_counts.replace('=', '-').split())
log_dir = os.path.join(work_dir, 'log', lm_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
counts_dir = os.path.join(work_dir, 'counts_' + lm_name)
last_done_files = [done_file]
done_file = os.path.join(counts_dir, '.done')
if not CheckFreshness(done_file, last_done_files):
    LogMessage("Skip getting counts")
else:
    log_file = os.path.join(log_dir, 'get_counts.log')
    LogMessage("Getting ngram counts... log in " + log_file)
    command = "get_counts.py --min-counts='{0}' --max-memory={1} --limit-unk-history={5} {2} {3} {4}".format(
            args.min_counts, args.max_memory, int_dir, args.order, counts_dir,
            args.limit_unk_history)
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# cleanup int dir
if args.cleanup == 'true' and args.keep_int_data == 'false':
    if os.system("cleanup_int_dir.py " + int_dir) != 0:
        sys.exit("train_lm.py: failed to cleanup int dir: " + int_dir)
    os.remove(os.path.join(int_dir, '.done'))

metaparam_file = ''
if args.bypass_metaparameter_optimization != None:
    LogMessage("Bypass optimization steps")

    for name in [ 'ngram_order', 'num_train_sets' ]:
        f = open(os.path.join(counts_dir, name))
        globals()[name] = int(f.readline())
        f.close()

    metaparameters = ParseMetaparameters(args.bypass_metaparameter_optimization,
        ngram_order, num_train_sets)
    metaparam_file = os.path.join(work_dir, 'bypass.metaparams')
    WriteMetaparameters(metaparameters, ngram_order, num_train_sets, metaparam_file)
else:
    if args.warm_start_ratio > 1:
        # Do a first pass of metaparameter optimization with a subset of the
        # data (it gives a better starting point for the final metaparameter
        # optimization).

        # subset counts dir
        subset_counts_dir = counts_dir + '_subset' + str(args.warm_start_ratio)
        last_done_files = [done_file]
        done_file = os.path.join(subset_counts_dir, '.done')
        if not CheckFreshness(done_file, last_done_files):
            LogMessage("Skip subsetting counts dir")
        else:
            log_file = os.path.join(log_dir, 'subset_count_dir.log')
            LogMessage("Subsetting counts dir... log in " + log_file)
            command = "subset_count_dir.sh {0} {1} {2}".format(counts_dir, \
                                                               args.warm_start_ratio, subset_counts_dir)
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)

        # warm-start optimize metaparameters
        subset_optimize_dir = os.path.join(work_dir, "optimize_{0}_subset{1}".format(lm_name, \
                                                                                     args.warm_start_ratio))
        last_done_files = [done_file]
        done_file = os.path.join(subset_optimize_dir, '.done')
        if not CheckFreshness(done_file, last_done_files):
            LogMessage("Skip warm-start optimizing metaparameters")
        else:
            log_file = os.path.join(log_dir, 'optimize_metaparameters_warm_start.log')
            LogMessage("Optimizing metaparameters for warm-start... log in " + log_file)
            command = "optimize_metaparameters.py --cleanup={3} --progress-tolerance=1.0e-05 --num-splits={0} {1} {2}".format(
                args.num_splits, subset_counts_dir, subset_optimize_dir, args.cleanup)
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)

        # cleanup subset counts dir
        if args.cleanup == 'true':
            if os.system("cleanup_count_dir.py " + subset_counts_dir) != 0:
                sys.exit("train_lm.py: failed to cleanup subset count dir: " + subset_counts_dir)
            os.remove(os.path.join(subset_counts_dir, '.done'))
        warm_start_opt = ("--gradient-tolerance=0.01 --progress-tolerance=1.0e-03 "
                          "--warm-start-dir=" + subset_optimize_dir)
    else:
        warm_start_opt = ""

    # optimize metaparameters
    optimize_dir = os.path.join(work_dir, "optimize_{0}".format(lm_name))
    last_done_files = [done_file]
    done_file = os.path.join(optimize_dir, '.done')
    if not CheckFreshness(done_file, last_done_files):
        LogMessage("Skip optimizing metaparameters")
    else:
        log_file = os.path.join(log_dir,'optimize_metaparameters.log')
        LogMessage("Optimizing metaparameters... log in " + log_file)
        command = "optimize_metaparameters.py {0} \
                   --num-splits={1} {2} {3}".format(warm_start_opt,
                args.num_splits, counts_dir, optimize_dir)
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

    metaparam_file = os.path.join(optimize_dir, 'final.metaparams')
    metaparameters = ReadMetaparameters(metaparam_file)
    LogMessage("You can set --bypass-metaparameter-optimization='{0}' "
               "to get equivalent results".format(
                   FormatMetaparameters(metaparameters)))

# make lm dir
if args.lm_dir != '':
  lm_dir = args.lm_dir
else:
  lm_dir = os.path.join(args.work_dir, lm_name + '.pocolm')

last_done_files = [done_file]
done_file = os.path.join(lm_dir, '.done')
if not CheckFreshness(done_file, last_done_files):
    LogMessage("Skip making lm dir")
else:
    log_file = os.path.join(log_dir, 'make_lm_dir.log')
    LogMessage("Making lm dir... log in " + log_file)
    opts = []
    if args.num_splits > 1:
        opts.append('--keep-splits=true')
    if args.fold_dev_into != None:
        opts.append('--fold-dev-into=' + args.fold_dev_into)
    command = "make_lm_dir.py --cleanup={5} --num-splits={0} {1} {2} {3} {4}".format(
            args.num_splits, ' '.join(opts), counts_dir, metaparam_file, lm_dir, args.cleanup)
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# cleanup subset counts dir
if args.cleanup == 'true':
    if os.system("cleanup_count_dir.py " + counts_dir) != 0:
        sys.exit("train_lm.py: failed to cleanup count dir: " + counts_dir)
    os.remove(os.path.join(counts_dir, '.done'))

if os.system("validate_lm_dir.py " + lm_dir) != 0:
    sys.exit("train_lm.py: failed to validate output LM-dir")

num_ngrams = GetNumNgrams(lm_dir)
line = "Ngram counts: "
for order in range(len(num_ngrams) - 2):
    line += str(num_ngrams[order]) + ' + '
line += str(num_ngrams[-2]) + ' = ' + str(num_ngrams[-1])
LogMessage("" + line)

LogMessage("Success to train lm, output dir is {0}.".format(lm_dir))
LogMessage("You may call format_arpa_lm.py to get ARPA-format lm, ")
LogMessage("Or call prune_lm_dir.py to prune the lm.")
# print the final lm dir to the caller
print(lm_dir)
