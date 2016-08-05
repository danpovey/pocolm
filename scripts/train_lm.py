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
                                 "Pruning a model could be achieve by call prune_lm_dir.py with <lm-dir>.",
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
                    "metaparameters optimization.")
parser.add_argument("--min-counts", type=str, default='',
                    help="If specified, apply min-count when we get the ngram counts from training text.")
parser.add_argument("--bypass-metaparameter-optimization", type=str, default=None,
                    help="This option accepts a string encoding the metaparameters as "
                    "a comma separated list. If this is specified, the stages of metaparameter optimization "
                    "would be completely bypassed. One can get the approaviate numbers after "
                    "running one time of train_lm.py.")
parser.add_argument("--skip-computing-ppl", type=str, default='false',
                    choices=['true','false'],
                    help="If true, skip computing final perplexity of dev set.")
parser.add_argument("--verbose", type=str, default='false',
                    choices=['true','false'],
                    help="If true, print commands as we execute them.")
parser.add_argument("--cleanup",  type=str, choices=['true','false'],
                    default='true', help='Set this to false to disable clean up of the '
                    'work directory.')
parser.add_argument("text_dir",
                    help="Directory containing the training text.")
parser.add_argument("order",
                    help="Order of N-gram model to be trained.")
parser.add_argument("lm_dir",
                    help="Output directory where the language model is created.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

if args.num_words < 0:
    sys.exit("train_lm.py: --num-splits must be >=0.")

if args.num_splits < 1:
    sys.exit("train_lm.py: --num-splits must be >=1.")

if args.warm_start_ratio < 1:
    sys.exit("train_lm.py: --warm-start-ratio must be >=1.")

work_dir = args.lm_dir + os.sep + 'work' + os.sep
if not os.path.isdir(work_dir + "/log"):
    os.makedirs(work_dir + "/log")

def GetNumNgrams(lm_dir_in):
    tot_num_ngrams = 0
    num_ngrams = []
    f = open(lm_dir_in + "/num_ngrams");
    for line in f:
        n = int(line.split()[1])
        num_ngrams.append(n)
        tot_num_ngrams += n
    num_ngrams.append(tot_num_ngrams)

    return num_ngrams

# get word counts
word_counts_dir = work_dir + 'word_counts'
if os.path.isdir(word_counts_dir):
    print("train_lm.py: Skip getting word counts", file=sys.stderr)
else:
    print("train_lm.py: Getting word counts...", file=sys.stderr)
    command = "get_word_counts.py {0} {1}".format(args.text_dir, word_counts_dir)
    log_file = work_dir + '/log/get_word_counts.log'
    RunCommand(command, log_file, args.verbose == 'true')

# get unigram weights
unigram_weights = work_dir + 'unigram_weights'
if os.path.exists(unigram_weights):
    print("train_lm.py: Skip getting unigram weights", file=sys.stderr)
else:
    print("train_lm.py: Getting unigram weights...", file=sys.stderr)
    command = "get_unigram_weights.py {0} > {1}".format(word_counts_dir, unigram_weights)
    log_file = work_dir + '/log/get_unigram_weights.log'
    RunCommand(command, log_file, args.verbose == 'true')

# generate vocab
vocab_name = ''
vocab = ''
if args.wordlist == None:
    if args.num_words > 0:
        vocab_name = str(args.num_words)
        vocab = work_dir + 'vocab_' + vocab_name + '.txt'
        if os.path.exists(vocab):
            print("train_lm.py: Skip generating vocab", file=sys.stderr)
        else:
            print("train_lm.py: Generating vocab with num-words={0} ...".format(args.num_words), file=sys.stderr)
            command = "word_counts_to_vocab.py --num-words={0} {1} > {2}".format(args.num_words, \
                    word_counts_dir,  vocab)
            log_file = work_dir + '/log/word_counts_to_vocab.log'
            RunCommand(command, log_file, args.verbose == 'true')
    else:
        vocab_name = 'unlimited'
        vocab = work_dir + 'vocab_' + vocab_name + '.txt'
        if os.path.exists(vocab):
            print("train_lm.py: Skip generating vocab", file=sys.stderr)
        else:
            print("train_lm.py: Generating vocab with unlmited num-words ...", file=sys.stderr)
            command = "word_counts_to_vocab.py {0} > {1}".format(word_counts_dir,  vocab)
            log_file = work_dir + '/log/word_counts_to_vocab.log'
            RunCommand(command, log_file, args.verbose == 'true')
else:
    if args.num_words < 0:
        print("train_lm.py: Ignoring --num-words because --wordlist is specified", file=sys.stderr)

    vocab_name = args.wordlist
    vocab = work_dir + 'vocab_' + vocab_name + '.txt'
    if os.path.exists(vocab):
        print("train_lm.py: Skip generating vocab", file=sys.stderr)
    else:
        print("train_lm.py: Generating vocab with wordlist[{0}]...".format(args.wordlist), file=sys.stderr)
        command = "wordlist_to_vocab.py {1} > {2}".format(word_counts_dir, vocab)
        log_file = work_dir + '/log/wordlist_to_vocab.log'
        RunCommand(command, log_file, args.verbose == 'true')

# preparing int data
int_dir = work_dir + 'int_' + vocab_name
if os.path.isdir(int_dir):
    print("train_lm.py: Skip preparing int data", file=sys.stderr)
else:
    print("train_lm.py: Preparing int data...", file=sys.stderr)
    command = "prepare_int_data.py {0} {1} {2}".format(args.text_dir, vocab, int_dir)
    log_file = work_dir + '/log/prepare_int_data.log'
    RunCommand(command, log_file, args.verbose == 'true')

# get ngram counts
counts_dir = work_dir + 'counts_' + vocab_name + '_' + str(args.order)
if os.path.isdir(counts_dir):
    print("train_lm.py: Skip getting counts", file=sys.stderr)
else:
    print("train_lm.py: Getting ngram counts", file=sys.stderr)
    command = "get_counts.py --min-counts='{0}' {1} {2} {3}".format(args.min_counts, \
            int_dir, args.order, counts_dir)
    log_file = work_dir + '/log/get_counts.log'
    RunCommand(command, log_file, args.verbose == 'true')

# subset counts dir
subset_counts_dir = counts_dir + '_subset' + str(args.warm_start_ratio)
if os.path.isdir(subset_counts_dir):
    print("train_lm.py: Skip subsetting counts dir", file=sys.stderr)
else:
    print("train_lm.py: Subsetting counts dir...", file=sys.stderr)
    command = "subset_count_dir.sh {0} {1} {2}".format(counts_dir, \
            args.warm_start_ratio, subset_counts_dir)
    log_file = work_dir + '/log/subset_count_dir.log'
    RunCommand(command, log_file, args.verbose == 'true')

# warm-start optimize metaparameters
subset_optimize_dir = work_dir + "optimize_{0}_{1}_subset{2}".format(vocab_name, \
        args.order, args.warm_start_ratio)
if os.path.isdir(subset_optimize_dir):
    print("train_lm.py: Skip warm-start optimizing metaparameters", file=sys.stderr)
else:
    print("train_lm.py: Optimizing metaparameters for warm-start...", file=sys.stderr)
    command = "optimize_metaparameters.py --progress-tolerance=1.0e-05 --num-splits={0} {1} {2}".format(
            args.num_splits, subset_counts_dir, subset_optimize_dir)
    log_file = work_dir + '/log/optimize_metaparameters_warm_start.log'
    RunCommand(command, log_file, args.verbose == 'true')

# optimize metaparameters
optimize_dir = work_dir + "optimize_{0}_{1}".format(vocab_name, args.order)
if os.path.isdir(optimize_dir):
    print("train_lm.py: Skip optimizing metaparameters", file=sys.stderr)
else:
    print("train_lm.py: Optimizing metaparameters", file=sys.stderr)
    command = "optimize_metaparameters.py --warm-start-dir={0} \
               --progress-tolerance=1.0e-03 --gradient-tolerance=0.01 \
               --num-splits={1} {2} {3}".format(subset_optimize_dir,
            args.num_splits, counts_dir, optimize_dir)
    log_file = work_dir + '/log/optimize_metaparameters.log'
    RunCommand(command, log_file, args.verbose == 'true')

# make lm dir
lm_dir = args.lm_dir + os.sep + vocab_name + '_' + str(args.order) + '.pocolm'
if os.path.isdir(lm_dir):
    print("train_lm.py: Skip making lm dir", file=sys.stderr)
else:
    print("train_lm.py: Making lm dir...", file=sys.stderr)
    opts = []
    if args.num_splits > 1:
        opts.append('--keep-splits=true')
    command = "make_lm_dir.py --num-splits={0} {1} {2} {3} {4}".format(
            args.num_splits, ' '.join(opts), counts_dir, optimize_dir + os.sep + 'final.metaparams', lm_dir)
    log_file = work_dir + '/log/make_lm_dir.log'
    RunCommand(command, log_file, args.verbose == 'true')

if os.system("validate_lm_dir.py " + lm_dir) != 0:
    sys.exit("train_lm.py: failed to validate output LM-dir")

num_ngrams = GetNumNgrams(lm_dir)
line = "Ngram counts: "
for order in range(len(num_ngrams) - 2):
    line += str(num_ngrams[order]) + ' + '
line += str(num_ngrams[-2]) + ' = ' + str(num_ngrams[-1])
print("train_lm.py: " + line, file=sys.stderr)

if args.skip_computing_ppl == 'false':
    print("train_lm.py: Computing perplexity for dev set...", file=sys.stderr)
    command = "get_data_prob.py {0} {1} 2>&1 | grep -F '[perplexity'".format(
            args.text_dir + os.sep + 'dev.txt', lm_dir)
    log_file = work_dir + '/log/get_data_prob.log'
    output = GetCommandStdout(command, log_file, args.verbose == 'true')
    for line in output.split('\n'):
        m = re.search('\[perplexity = (.*)\]', line)
        if m:
            ppl = m.group(1)
            print("train_lm.py: Perplexity: {0}".format(ppl), file=sys.stderr)

## clean up the work directory.
#if args.cleanup == 'true':
#    shutil.rmtree(work_dir)
