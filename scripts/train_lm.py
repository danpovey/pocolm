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
                    help="If specified, apply min-count when we get the ngram counts from training text. "
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
                    "running one time of train_lm.py.")
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

work_dir = os.path.join(args.lm_dir, 'work')
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

# get word counts
word_counts_dir = os.path.join(work_dir, 'word_counts')
done_file = os.path.join(word_counts_dir, '.done')
if os.path.exists(done_file):
    LogMessage("Skip getting word counts")
else:
    LogMessage("Getting word counts...")
    command = "get_word_counts.py {0} {1}".format(args.text_dir, word_counts_dir)
    log_file = os.path.join(log_dir, 'get_word_counts.log')
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# get unigram weights
unigram_weights = os.path.join(args.text_dir, 'unigram_weights')
done_file = os.path.join(work_dir, '.unigram_weights.done')
if os.path.exists(done_file):
    LogMessage("Skip getting unigram weights")
else:
    LogMessage("Getting unigram weights...")
    command = "get_unigram_weights.py {0} > {1}".format(word_counts_dir, unigram_weights)
    log_file = os.path.join(log_dir, 'get_unigram_weights.log')
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
        done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
        if os.path.exists(done_file):
            LogMessage("Skip generating vocab")
        else:
            LogMessage("Generating vocab with num-words={0} ...".format(args.num_words))
            command = "word_counts_to_vocab.py --num-words={0} {1} > {2}".format(args.num_words, \
                    word_counts_dir,  vocab)
            log_file = os.path.join(log_dir, 'word_counts_to_vocab.log')
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)
    else:
        vocab_name = 'unlimited'
        log_dir = os.path.join(work_dir,'log', vocab_name)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        vocab = os.path.join(work_dir, 'vocab_' + vocab_name + '.txt')
        done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
        if os.path.exists(done_file):
            LogMessage("Skip generating vocab")
        else:
            LogMessage("Generating vocab with unlmited num-words ...")
            command = "word_counts_to_vocab.py {0} > {1}".format(word_counts_dir,  vocab)
            log_file = os.path.join(log_dir, 'word_counts_to_vocab.log')
            RunCommand(command, log_file, args.verbose == 'true')
            TouchFile(done_file)
else:
    if args.num_words < 0:
        LogMessage("Ignoring --num-words because --wordlist is specified")

    vocab_name = os.path.basename(args.wordlist)
    log_dir = os.path.join(work_dir,'log', vocab_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    vocab = os.path.join(work_dir, 'vocab_' + vocab_name + '.txt')
    done_file = os.path.join(work_dir, '.vocab_' + vocab_name + '.txt.done')
    if os.path.exists(done_file):
        LogMessage("Skip generating vocab")
    else:
        LogMessage("Generating vocab with wordlist[{0}]...".format(args.wordlist))
        command = "wordlist_to_vocab.py {0} > {1}".format(args.wordlist, vocab)
        log_file = os.path.join(log_dir, 'wordlist_to_vocab.log')
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

# preparing int data
int_dir = os.path.join(work_dir, 'int_' + vocab_name)
done_file = os.path.join(int_dir, '.done')
if os.path.exists(done_file):
    LogMessage("Skip preparing int data")
else:
    LogMessage("Preparing int data...")
    command = "prepare_int_data.py {0} {1} {2}".format(args.text_dir, vocab, int_dir)
    log_file = os.path.join(log_dir, 'prepare_int_data.log')
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# get ngram counts
lm_name = vocab_name + '_' + str(args.order)
log_dir = os.path.join(work_dir, 'log', lm_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
counts_dir = os.path.join(work_dir, 'counts_' + lm_name)
done_file = os.path.join(counts_dir, '.done')
if os.path.exists(done_file):
    LogMessage("Skip getting counts")
else:
    LogMessage("Getting ngram counts...")
    command = "get_counts.py --min-counts='{0}' --max-memory={1} {2} {3} {4}".format(
            args.min_counts, args.max_memory, int_dir, args.order, counts_dir)
    log_file = os.path.join(log_dir, 'get_counts.log')
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# cleanup int dir
if args.cleanup == 'true' and args.keep_int_data == 'false':
    if os.system("cleanup_int_dir.py " + int_dir) != 0:
        sys.exit("train_lm.py: failed to cleanup int dir: " + int_dir)
    done_file = os.path.join(int_dir, '.done')
    os.remove(done_file)

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
    # subset counts dir
    subset_counts_dir = counts_dir + '_subset' + str(args.warm_start_ratio)
    done_file = os.path.join(subset_counts_dir, '.done')
    if os.path.exists(done_file):
        LogMessage("Skip subsetting counts dir")
    else:
        LogMessage("Subsetting counts dir...")
        command = "subset_count_dir.sh {0} {1} {2}".format(counts_dir, \
                args.warm_start_ratio, subset_counts_dir)
        log_file = os.path.join(log_dir, 'subset_count_dir.log')
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

    # warm-start optimize metaparameters
    subset_optimize_dir = os.path.join(work_dir, "optimize_{0}_subset{1}".format(lm_name, \
            args.warm_start_ratio))
    done_file = os.path.join(subset_optimize_dir, '.done')
    if os.path.exists(done_file):
        LogMessage("Skip warm-start optimizing metaparameters")
    else:
        LogMessage("Optimizing metaparameters for warm-start...")
        command = "optimize_metaparameters.py --cleanup={3} --progress-tolerance=1.0e-05 --num-splits={0} {1} {2}".format(
                args.num_splits, subset_counts_dir, subset_optimize_dir, args.cleanup)
        log_file = os.path.join(log_dir, 'optimize_metaparameters_warm_start.log')
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

    # cleanup subset counts dir
    if args.cleanup == 'true':
        if os.system("cleanup_count_dir.py " + subset_counts_dir) != 0:
            sys.exit("train_lm.py: failed to cleanup subset count dir: " + subset_counts_dir)
        done_file = os.path.join(subset_counts_dir, '.done')
        os.remove(done_file)

    # optimize metaparameters
    optimize_dir = os.path.join(work_dir, "optimize_{0}".format(lm_name))
    done_file = os.path.join(optimize_dir, '.done')
    if os.path.exists(done_file):
        LogMessage("Skip optimizing metaparameters")
    else:
        LogMessage("Optimizing metaparameters...")
        command = "optimize_metaparameters.py --warm-start-dir={0} \
                   --progress-tolerance=1.0e-03 --gradient-tolerance=0.01 \
                   --num-splits={1} {2} {3}".format(subset_optimize_dir,
                args.num_splits, counts_dir, optimize_dir)
        log_file = os.path.join(log_dir,'optimize_metaparameters.log')
        RunCommand(command, log_file, args.verbose == 'true')
        TouchFile(done_file)

    metaparam_file = os.path.join(optimize_dir, 'final.metaparams')
    metaparameters = ReadMetaparameters(metaparam_file)
    LogMessage("You can set --bypass-metaparameter-optimization='{0}' "
               "to get equivalent results".format(
                   FormatMetaparameters(metaparameters)))

# make lm dir
lm_dir = os.path.join(args.lm_dir, lm_name + '.pocolm')
done_file = os.path.join(lm_dir, '.done')
if os.path.exists(done_file):
    LogMessage("Skip making lm dir")
else:
    LogMessage("Making lm dir...")
    opts = []
    if args.num_splits > 1:
        opts.append('--keep-splits=true')
    if args.fold_dev_into != None:
        opts.append('--fold-dev-into=' + args.fold_dev_into)
    command = "make_lm_dir.py --cleanup={5} --num-splits={0} {1} {2} {3} {4}".format(
            args.num_splits, ' '.join(opts), counts_dir, metaparam_file, lm_dir, args.cleanup)
    log_file = os.path.join(log_dir, 'make_lm_dir.log')
    RunCommand(command, log_file, args.verbose == 'true')
    TouchFile(done_file)

# cleanup subset counts dir
if args.cleanup == 'true':
    if os.system("cleanup_count_dir.py " + counts_dir) != 0:
        sys.exit("train_lm.py: failed to cleanup count dir: " + counts_dir)
    done_file = os.path.join(counts_dir, '.done')
    os.remove(done_file)

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
