#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, tempfile, threading
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script evaluates the probability of some "
                                 "data (in text or gzipped-text format), given a language model "
                                 "in a pocolm 'lm-dir' (as validated by validate_lm_dir.py). "
                                 "The perplexity is printed to the standard output.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--max-memory", type=str, default='',
                    help="Memory limitation for sort.")
parser.add_argument("text_in", type=str,
                    help="Filename of input data (one sentence per line, no BOS or "
                    "EOS symbols; text or gzipped text")
parser.add_argument("lm_dir_in",
                    help="Source directory, for the input language model.")


args = parser.parse_args()

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");
# this will affect the program "sort" that we call.
os.environ['LC_ALL'] = 'C'


if os.system("validate_lm_dir.py " + args.lm_dir_in) != 0:
    sys.exit("get_data_prob.py: failed to validate input LM-dir")

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
                    sys.exit("get_data_prob.py: --max-memory should be formatted as "
                             "'a positive integer' or 'a positive integer appended "
                             "with 'b', 'K', 'M','G', or '%''.")
            # max memory size must be larger than zero
            if int(s[:-1]) == 0:
                sys.exit("get_data_prob.py: --max-memory must be > 0 {unit}.".format(
                         unit = s[-1]))    
        else:
            sys.exit("get_data_prob.py: the format of string --max-memory is not correct.")
    else:
         sys.exit("get_data_prob.py: the lenght of string --max-memory must >= 2.")

num_splits = None

if os.path.exists(args.lm_dir_in + "/num_splits"):
    f = open(args.lm_dir_in + "/num_splits")
    num_splits = int(f.readline())
    f.close()

if not os.path.exists(args.text_in):
    sys.exit("get_data_prob.py: input text data {0} does not exist".format(args.text_in))

def GetNgramOrder(lm_dir):
    f = open(lm_dir + "/ngram_order");
    return int(f.readline())

def RunCommand(command):
    # print the command for logging
    print(command, file=sys.stderr)
    if os.system(command) != 0:
        sys.exit("get_data_prob.py: error running command: " + command)

work_dir = tempfile.mkdtemp(dir = args.lm_dir_in)

# this temporary directory will be used by "sort".
os.environ['TMPDIR'] = work_dir

ngram_order = GetNgramOrder(args.lm_dir_in)

# set the memory restriction for "sort"
sort_mem_opt = ''
if args.max_memory != '':
  sort_mem_opt = ("--buffer-size={0} ".format(args.max_memory))

# create
if args.text_in[-3:] == '.gz':
    command = "gunzip -c {0} | text_to_int.py {1}/words.txt ".format(args.lm_dir_in,
                                                                     args.text_in)
else:
    command = "text_to_int.py {0}/words.txt <{1}".format(args.lm_dir_in,
                                                         args.text_in)
command += "| get-text-counts {0} | sort {1} | uniq -c | get-int-counts ".format(
            ngram_order, sort_mem_opt)
if num_splits == None:
    command += "{0}/int.dev".format(work_dir)
else:
    command += "/dev/stdout | split-int-counts " + ' '.join([ work_dir + "/int.dev." + str(n)
                                                              for n in range(1, num_splits + 1) ])

RunCommand(command)

tot_num_words = 0.0
tot_logprob = 0.0

def ComputeProbs(split_index):
    if split_index == None:
        command = "compute-probs {0}/float.all {1}/int.dev".format(
            args.lm_dir_in, work_dir)
    else:
        command = "compute-probs {0}/float.all.{2} {1}/int.dev.{2}".format(
            args.lm_dir_in, work_dir, split_index)
    print (command, file=sys.stderr)
    try:
        output = subprocess.check_output(command, shell = True)
    except:
        sys.exit("get_data_prob.py: error running command: " + command)
    [ num_words, tot_objf ] = output.split()
    global tot_num_words, tot_logprob
    tot_num_words += float(num_words)
    tot_logprob += float(tot_objf)

if num_splits == None:
    ComputeProbs(None)
else:
    threads = []
    for split_index in range(1, num_splits + 1):
        threads.append(threading.Thread(target = ComputeProbs,
                                        args = [split_index]))
        threads[-1].start()
    for t in threads:
        t.join()


logprob = tot_logprob / tot_num_words
perplexity = math.exp(-logprob)

print("get_data_prob.py: log-prob of {0} given model {1} was "
      "{2} per word [perplexity = {3}] over {4} words.".format(
        args.text_in, args.lm_dir_in, logprob, perplexity,
        tot_num_words), file=sys.stderr)

print(logprob, file=sys.stdout)

shutil.rmtree(work_dir)

