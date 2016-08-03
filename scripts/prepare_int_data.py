#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, threading

# make sure scripts/internal is on the pythonpath.
sys.path = [ os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal" ] + sys.path

# for ExitProgram and RunCommand
from pocolm_common import *

parser = argparse.ArgumentParser(description="This program uses the vocabulary"
                                 "file to turn the data into ASCII-integer form,"
                                 "and to give it a standard format in preparation"
                                 " for language model training.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--fold-dev-into", type=str,
                    help="If supplied, the name of data-source into which to fold the "
                    "counts of the dev data for purposes of vocabulary estimation "
                    "(typically the same data source from which the dev data was "
                    "originally excerpted).")
parser.add_argument("--parallel", type=str, choices=["false", "true"], default="true",
                    help="Setting --parallel false will disable the (default) "
                    "parallel processing of multiple data sources by this script.")
parser.add_argument("text_dir",
                    help="Directory of original data")
parser.add_argument("vocab",
                    help="Vocabulary")
parser.add_argument("int_dir",
                    help="Directory of formed data")

args = parser.parse_args()

def GetNumTrainSets(int_dir):
  with open(int_dir) as f:
    for line in f:
      try:
        a = line.split()
        assert len(a) == 2
        ans = int(a[0])
      except:
        ExitProgram("failed to get the num_train_sets from {0}".format(int_dir))

  return ans

def GetNumWords(vocab):
  command = "tail -n 1 {0}".format(vocab)
  line = subprocess.check_output(command, shell = True)
  try:
    a = line.split()
    assert len(a) == 2
    ans = int(a[1])
  except:
    ExitProgram("failed to get the num_words from {0}".format(vocab))

  return ans

def GetNames(text_dir, int_dir):
  command = "get_names.py {0} > {1}/names".format(
              text_dir, int_dir)
  log_file = "{int_dir}/log/get_names.log".format(int_dir = int_dir)
  RunCommand(command, log_file)

def CopyFile(src, dest):
  try:
    shutil.copy(src, dest)
  except:
    ExitProgram("prepare_int_data.py: error copying {0} to {1}".format(src, dest))

# preparation of the train and dev data
def GetData(int, name):
  if os.path.exists(args.text_dir + "/" + name + ".txt.gz"):
    command = "set -o pipefail; gunzip -c {text_dir}/{name}.txt.gz | "\
              "text_to_int.py {vocab} | gzip -c > {int_dir}/{int}.txt.gz "\
               "2>{int_dir}/log/{int}.log".format(text_dir = args.text_dir, \
               name = name, vocab = args.vocab, int_dir = args.int_dir, int = int)
    log_file = "{int_dir}/log/{int}.log".format(int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command, log_file)
  else:
    command = "set -o pipefail; cat {text_dir}/{name}.txt | text_to_int.py {vocab} "\
               "| gzip -c > {int_dir}/{int}.txt.gz 2>{int_dir}/log/{int}.log".format(
               text_dir = args.text_dir, name = name, vocab = args.vocab, \
               int_dir = args.int_dir, int = int)
    log_file = "{int_dir}/log/{int}.log".format(int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command, log_file)

# make sure 'scripts', 'scripts/internal', and 'src' directory are on the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src" + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal")

if os.system("validate_text_dir.py " + args.text_dir) != 0:
  ExitProgram("command validate_text_dir.py {0} failed".format(args.text_dir))

if os.system("validate_vocab.py " + args.vocab) != 0:
  ExitProgram("command validate_vocab.py {0} failed".format(args.vocab))

if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0])) + "/text_to_int.py"):
  ExitProgram("prepare_int_data.py: expected text_to_int.py to be on the path")

# create the output data directory
if not os.path.exists(args.int_dir + "/log"):
  os.makedirs(args.int_dir + "/log")

# remove any old *.int.gz files in the output data directory
filelist = [f for f in os.listdir(args.int_dir) if f.endswith(".int.gz")]
for f in filelist:
  try:
    os.remove(f)
  except OSError:
    pass

GetNames(args.text_dir, args.int_dir)

# copy vocab to the output directory
CopyFile(args.vocab, args.int_dir + "/words.txt")

# get file 'num_train_sets' in int_dir from file 'names' in int_dir
with open(args.int_dir + os.sep + "num_train_sets", "w") as f:
  num_train_sets = GetNumTrainSets(args.int_dir + os.sep + "names")
  f.write(str(num_train_sets) + "\n")

# get file 'num_words' in int_dir from vocab
with open(args.int_dir + os.sep + "num_words", "w") as f:
  num_words = GetNumWords(args.vocab)
  f.write(str(num_words) + "\n")

# parallel/sequential processing
threads = []
with open(args.int_dir + "/names", "r") as f:
  for line in f:
    [int, name] = line.split()
    threads.append(threading.Thread(target = GetData, args = [int, name]))
threads.append(threading.Thread(target = GetData, args = ["dev", "dev"]))

if args.parallel == "true":
  for t in threads:
    t.start()
  for t in threads:
    t.join()
else:
  for t in threads:
    t.start()
    t.join()

# copy the unigram_weights to the output data directory
if os.path.exists(args.text_dir + "/unigram_weights"):
  CopyFile(args.text_dir + "/unigram_weights", args.int_dir)
else:
  try:
    os.remove(args.int_dir + "/unigram_weights")
  except OSError:
    pass

# validate the output data directory
if os.system("validate_int_dir.py " + args.int_dir) != 0:
  ExitProgram("command validate_int_dir.py {0} failed".format(args.int_dir))
