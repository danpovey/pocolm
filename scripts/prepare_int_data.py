#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil, threading

parser = argparse.ArgumentParser(description="This program uses the vocabulary" 
                                 "file to turn the data into ASCII-integer form," 
                                 "and to give it a standard format in preparation"
                                 " for language model training.")

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

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src" + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal") 

# do some checking
if os.system("validate_text_dir.py " + args.text_dir) != 0:
  sys.exit("prepare_int_data.py: failed to validate text directory")

if os.system("validate_vocab.py " + args.vocab) != 0:
  sys.exit("prepare_int_data.py: failed to validate vocab")

if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0])) + "/text_to_int.py"):
  sys.exit("prepare_int_data.py: expected text_to_int.py to be on the path")

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

# read the variables 'num_train_sets' and 'num_words' from the corresponding
# files in in_dir.
for name in [ 'num_train_sets', 'num_words' ]:
    f = open(args.int_dir + os.sep + name)
    globals()[name] = int(f.readline())
    f.close()

def ExitProgram(message):
  print(message, file=sys.stderr)
  os._exit(1)

def RunCommand(command):
  # print the command for logging
  print(command, file=sys.stderr)
  if os.system(command) != 0:
    ExitProgram("prepare_int_data.py: error running command: " + command)

def GetCommandStdout(command):
  # print the command for logging
  print(command, file=sys.stderr)
  try:
    output = subprocess.check_output(command, shell = True)
  except:
    ExitProgram("prepare_int_data.py: error running command: " + command)
  return output

def GetNames(text_dir, int_dir):
  command = "get_names.py {0} > {1}/names".format(
              text_dir, int_dir) 
  RunCommand(command)

def CopyFile(src, dest):
  try:
    shutil.copy(src, dest)
  except: 
    sys.exit("prepare_int_data.py: error copying {0} to {1}".format(src, dest))

# we can include the preparation of the dev data in the following
# by adding "dev dev" to the contents of int_dir/names
def PrepareData(int, name):
  if os.path.exists(args.text_dir + "/" + name + ".txt.gz"):
    command = "set -o pipefail; gunzip -c {text_dir}/{name}.txt.gz | \
               text_to_int.py {vocab} | gzip -c > {int_dir}/{int}.txt.gz \
               2>{int_dir}/log/{int}.log".format(text_dir = args.text_dir, \
               name = name, vocab = args.vocab, int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command)
  else:
    command = "set -o pipefail; cat {text_dir}/{name}.txt | text_to_int.py {vocab} \
               | gzip -c > {int_dir}/{int}.txt.gz 2>{int_dir}/log/{int}.log".format(
               text_dir = args.text_dir, name = name, vocab = args.vocab, \
               int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command)

# return the names of train and dev datasets 
GetNames(args.text_dir, args.int_dir)

# copy vocab to the output directory 
CopyFile(args.vocab, args.int_dir + "/words.txt")

# parallel/sequential processing
threads = [] 
f = open(args.int_dir + "/names", "r")
for line in f:
  [int, name] = line.split()
  threads.append(threading.Thread(target = PrepareData, args = [int, name]))
f.close()
threads.append(threading.Thread(target = PrepareData, args = ["dev", "dev"]))

if args.parallel=="true":
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
  sys.exit(1)
