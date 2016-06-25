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
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

# do some checking
if os.system("validate_text_dir.py " + args.text_dir) != 0:
  sys.exit("prepare_int_data.py: failed to validate text directory")

if os.system("validate_vocab.py " + args.vocab) != 0:
  sys.exit("prepare_int_data.py: failed to validate vocab")

if not os.path.exists(os.path.abspath(os.path.dirname(sys.argv[0])) + "/text_to_int.py"):
  sys.exit("prepare_int_data.py: expected text_to_int.py to be on the path")

# create the output data directory
if not os.path.exists(args.int_dir):
  os.makedirs(args.int_dir)

if not os.path.exists(args.int_dir + "/log"):
  os.makedirs(args.int_dir + "/log")

# remove any old *.int.gz files in the output data directory
filelist = [f for f in os.listdir(args.int_dir) if f.endswith(".int.gz")]
for f in filelist:
  try:
    os.remove(f)
  except OSError:
    pass

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

def GetNames(script_dir, text_dir, int_dir):
  command = "{0}/internal/get_names.py {1} > {2}/names".format(
              script_dir, text_dir, int_dir) 
  RunCommand(command)

def GetNumTrainSets(int_dir):
  command = "cat {0}/names | wc -l".format(int_dir)
  line = subprocess.check_output(command, shell = True) 
  try:
    a = line.split()
    assert len(a) == 1
    ans = int(a[0])
  except:
    sys.exit("prepare_int_data.py: error: unexpected output '{0}' from command {1}".format(
                line, command))
  return ans

def CopyFile(src, dest):
  try:
    shutil.copyfile(src, dest)
  except: 
    sys.exit("prepare_int_data.py: error copying {0} to {1}".format(src, dest))

# returns num-words in vocab.
def GetNumWords(vocab):
  command = "tail -n 1 {0}".format(vocab)
  line = subprocess.check_output(command, shell = True)
  try:
    a = line.split()
    assert len(a) == 2
    ans = int(a[1])
  except:
    sys.exit("prepare_int_data.py: error: unexpected output '{0}' from command {1}".format(
                line, command))
  return ans

# we can include the preparation of the dev data in the following
# by adding "dev dev" to the contents of int_dir/names
def PrepareData(line):
  [int, name] = line.split()
  if os.path.exists(args.text_dir + "/" + name + ".txt.gz"):
    command = "set -o pipefail; gunzip -c {text_dir}/{name}.txt.gz | text_to_int.py {vocab} | gzip -c > {int_dir}/{int}.txt.gz 2>{int_dir}/log/{int}.log".format(
                  text_dir = args.text_dir, name = name, vocab = args.vocab, int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command)
  else:
    command = "set -o pipefail; cat {text_dir}/{name}.txt | text_to_int.py {vocab} | gzip -c > {int_dir}/{int}.txt.gz 2>{int_dir}/log/{int}.log".format(
                  text_dir = args.text_dir, name = name, vocab = args.vocab, int_dir = args.int_dir, int = int)
    output = GetCommandStdout(command)

# return the names of train and dev datasets 
scriptdir = os.path.abspath(os.path.dirname(sys.argv[0]))
GetNames(scriptdir, args.text_dir, args.int_dir)

# copy vocab to the output directory 
src = args.vocab 
dest = args.int_dir + "/words.txt"
CopyFile(src, dest)

# return the number of train sets 
with open(args.int_dir + "/num_train_sets", "w") as file:
  file.write(str(GetNumTrainSets(args.int_dir)))

# return num-words in vocab 
with open(args.int_dir + "/num_words", "w") as file:
  file.write(str(GetNumWords(args.vocab)))

src = args.int_dir + "/names"
dest = args.int_dir + "/names_modified"
file = open(dest, "w")
file.write("dev dev \n")
sourcefile = open(src, "r")
lines = sourcefile.readlines()
for line in lines:
  file.write(line)
file.close()

# parallel/sequential processing
threads = [] 
f = open(dest, "r")
for line in f:
  threads.append(threading.Thread(target = PrepareData, args = [line]))
f.close()
os.remove(dest)

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
src = args.text_dir + "/unigram_weights"
dest = args.int_dir
if os.path.exists(src):
  CopyFile(src, dest)
else:
  try:
    os.remove(dest + "/unigram_weights")
  except OSError:
    pass

# validate the output data directory  
if os.system("validate_int_dir.py " + args.int_dir) != 0:
  sys.exit(1)
