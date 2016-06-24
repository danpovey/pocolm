#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, threading

parser = argparse.ArgumentParser(description="Usage: "
   "get_counts.py [options] <source-int-dir> <ngram-order> <dest-count-dir>"
   "e.g.:  get_counts.py data/int 3 data/counts_3"
   "This script computes data-counts of the specified n-gram order"
   "for each data-source in <source-int-dir>, and puts them all in"
   "<dest-counts-dir>.");

parser.add_argument("--parallel", type=str, default="false", choices=["true","false"],
                    help="Specify whether script runs in parallel default: false")
parser.add_argument("source_int_dir",
                    help="Specify <source_int_dir> the data-source")
parser.add_argument("ngram_order", type=int,
                    help="Specify the order of ngram")
parser.add_argument("dest_count_dir",
                    help="Specify <dest_count_dir> the destination to puts the counts")

args = parser.parse_args()

def ExitProgram(message):
  print(message, file=sys.stderr)
  os._exit(1)

def RunCommand(command):
  # print the command for logging
  print(command, file=sys.stderr)
  if os.system(command) != 0:
    ExitProgram("get_counts.py: error running command: " + command)

def GetCommandStdout(command):
  # print the command for logging
  print(command, file=sys.stderr)
  try:
      output = subprocess.check_output(command, shell = True)
  except:
      ExitProgram("get_objf_and_derivs_split.py: error running command: " + command)
  return output

def MakeDir(dest_count_dir):
  command = "mkdir -p " + dest_count_dir + "/log"
  RunCommand(command)

def GetNumTrainSets(source_int_dir):
  command = "cat " + source_int_dir + "/num_train_sets"
  return GetCommandStdout(command)

# copy over some meta-info into the 'counts' directory.
def CopyMetaInfo(source_int_dir, dest_count_dir):
  command = "for f in num_train_sets num_words names words.txt; do "\
            "  cp {0}/$f {1}/$f; "\
            "done".format(source_int_dir, dest_count_dir)
  RunCommand(command)

# save the n-gram order.
def SaveNgramOrder(dest_count_dir, ngram_order):
  command = "echo {0} > {1}/ngram_order".format(ngram_order, dest_count_dir)
  RunCommand(command)
# get-int-counts has an output for each order of count, but the maximum order
# is >1, and in this case the 1-gram counts are always zero (we always output
# the highest possible order for each word, which is normally $ngram_order,
# but can be as low as 2 for the 1st word of the sentence). So just put the
# output for order 1 in /dev/null.
def GetCounts(source_int_dir, dest_count_dir, ngram_order, n):
  if n == "dev":
    args = dest_count_dir+"/int.dev"
  else:
    args = "/dev/null"
    for o in range(2, ngram_order + 1):
      args += " {dest_count_dir}/int.{n}.{o}"
      args = args.format(dest_count_dir=dest_count_dir,n=n,o=o)

  command = "set -o pipefail;"\
          "( gunzip -c {source_int_dir}/{n}.txt.gz | "\
          "get-text-counts {ngram_order} | sort | uniq -c | "\
          "get-int-counts {args} )  2>>{dest_count_dir}/log/get_counts.{n}.log"
  command = command.format(source_int_dir=source_int_dir, n=n, ngram_order=ngram_order, \
                           dest_count_dir=dest_count_dir, args=args)
  RunCommand(command)

# we also want the files $dir/int.dev.{2,3,...}, i.e. the dev data split up by
# n-gram order, because this will be required if the user specifies to fold the
# dev data into some other set for the final build.
def SplitDevData(dest_count_dir, ngram_order):
  command = "split-int-counts-by-order /dev/null $(for o in $(seq 2 {ngram_order}); "\
        " do echo -n {dest_count_dir}/int.dev.$o ''; done) <{dest_count_dir}/int.dev "\
        "2>{dest_count_dir}/log/split_int_counts.log"
  command = command.format(ngram_order=ngram_order, dest_count_dir=dest_count_dir)
  RunCommand(command)

# make sure 'scripts' and 'src' directory are on the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");


if os.system("validate_int_dir.py " + args.source_int_dir) != 0:
  ExitProgram("validate_int_dir.py: fail")

if args.ngram_order < 2:
  ExitProgram(args.ngram_order + ": ngram-order must be at least 2 (if you want a unigram LM, do it by hand")

MakeDir(args.dest_count_dir)

num_train_sets = int(GetNumTrainSets(args.source_int_dir) )

CopyMetaInfo(args.source_int_dir, args.dest_count_dir)

SaveNgramOrder(args.dest_count_dir, args.ngram_order)


threads = []


for n in range(1, num_train_sets+1):
  threads.append(threading.Thread(target = GetCounts,
                  args = [args.source_int_dir,args.dest_count_dir,args.ngram_order,n] ))
threads.append(threading.Thread(target = GetCounts,
                args = [args.source_int_dir,args.dest_count_dir,args.ngram_order,"dev"] ))

if args.parallel == "false":
  for t in threads:
    t.start()
    t.join()
else:
  for t in threads:
    t.start()
  for t in threads:
    t.join()

SplitDevData(args.dest_count_dir, args.ngram_order)

