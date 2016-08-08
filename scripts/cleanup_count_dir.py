#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass

parser = argparse.ArgumentParser(description="Cleanup the largish files. "
                                 "This may be called when the counts no longer useful.",
                                 epilog="E.g. cleanup_count_dir.py data/lm/work/counts_20000_3",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("count_dir",
                    help="Directory to cleanup")

args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

def CleanupDir(count_dir, ngram_order, num_train_sets):
    for n in [ 'dev' ] + range(1, num_train_sets + 1):
        for o in range(2, ngram_order +1):
            filename = os.path.join(count_dir, "int.{0}.{1}".format(n, o))
            if os.path.isfile(filename):
                os.remove(filename)
    filename = os.path.join(count_dir, "int.dev")
    if os.path.isfile(filename):
        os.remove(filename)

if os.system("validate_count_dir.py " + args.count_dir) != 0:
  sys.exit("command validate_count_dir.py {0} failed".format(args.count_dir))

f = open(os.path.join(args.count_dir, 'ngram_order'))
line = f.readline()
ngram_order = int(line)
f.close()

f = open(os.path.join(args.count_dir, 'num_train_sets'))
line = f.readline()
num_train_sets = int(line)
f.close()

# cleanup top-level dir
CleanupDir(args.count_dir, ngram_order, num_train_sets)

# find split-dir and cleanup
entities = os.listdir(args.count_dir)
for dirname in entities:
    if os.path.isdir(os.path.join(args.count_dir, dirname)) and dirname[0:5] == 'split':
        for n in range(1, int(dirname[5:]) + 1):
            count_dir = os.path.join(args.count_dir, dirname, str(n))
            if os.path.isdir(count_dir):
                CleanupDir(count_dir, ngram_order, num_train_sets)
        break
