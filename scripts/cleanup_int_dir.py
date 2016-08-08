#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass


parser = argparse.ArgumentParser(description="Cleanup the largish files. "
                                 "This may be called when the ints no longer useful.",
                                 epilog="E.g. cleanup_int_dir.py data/lm/work/int_20000",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("int_dir",
                    help="Directory in which to find the data");

args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if os.system("validate_int_dir.py " + args.int_dir) != 0:
  sys.exit("command validate_int_dir.py {0} failed".format(args.int_dir))

f = open(os.path.join(args.int_dir, 'num_train_sets'))
line = f.readline()
num_train_sets = int(line)
f.close()

names = ['dev']
for n in range(1, num_train_sets + 1):
    names.append(str(n))

for name in names:
    filename = os.path.join(args.int_dir, name + '.txt.gz')
    if os.path.isfile(filename):
        os.remove(filename)
