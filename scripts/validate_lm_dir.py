#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass

parser = argparse.ArgumentParser(description="Validates directory containing pocolm-format "
                                 "language model, as produced by make_lm_dir.py")

parser.add_argument("lm_dir",
                    help="Directory to validate")

args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if not os.path.exists(args.lm_dir):
    sys.exit("validate_lm_dir.py: Expected directory {0} to exist".format(args.lm_dir))

# the following code checks ngram_order
f = open("{0}/ngram_order".format(args.lm_dir))
line = f.readline()
try:
    ngram_order = int(line)
    assert ngram_order > 1 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_lm_dir.py: Expected file {0}/ngram_order to contain "
             "an integer >1: {1}".format(args.lm_dir, str(e)))
f.close()

# call validate_vocab.py to check the vocab.
if os.system("validate_vocab.py {0}/words.txt".format(args.lm_dir)) != 0:
    sys.exit("validate_lm_dir.py: failed to validate {0}/words.txt".format(args.lm_dir))


if os.system("echo true | cmp -s - {0}/was_pruned || "
            "echo false | cmp -s - {0}/was_pruned".format(args.lm_dir)) != 0:
    sys.exit("validate_lm_dir.py: {0}/was_pruned should contain "
             "'true' or 'false'.".format(args.lm_dir))


# check the 'names' file; it should have lines like:
#  1 switchboard
#  2 fisher
# etc.
f = open("{0}/names".format(args.lm_dir))
num_train_sets = 0
while True:
    line = f.readline()
    if line == '':
        break
    num_train_sets += 1
    try:
        [ m, name ] = line.split()
        assert int(m) == num_train_sets
    except:
        sys.exit("validate_lm_dir.py: bad {0}'th line of {1}/names: '{2}'".format(
                num_train_sets, args.lm_dir, line[0:-1]))
f.close()

count_file = args.lm_dir + "/float.all"
if not os.path.exists(count_file):
    sys.exit("validate_lm_dir.py: Expected file {0} to exist".format(count_file))
if not os.path.getsize(count_file) > 0:
    sys.exit("validate_lm_dir.py: Expected file {0} to be nonempty".format(count_file))

if os.system("validate_metaparameters.py --ngram-order={0} --num-train-sets={1} "
             "{2}/metaparameters".format(ngram_order,
                                         num_train_sets, args.lm_dir)) != 0:
    sys.exit("validate_lm_dir.py: failed to validate {0}/metaparameters".format(
            args.lm_dir))

print("validate_lm_dir.py: validated LM directory " + args.lm_dir,
      file=sys.stderr)
