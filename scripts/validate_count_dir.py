#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass

parser = argparse.ArgumentParser(description="Validates directory containing binary "
                                 "counts, as produced by prepare_counts.sh",
                                 epilog="E.g. validate_count_dir.py data/counts.100k",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("count_dir",
                    help="Directory to validate")

args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if not os.path.exists(args.count_dir):
    sys.exit("validate_count_dir.py: Expected directory {0} to exist".format(args.count_dir))

if not os.path.exists("{0}/num_train_sets".format(args.count_dir)):
    sys.exit("validate_count_dir.py: Expected file {0}/num_train_sets to exist".format(args.count_dir))

# the following code checks num_train_sets and sets num_train_sets
# to the appropriate variable.
f = open("{0}/num_train_sets".format(args.count_dir))
line = f.readline()
try:
    num_train_sets = int(line)
    assert num_train_sets > 0 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_count_dir.py: Expected file {0}/num_train_sets to contain "
             "an integer >0: {1}".format(args.count_dir, str(e)))
f.close()

# the following code checks num_words.
f = open("{0}/num_words".format(args.count_dir))
line = f.readline()
try:
    num_words = int(line)
    assert num_words > 0 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_count_dir.py: Expected file {0}/num_words to contain "
             "an integer >0: {1}".format(args.count_dir, str(e)))
f.close()


# the following code checks split_modulus (which doesn't have to exist,
# it's optional).
if os.path.exists("{0}/split_modulus".format(args.count_dir)):
    f = open("{0}/num_words".format(args.count_dir))
    line = f.readline()
    try:
        split_modulus = int(line)
        assert split_modulus > 0 and len(line.split()) == 1
        assert f.readline() == ''
    except Exception as e:
        sys.exit("validate_count_dir.py: Expected file {0}/split_modulus (if it exists) "
                 "to contain an integer >0: {1}".format(args.count_dir, str(e)))
    f.close()



# the following code checks ngram_order
f = open("{0}/ngram_order".format(args.count_dir))
line = f.readline()
try:
    ngram_order = int(line)
    assert ngram_order > 1 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_count_dir.py: Expected file {0}/ngram_order to contain "
             "an integer >1: {1}".format(args.count_dir, str(e)))
f.close()


# call validate_vocab.py to check the vocab.
if os.system("validate_vocab.py --num-words={0} {1}/words.txt".format(
        num_words, args.count_dir)) != 0:
    sys.exit("validate_vocab.py: failed to validate {0}/words.txt".format(args.lm_dir))


names = set()
# check the 'names' file; it should have lines like:
#  1 switchboard
#  2 fisher
# etc.
f = open("{0}/names".format(args.count_dir))
for n in range(1, num_train_sets + 1):
    line = f.readline()
    try:
        [ m, name ] = line.split()
        if name in names:
            sys.exit("validate_count_dir.py: repeated name {0} in {1}/names".format(
                    name, args.count_dir))
        names.add(name)
        assert int(m) == n
    except:
        sys.exit("validate_count_dir.py: bad {0}'th line of {1}/names: '{2}'".format(
                n, args.count_dir, line[0:-1]))
f.close()

# validate the 'unigram_weights' file, if it exists.  the 'unigram_weights' file
# is an optional part of the directory format; we put it here so it can be used
# to initialize the metaparameters in a reasonable way.
if os.path.exists("{0}/unigram_weights".format(args.count_dir)):
    f = open("{0}/unigram_weights".format(args.count_dir))
    names_with_weights = set()
    while True:
        line = f.readline()
        if line == '':
            break
        try:
            [ name, weight ] = line.split()
            weight = float(weight)
            assert weight >= 0.0 and weight <= 1.0
            if not name in names:
                sys.exit("validate_count_dir.py: bad line '{0}' in file {1}/unigram_weights: "
                         "name {2} does not appear in {1}/names".format(
                        line[:-1], args.count_dir, name))
            if name in names_with_weights:
                sys.exit("validate_count_dir.py: bad line '{0}' in file {1}/unigram_weights: "
                         "name {2} appears twice".format(
                        line[:-1], args.count_dir, name))
            names_with_weights.add(name)
        except Exception as e:
            sys.exit("validate_count_dir.py: bad line '{0}' in file {1}/unigram_weights: {2}".format(
                    line[:-1], args.count_dir, str(e)))
    for name in names:
        if not name in names_with_weights:
            sys.exit("validate_count_dir.py: expected the name {0} to appear in "
                     "{1}/unigram_weights".format(name, args.count_dir))
    f.close()


for n in [ 'dev' ] + range(1, num_train_sets + 1):
    for o in range(2, ngram_order +1):
        filename = "{0}/int.{1}.{2}".format(args.count_dir, n, o)
        if not os.path.exists(filename):
            sys.exit("validate_count_dir.py: Expected file {0} to exist".format(filename))
        # commenting the check that the file is nonempty, since in split directories, we
        # do get empty files.
        #if not os.path.getsize(filename) > 0:
        #    sys.exit("validate_count_dir.py: Expected file {0} to be nonempty".format(filename))

if not os.path.exists("{0}/int.dev".format(args.count_dir)):
    sys.exit("validate_count_dir.py: Expected file {0}/int.dev "
             "to exist".format(args.count_dir))


print("validate_count_dir.py: validated counts directory " + args.count_dir,
      file=sys.stderr)
