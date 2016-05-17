#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass


parser = argparse.ArgumentParser(description="Validates directory containing integerized "
                                 "text data, as produced by prepare_int_data.sh",
                                 epilog="E.g. validate_int_dir.py data/int.100k");

parser.add_argument("int_dir",
                    help="Directory in which to find the data");

args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if not os.path.exists(args.int_dir):
    sys.exit("validate_int_dir.py: Expected directory {0} to exist".format(args.int_dir))

if not os.path.exists("{0}/dev.txt.gz".format(args.int_dir)):
    sys.exit("validate_int_dir.py: Expected file {0}/dev.txt.gz to exist".format(args.int_dir))

if not os.path.exists("{0}/num_train_sets".format(args.int_dir)):
    sys.exit("validate_int_dir.py: Expected file {0}/num_train_sets to exist".format(args.int_dir))

# the following code checks num_train_sets and sets num_train_sets
# to the appropriate variable.
f = open("{0}/num_train_sets".format(args.int_dir))
line = f.readline()
try:
    num_train_sets = int(line)
    assert num_train_sets > 0 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_int_dir.py: Expected file {0}/num_train_sets to contain "
             "an integer >0: {1}".format(args.int_dir, str(e)))
f.close()

# the following code checks num_words.
f = open("{0}/num_words".format(args.int_dir))
line = f.readline()
try:
    num_words = int(line)
    assert num_words > 0 and len(line.split()) == 1
    assert f.readline() == ''
except Exception as e:
    sys.exit("validate_int_dir.py: Expected file {0}/num_words to contain "
             "an integer >0: {1}".format(args.int_dir, str(e)))
f.close()


# call validate_vocab.py to check the vocab.
if os.system("validate_vocab.py --num-words={0} {1}/words.txt".format(
        num_words, args.int_dir)) != 0:
    sys.exit(1)

num_words = subprocess.check_output("cat {0}/words.txt | wc -l".format(args.int_dir), shell=True)
try:
    num_words = int(num_words) + 1
except:
    sys.exit("validate_int_dir.py: error getting number of words from {0}/words.txt".format(
            args.int_dir))


# check the 'names' file; it should have lines like:
#  1 switchboard
#  2 fisher
# etc.
f = open("{0}/names".format(args.int_dir))
for n in range(1, num_train_sets + 1):
    line = f.readline()
    try:
        [ m, name ] = line.split()
        assert int(m) == n
    except:
        sys.exit("validate_int_dir.py: bad {0}'th line of {1}/names: '{2}'".format(
                n, args.int_dir, line[0:-1]))
f.close()

names = ['dev']
for n in range(1, num_train_sets + 1):
    names.append(str(n))

for name in names:
    p = subprocess.Popen("gunzip -c {0}/{1}.txt.gz".format(args.int_dir, name),
                         stdout=subprocess.PIPE, shell=True)
    num_ints = 0
    for l in range(10):
        line = p.stdout.readline()
        if line == None:
            break
        try:
            ints = [ int(x) for x in line.split() ]
            num_ints += len(ints)
            for i in ints:
                if i < 3 or i > num_words:
                    sys.exit("validate_int_dir.py: value {0} out of range in file {1}/{2}.txt.gz".format(
                            i, args.int_dir, name))
        except:
            sys.exit("validate_int_dir.py: bad line {0} in file {1}/{2}.txt.gz".format(
                    line.strip('\n'), args.int_dir, name))
    if num_ints == 0:
        # in theory it's possible that a file whose first 10 lines is empty
        # could be valid, a there is nothing wrong in principle with modeling
        # empty sequences.  But it's very odd.
        sys.exit("validate_int_dir.py: did not see any data in file {1}/{2}.txt.gz".format(
                args.int_dir, name))
    p.terminate()
