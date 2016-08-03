#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass


parser = argparse.ArgumentParser(description="Validates input directory containing text "
                                 "files from one or more data sources, including dev.txt.",
                                 epilog="E.g. validate_test_dir.py data/text",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("text_dir",
                    help="Directory in which to look for text data");

args = parser.parse_args()


if not os.path.exists(args.text_dir):
    sys.exit("validate_text_dir.py: Expected directory {0} to exist".format(args.text_dir))

if (not os.path.exists("{0}/dev.txt".format(args.text_dir)) and
    not os.path.exists("{0}/dev.txt.gz".format(args.text_dir))):
    sys.exit("validate_text_dir.py: Expected file {0}/dev.txt (or {0}/dev.txt.gz) to exist".format(args.text_dir))


num_text_files = 0;

def SpotCheckTextFile(text_file):
    try:
        if text_file.endswith(".gz"):
            f = gzip.open(text_file, 'r')
        else:
            f = open(text_file, 'r')
    except Exception as e:
        sys.exit("validate_text_dir.py: Failed to open {0} for reading: {1}".format(
                text_file, str(e)))
    found_nonempty_line = False
    for x in range(1,10):
        line = f.readline().strip("\n");
        if line is None:
            break
        words = line.split()
        if len(words) != 0:
            found_nonempty_line = True
            if (words[0] == "<s>" or words[0] == "<S>" or
                words[-1] == "</s>" or words[-1] == "</S>"):
                sys.exit("validate_text_dir.py: Found suspicious line '{0}' in file {1} (BOS and "
                         "EOS symbols are disallowed!)".format(line, text_file));
    if not found_nonempty_line:
        sys.exit("validate_text_dir.py: Input file {0} doesn't look right.".format(text_file));
    # close and open again.  Next we're going to check that it's not the case
    # that the first and second fields have disjoint words on them, and the
    # first field is always unique, which would be the case if the lines started
    # with some kind of utterance-id
    f.close();
    if text_file.endswith(".gz"):
        f = gzip.open(text_file, 'r')
    else:
        f = open(text_file, 'r')
    first_field_set = set()
    other_fields_set = set()
    for line in f:
        array = line.split()
        if len(array) > 0:
            first_word = array[0]
            if first_word in first_field_set or first_word in other_fields_set:
                # the first field isn't always unique, or is shared with other
                # fields.
                return;
            first_field_set.add(first_word)
        for i in range(1, len(array)):
            other_word = array[i]
            if other_word in first_field_set:
                # the first field has a value shared by some word not in the
                # first position.
                return;
            other_fields_set.add(other_word)
    print("validate_text_dir.py: input file {0} looks suspicious; check that you "
          "don't have utterance-ids in the first field (i.e. you shouldn't provide "
          "lines that look like 'utterance-id1 hello there').  Ignore this warning "
          "if you don't have that problem.".format(text_file), file=sys.stderr);


for f in os.listdir(args.text_dir):
    full_path = args.text_dir + "/" + f
    if os.path.isdir(full_path):
        continue
    if f.endswith(".txt") or f.endswith(".txt.gz"):
        if f.endswith(".txt") and os.path.isfile(full_path + ".gz"):
            sys.exit("validate_text_dir.py: both {0} and {0}.gz exist.".format(full_path))
        if not os.path.isfile(full_path):
            sys.exit("validate_text_dir.py: Expected {0} to be a file.".format(full_path))
        SpotCheckTextFile(full_path)
        num_text_files += 1
    elif f != "unigram_weights":
        sys.exit("validate_text_dir.py: Text directory should not contain files with suffixes "
                 "other than .txt (and not called 'unigram_weights'): " + f);

if num_text_files < 2:
    sys.exit("validate_text_dir.py: Directory {0} should contain at least one .txt file "
              "other than dev.txt.".format(args.text_dir));
