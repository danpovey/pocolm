#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings
from collections import defaultdict
try:              # since gzip will only be needed if there are gzipped files, accept
    import gzip   # failure to import it.
except:
    pass

parser = argparse.ArgumentParser(description="Extracts word counts from a data directory "
                                 "and creates a count directory with similar structure. "
                                 "Input directory has *.txt, counts directory has *.counts. "
                                 "Format of counts files is 'count word', e.g. '124 hello' ",
                                 epilog="See egs/swbd/run.sh for example.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("text_dir",
                    help="Directory in which to look for input text data\n");
parser.add_argument("count_dir",
                    help="Directory, to be written to, for counts files\n");


args = parser.parse_args()

os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])))

if os.system("validate_text_dir.py " + args.text_dir) != 0:
    sys.exit(1)


if not os.path.exists(args.count_dir):
    os.mkdir(args.count_dir)

def ProcessFile(text_file, counts_file):
    try:
        if text_file.endswith(".gz"):
            f = gzip.open(text_file, 'r')
        else:
            f = open(text_file, 'r')
    except Exception as e:
        sys.exit("Failed to open {0} for reading: {1}".format(
                text_file, str(e)))
    word_to_count = defaultdict(int)
    for line in f:
        for word in line.split():
            word_to_count[word] += 1
    f.close()
    try:
        cf = open(counts_file, "w");
    except:
        sys.exit("Failed to open {0} for writing".format(text_file))
    for word, count in word_to_count.items():
        print("{0} {1}".format(count, word), file=cf);
    cf.close()

num_files_processed = 0;

for f in os.listdir(args.text_dir):
    num_files_processed += 1
    text_path = args.text_dir + os.sep + f
    if os.path.isdir(text_path):
        continue
    if f.endswith(".txt"):
        counts_path = args.count_dir + os.sep + f[:-4] + ".counts"
        ProcessFile(text_path, counts_path)
    elif f.endswith(".txt.gz"):
        counts_path = args.count_dir + os.sep + f[:-7] + ".counts"
        ProcessFile(text_path, counts_path)
    elif f != "unigram_weights":
        sys.exit("get_word_counts.py: did not expect to find file {0}/{1} in "
                 "text directory".format(args.text_dir, f))

num_files_in_dest = 0;
for f in os.listdir(args.count_dir):
    if f.endswith(".counts"):
        num_files_in_dest += 1

if num_files_in_dest > num_files_processed:
    sys.exit("get_word_counts.py: It looks like your destination directory " +
             args.count_dir + " contains some extra counts files. "
             "Please clean up.");

print("Created {0} .counts files in {1}".format(num_files_processed,
                                                args.count_dir),
      file=sys.stderr);

