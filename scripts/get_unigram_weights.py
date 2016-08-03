#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings
from collections import defaultdict

parser = argparse.ArgumentParser(description="Given a directory containing word counts "
                                 "as created by get_counts.py, this program obtains "
                                 "weighting factors for each of the non-dev counts files, "
                                 "based on maximizing a unigram probability of "
                                 "the dev data's counts.  It writes to the standard output "
                                 "the weights of the form '<basename> <weight>', one weight "
                                 "per line e.g. 'switchboard 0.23'.",
                                 epilog="See egs/swbd/run.sh for example.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)


parser.add_argument("--verbose", type=str,
                    help="If true, print more verbose output",
                    default="false", choices = ["false", "true"])
parser.add_argument("count_dir",
                    help="Directory from which to obtain counts files\n");


args = parser.parse_args()


# this reads the counts file and returns a dictionary from word to
# count.
def ReadCountsFile(counts_file):
    try:
        f = open(counts_file, "r");
    except:
        sys.exit("Failed to open {0} for reading".format(counts_file))
    word_to_count = defaultdict(int)
    for line in f:
        for word in line.split():
            word_to_count[word] += 1
    f.close()
    return word_to_count

train_counts = { }

num_files_in_dest = 0;
for f in os.listdir(args.count_dir):
    full_path = args.count_dir + os.sep + f
    if f.endswith(".counts"):
        if f == "dev.counts":
            dev_counts = ReadCountsFile(full_path)
        else:
            train_counts[f[0:-7]] = ReadCountsFile(full_path)

train_keys = train_counts.keys()
num_train_files = len(train_keys)

assert num_train_files > 0

if num_train_files == 1:
    # don't bother reading anything in: if there is only
    # one source of data, the weight won't matter, and we'll
    # just write one.
    if args.verbose == "true":
        print("get_unigram_weights.py: only one data source so not "
              "really estimating weights.", file=sys.stderr)
    print(train_keys[0], 1.0)
    sys.exit(0)


# for efficiency, change the format...
# we'll make it a matrix, with the following format:
# for each word that appears both in the dev counts and at least
# one train count, the matrix has a row of the form
# [ dev-count train-count1 train-count2 ]
# To avoid a numpy dependency we just make it a list of lists.
tot_counts = [ 0 ] * num_train_files
for i in range(num_train_files):
    tot_counts[i] = sum(train_counts[train_keys[i]].values()) * 1.0

all_counts = []
for word, count in dev_counts.items():
    this_row = [ 0 ] * (num_train_files + 1)
    found_train_count = False
    this_row[0] = count
    for i in range(num_train_files):
        if word in train_counts[train_keys[i]]:
            this_row[i+1] = train_counts[train_keys[i]][word] / tot_counts[i]
            found_train_count = True
    if found_train_count:
        all_counts.append(this_row)

if len(all_counts) == 0:
    sys.exit("can't get unigram weights because dev and train data have "
             "no overlap in words")

# print("All_counts [normalized] = " + str(all_counts))

current_weights = [ 1.0 / num_train_files ] * num_train_files

threshold = 1.0e-03
iter = 0
while True:
    # this is an E-M procedure to re-estimate the weights.
    next_weights = [ 0.0 ] * num_train_files
    tot_logprob = 0.0
    tot_count = 0
    num_rows = len(all_counts)
    for i in range(num_rows):
        this_row = all_counts[i]
        this_count = this_row[0]  # the dev-data count.
        tot_count += this_count
        this_prob = 0.0
        for j in range(num_train_files):
            this_prob += current_weights[j] * this_row[j+1]
        tot_logprob += math.log(this_prob) * this_count
        for j in range(num_train_files):
            next_weights[j] += (this_count * current_weights[j] *
                                this_row[j+1] / this_prob)

    if args.verbose == "true":
        print("Average log-prob per word on iteration {0} is {1} over {2} observations".format(
                iter, tot_logprob / tot_count, tot_count), file=sys.stderr)
    tot_diff = 0.0
    for j in range(num_train_files):
        next_weights[j] /= tot_count
        tot_diff += (next_weights[j] - current_weights[j]) ** 2
    if args.verbose == "true":
        print("Weights on iteration {0} are {1}".format(iter, str(next_weights)),
              file=sys.stderr)
    current_weights = next_weights
    if math.sqrt(tot_diff) < threshold:
        break;
    iter += 1


# Now we renormalize the weights so that instead of weighting the unigram
# probabilities, they weight the actual counts.  If the datasets have different
# total numbers of words, the weights will be different.
for i in range(num_train_files):
    current_weights[i] *= tot_counts[i]
# the scalar constant actually makes no difference to any valid use of these
# weights, and we set the largest weight to 1 by dividing by the max instead of
# by the total, partly in order to point out that these aren't the kind of
# weights that inherently sum to one.
m = max(current_weights)
for i in range(num_train_files):
    current_weights[i] /= m;

if args.verbose == "true":
    print("get_unigram_weights.py: Final weights after renormalizing so they "
          "can be applied to the raw counts, are: " + str(next_weights),
          file=sys.stderr)


for i in range(num_train_files):
    print(train_keys[i], current_weights[i])

