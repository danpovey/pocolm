#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, shutil
from collections import defaultdict

parser = argparse.ArgumentParser(description="This script, given counts and metaparameters, will "
                                 "estimate an LM  in the 'pocolm-internal' format.  This consists of "
                                 "a directory with a particular structure, containing the files: "
                                 "float.all (which contains most of the parameters), words.txt, "
                                 "ngram_order, names, metaparameters and was_pruned. "
                                 "If you specify --keep-splits=true, the file num_splits will "
                                 "be created (containing the number of split pieces of the counts), "
                                 "and float.all won't exist, but float.all.split1, float.all.split2 "
                                 "etc., will be created.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--fold-dev-into', type=str,
                    help='If supplied, the name of data-source into which to fold the '
                    'counts of the dev data when building the model (typically the '
                    'same data source from which the dev data was originally excerpted).')
parser.add_argument("--num-splits", type=int, default=1,
                    help="Controls the number of parallel processes used to "
                    "get objective functions and derivatives.  If >1, then "
                    "we split the counts and build the LM in parallel.")
parser.add_argument("--keep-splits", type=str, choices=['true','false'],
                    default='false',
                    help="If true, instead of creating float.all, we'll create "
                    "float.all.1, float.2 and so on, split by history-state according "
                    "to the most recent history-word (the unigram state is repeated), "
                    "and the file num_splits containing the --num-splits argument, which "
                    "must be >1.")
parser.add_argument("count_dir",
                    help="Directory from which to obtain counts files\n")
parser.add_argument("metaparameters",
                    help="Filename from which to read metaparameters")
parser.add_argument("lm_dir",
                    help="Output directory where the language model is created.")


args = parser.parse_args()


# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src");

work_dir = args.lm_dir + "/work"

if not os.path.exists(work_dir):
    os.makedirs(work_dir)

if os.system("validate_count_dir.py " + args.count_dir) != 0:
    sys.exit("make_lm_dir.py: failed to validate counts directory")

# read the variables 'ngram_order', 'num_train_sets' and 'num_words'
# from the corresponding files in count_dir.
for name in [ 'ngram_order', 'num_train_sets', 'num_words' ]:
    f = open(args.count_dir + os.sep + name)
    globals()[name] = int(f.readline())
    f.close()


if args.keep_splits == 'true':
    if args.num_splits <= 1:
        sys.exit("make_lm_dir.py: --num-splits must be >1 if "
                 "you set --keep-splits=true")
else:
    if os.path.exists(args.lm_dir + "/num_splits"):
        os.remove(args.lm_dir + "/num_splits")
    if os.path.exists(args.lm_dir + "/float.all"):
        os.remove(args.lm_dir + "/float.all")


if args.num_splits < 1:
    sys.exit("make_lm_dir.py: --num-splits must be >0.")
if args.num_splits > 1:
    if (os.system("split_count_dir.sh {0} {1}".format(
                args.count_dir, args.num_splits))) != 0:
        sys.exit("make_lm_dir.py: failed to create split count-dir.")

fold_dev_opt=''

if args.fold_dev_into != None:
    fold_dev_into_int = None
    f = open(args.count_dir + "/names")
    for line in f.readlines():
        # we already validated the count-dir so we can assume the names file is
        # correctly formatted.
        [number, name] = line.split()
        if name == args.fold_dev_into:
            fold_dev_into_int = int(number)
    if fold_dev_into_int is None:
        sys.exit("make_lm_dir.py: invalid option --fold-dev-into={0}, does not "
                 "correspond to any entry in {1}/names".format(args.fold_dev_into,
                                                               args.count_dir))
    fold_dev_opt='--fold-dev-into-int=' + str(fold_dev_into_int)

if os.system("validate_metaparameters.py --ngram-order={ngram_order} "
             "--num-train-sets={num_train_sets} {metaparameters}".format(
        ngram_order = ngram_order, num_train_sets = num_train_sets,
        metaparameters = args.metaparameters)) != 0:
    sys.exit("make_lm_dir.py: failed to validate metaparameters "
             + args.metaparameters)

for name in ['words.txt', 'ngram_order', 'names' ]:
    src = args.count_dir + os.sep + name
    dest = args.lm_dir + os.sep + name
    try:
        shutil.copy(src, dest)
    except:
        sys.exit("make_lm_dir.py: error copying {0} to {1}".format(src, dest))

try:
    shutil.copy(args.metaparameters,
                args.lm_dir + os.sep + "metaparameters")
except:
    sys.exit("make_lm_dir.py: error copying {0} to {1}".format(
            args.metaparameters,
            args.lm_dir + os.sep + "metaparameters"))

f = open(args.lm_dir + "/was_pruned", "w")
print("false", file=f)
f.close()

if args.num_splits == 1:
    command = ("get_objf_and_derivs.py {fold_dev_opt} {count_dir} {metaparameters} "
               "{work_dir}/objf {work_dir} 2>{work_dir}/log.txt".format(fold_dev_opt = fold_dev_opt,
                                                        count_dir = args.count_dir,
                                                        metaparameters = args.metaparameters,
                                                        work_dir = work_dir))
else:
    need_model_opt = '--need-model=true' if args.keep_splits == 'false' else ''
    command = ("get_objf_and_derivs_split.py --num-splits={num_splits} {need_model_opt} "
               "{fold_dev_opt} {count_dir} {metaparameters} {work_dir}/objf "
               "{work_dir} 2>{work_dir}/log.txt".format(
            need_model_opt = need_model_opt, fold_dev_opt = fold_dev_opt,
            num_splits = args.num_splits, count_dir = args.count_dir,
            metaparameters = args.metaparameters, work_dir = work_dir))

print("make_lm_dir.py: running command {0}".format(command), file=sys.stderr)

if os.system(command) != 0:
    sys.exit("make_lm_dir.py: error running command {0}".format(command))

src = work_dir + os.sep + "num_ngrams"
dst = args.lm_dir + os.sep + "num_ngrams"
try:
    shutil.copy(src, dst)
except:
    sys.exit("make_lm_dir.py: error copying {0} to {1}".format(src, dst))


if args.keep_splits == 'true':
    f = open(args.lm_dir + '/num_splits', 'w')
    print(str(args.num_splits), file=f)
    f.close()
    for i in range(1, args.num_splits + 1):
        src_file = "{0}/split{1}/{2}/float.all".format(work_dir, args.num_splits, i)
        dest_file = "{0}/float.all.{1}".format(args.lm_dir, i)
        try:
            shutil.move(src_file, dest_file)
        except:
            sys.exit("make_lm_dir.py: error moving {0} to {1}".format(src_file,
                                                                      dest_file))
else:
    try:
        shutil.move(work_dir + "/float.all",
                    args.lm_dir + "/float.all")
    except:
        sys.exit("make_lm_dir.py: error moving {0}/float.all to {1}/float.all".format(
                work_dir, args.lm_dir))



if os.system("validate_lm_dir.py " + args.lm_dir) != 0:
    sys.exit("make_lm_dir.py: error validating lm-dir " + args.lm_dir)
