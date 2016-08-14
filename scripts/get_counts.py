#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings, subprocess, threading, shutil
import tempfile
import platform


# make sure scripts/internal is on the pythonpath.
sys.path = [ os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal" ] + sys.path

# for ExitProgram and RunCommand
from pocolm_common import *


parser = argparse.ArgumentParser(description="Usage: "
                                 "get_counts.py [options] <source-int-dir> <ngram-order> <dest-count-dir>"
                                 "e.g.:  get_counts.py data/int 3 data/counts_3"
                                 "This script computes data-counts of the specified n-gram order"
                                 "for each data-source in <source-int-dir>, and puts them all in"
                                 "<dest-counts-dir>.",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--dump-counts-parallel", type=str, default='true',
                    choices=['true','false'],
                    help="If true, while obtaining the original counts, process multiple data-sources "
                    "in parallel (configurable because 'sort' may use a fair amount of memory).")
parser.add_argument("--verbose", type=str, default='false',
                    choices=['true','false'],
                    help="If true, print commands as we execute them.")
parser.add_argument("--cleanup", type=str, default='true',
                    choices=['true','false'],
                    help="If true, remove intermediate files (only relevant if --min-counts option "
                    "is supplied).")
parser.add_argument("--min-counts", type=str, default='',
                    help="This string allows you to specify minimum counts to be applied "
                    "to the stats.  You may in general specify min-counts per n-gram order "
                    "and per data-source, but they applied 'jointly' in a smart way so "
                    "that, for example, for some order if all data-sources have a min-count "
                    "of 2, an n-gram will be pruned from all data-sources if the total count "
                    "over all data-sources is 2.  Min-counts may be specified for order 3 "
                    "and above, in a comma-separated list, with values that must be "
                    "non-decreasing.  E.g. --min-counts=2,3.  In case of mismatch with "
                    "the actual n-gram order, excess min-counts will be truncated and "
                    "an deficit will be remedied by repeating the last min-count.  You "
                    "may specify different min-counts for different data-sources, e.g. "
                    "--min-counts='fisher=2,3 swbd1=1,1'.  You may also set min-counts for "
                    "some data-sources and use a default for others, as in "
                    "--min-counts='fisher=2,3 default=1,1'.  You may not set min-counts for "
                    "the dev set.");
parser.add_argument("--num-min-count-jobs", type=int, default=5,
                    help="The number of parallel jobs used for applying min-counts (only "
                    "relevant if --min-counts option is given")
parser.add_argument("--num-count-jobs", type=int, default=4,
                    help="The number of parallel processes per data source used for "
                    "getting initial counts")
parser.add_argument("--max-memory", type=str, default='',
                    help="Memory limitation for sort.")
parser.add_argument("source_int_dir",
                    help="Specify <source_int_dir> the data-source")
parser.add_argument("ngram_order", type=int,
                    help="Specify the order of ngram")
parser.add_argument("dest_count_dir",
                    help="Specify <dest_count_dir> the destination to puts the counts")

args = parser.parse_args()

# this reads the 'names' file (which has lines like "1 switchboard", "2 fisher"
# and so on), and returns a dictionary from integer id to name.
def ReadNames(names_file):
    try:
        f = open(names_file, "r");
    except:
        sys.exit("get_counts.py: failed to open --names={0}"
                 " for reading".format(names_file))
    number_to_name = { }
    for line in f:
        try:
            [ number, name ] = line.split();
            number = int(number)
        except:
            sys.exit("get_counts.py: Bad line '{0}' in names file {1}".format(
                    line[0:-1], names_file))
        if number in number_to_name:
            sys.exit("get_counts.py: duplicate number {0} in names file {1}".format(
                    number, names_file))
        number_to_name[number] = name
    f.close()
    return number_to_name

def GetNumTrainSets(source_int_dir):
    f = open(source_int_dir + '/num_train_sets')
    # the following should not fail, since we validated source_int_dir.
    num_train_sets = int(f.readline())
    assert f.readline() == ''
    f.close()
    return num_train_sets

# copy over some meta-info into the 'counts' directory.
def CopyMetaInfo(source_int_dir, dest_count_dir):
    for f in ['num_train_sets', 'num_words', 'names', 'words.txt' ]:
        try:
            src = source_int_dir + os.path.sep + f
            dest = dest_count_dir + os.path.sep + f
            shutil.copy(src, dest)
        except:
            ExitProgram('error copying {0} to {1}'.format(src, dest))

def IsCygwin():
    return platform.system()[0:3].lower() == 'win' or platform.system()[0:3].lower() == 'cyg'

# This function, called from FormatMinCounts, takes an array of
# min-counts like [2,3], and normalizes its length to ngram_order - 2
# by either removing elements from the end, or duplicating the last
# element.  If it makes any change, it prints a warning.
def NormalizeMinCountsLength(min_counts, ngram_order):
    if len(min_counts) == 0:
        # this point in the code should not be reached, actually.
        sys.exit("get_counts.py: invalid --min-counts string or code error.")
    ans = min_counts
    # Check that the min-counts are non-decreasing and are >= 1.
    for i in range(len(min_counts) - 1):
        if min_counts[i] < 1:
            sys.exit("get_counts.py: invalid --min-counts string, min-counts must "
                     "be >= 1.")
        if min_counts[i] > min_counts[i+1]:
            sys.exit("get_counts.py: invalid --min-counts string, min-counts must "
                     "not decrease from one n-gram order to the next.")
    if len(ans) < ngram_order - 2:
        while len(ans) < ngram_order - 2:
            ans.append(ans[-1]) # duplicate the last element
        print("get_counts.py: extending min-counts from {0} to {1} since "
              "ngram order is {2}".format(','.join([str(x) for x in min_counts]),
                                          ','.join([str(x) for x in ans]), ngram_order))
    if len(ans) > ngram_order - 2:
        ans = ans[0:ngram_order-2]
        print("get_counts.py: truncating min-counts from {0} to {1} since "
              "ngram order is {2}".format(','.join([str(x) for x in min_counts]),
                                          ','.join([str(x) for x in ans]), ngram_order))
    return ans


# This function converts from the format of --min-counts string accepted by this
# program to the format that is accepted by int-counts-enforce-min-counts; it
# returns a string (such as --min-counts="2,3" -> "2 3", which would be a valid
# string for a 4-gram setup and arbitrary number of inputs; or, supposing
# --min-counts="fisher=2,3 swbd=1,2" and the "names"  file maps "fisher" -> 1
# and "swbd" -> 2, this function would return the string "2,1 3,2".
# If the ngram-order is <3, this function will return the empty string, and
# in that case you shouldn't try to apply min-counts.
def FormatMinCounts(source_int_dir, num_train_sets, ngram_order, min_counts):
    if len(min_counts) == 0:
        sys.exit("get_counts.py: empty --min-counts string.")
    if ngram_order < 3:
        print("get_counts.py: ignoring --min-counts string since ngram "
              "order is {0} and min-counts are only supported for orders "
              "3 and greater.".format(ngram_order), file=sys.stderr)
        return ''

    pieces = min_counts.split()
    # 'pieces' is the whitespace-separated pieces of the string.
    if len(pieces) == 1 and len(pieces[0].split('=')) == 1:
        # the user has specified something like --min-counts=2,3, and we have
        # something like pieces = ['2,3'].  So there is no attempt to have
        # different min-counts for different data sources.
        try:
            min_counts_per_order = [ float(x) for x in pieces[0].split(',') ]
        except:
            sys.exit("get_counts.py: --min-counts={0} has unexpected format".format(
                    min_counts))
        min_counts_per_order = NormalizeMinCountsLength(min_counts_per_order,
                                                        ngram_order)
        ans = ' '.join([str(int(x)) if x == int(x) else str(x)
                        for x in min_counts_per_order])
    else:
        # we expect 'pieces' to be something like [ 'fisher=2,3' 'swbd=1,2' ].

        # we'll set up a dictionary from name to min-count array, something
        # like name_to_mincounts = [ 'fisher':[2,3], 'swbd':[1,2] ]
        name_to_mincounts = dict()
        for piece in pieces:
            try:
                [name,comma_separated_list] = piece.split('=')
                this_mincounts = [ float(x) for x in comma_separated_list.split(',') ]
                this_mincounts = NormalizeMinCountsLength(this_mincounts,
                                                          ngram_order)
            except:
                sys.exit("get_counts.py: could not parse --min-counts='{0}'.".format(
                        min_counts))
            if name in name_to_mincounts:
                sys.exit("get_counts.py: duplicate entry found in --min-counts='{0}'.".format(
                        min_counts))
            name_to_mincounts[name] = this_mincounts
        names_used = set() # the set of keys of 'name_to_mincounts' that have been used.
        # names is a map from integer to name, e.g.
        # names = [ 1:'fisher', 2:'swbd' ]
        names = ReadNames(source_int_dir + "/names")
        # min_counts_per_order will be an array (one per order from 2,...)
        # of arrays, one per training set, of the respective min-counts per
        # dataset, e.g. in our example it would be [ [ 2,1 ], [3,2] ]
        min_counts_per_order = [ ]
        for o in range(ngram_order - 2):
            min_counts_per_order.append([])

        for n in range(1, num_train_sets + 1):
            # the next line shouldn't fail since the data-dir did validate correctly.
            name = names[n]
            if name in name_to_mincounts:
                this_mincounts = name_to_mincounts[name]
                names_used.add(name)
            elif 'default' in name_to_mincounts:
                this_mincounts = name_to_mincounts['default']
                names_used.add('default')
            else:
                sys.exit("get_counts.py: invalid min-counts --min-counts='{0}' since there "
                         "is no min-count specified for {1}.".format(min_counts, name))
            for o in range(ngram_order - 2):
                min_counts_per_order[o].append(this_mincounts[o])

        ans = ' '.join([ ','.join([str(int(x)) if x == int(x) else str(x)
                                   for x in array])
                         for array in min_counts_per_order ])
        for name in name_to_mincounts.keys():
            if not name in names_used:
                sys.exit("get_counts.py: invalid min-counts --min-counts='{0}' since the key "
                         "{1} is never used.".format(min_counts, name))
    if args.verbose == 'true':
        print("get_counts.py: converted min-counts from --min-counts='{0}' to '{1}'".format(
                min_counts, ans))

    # test whether ans is all ones, and warn if so.
    a = ans.replace(',', ' ').split()
    if a == [ '1' ] * len(a):
        print("get_counts.py: **warning: --min-counts={0} is equivalent to not applying any "
              "min-counts, it would be more efficient not to use the option at all, or "
              "to set it to the empty string.".format(min_counts))
    return ans




# save the n-gram order.
def SaveNgramOrder(dest_count_dir, ngram_order):
    try:
        f = open('{0}/ngram_order'.format(dest_count_dir), 'w')
    except:
        ExitProgram('error opening file {0}/ngram_order for writing'.format(dest_count_dir))
    assert ngram_order >= 2
    print(ngram_order, file=f)
    f.close()


# this function dumps the counts to disk.

#  if num_splits == 0 [relevant when we're not using min-counts], then it dumps
# its output to {dest_count_dir}/int.{n}.{o} with n = 1..num_train_sets,
# o=2..ngram_order.  (note: n is supplied to this function).
#
# If num-splits >= 1 [relevant when we're using min-counts], then it dumps its output
## {dest_count_dir}/int.{n}.split{j} with n = 1..num_train_sets, j=1..num_splits.

def GetCountsSingleProcess(source_int_dir, dest_count_dir, ngram_order, n, num_splits = 0):
    if num_splits == 0:
        int_counts_output = "/dev/null " + " ".join([ "{0}/int.{1}.{2}".format(dest_count_dir, n, o)
                                                      for o in range(2, ngram_order + 1) ])
    else:
        assert num_splits >= 1
        int_counts_output = '/dev/stdout | split-int-counts ' + \
            ' '.join([ "{0}/int.{1}.split{2}".format(dest_count_dir, n, j)
                       for j in range(1, num_splits + 1) ])

    command = "bash -c 'set -o pipefail; export LC_ALL=C; gunzip -c {source_int_dir}/{n}.txt.gz | "\
            "get-text-counts {ngram_order} | sort {mem_opt}| uniq -c | "\
            "get-int-counts {int_counts_output}'".format(source_int_dir = source_int_dir,
                                              n = n , ngram_order = ngram_order,
                                              mem_opt = sort_mem_opt,
                                              int_counts_output = int_counts_output)
    log_file = "{dest_count_dir}/log/get_counts.{n}.log".format(
        dest_count_dir = dest_count_dir, n = n)
    RunCommand(command, log_file, args.verbose == 'true')


# This function uses multiple parallel processes to dumps the counts to files.
# if num_splits == 0 [relevant when we're not using min-counts], then it dumps its output to
# {dest_count_dir}/int.{n}.{o} with n = 1..num_train_sets, o=2..ngram_order.
# (note: n is supplied to this function).
#
# If num-splits >= 1 [relevant when we're using min-counts], then it dumps its output
## {dest_count_dir}/int.{n}.split{j} with n = 1..num_train_sets, j=1..num_splits.

# This function uses multiple processes (num_proc) in parallel to run
# 'get-text-counts' (this tends to be the bottleneck).
# It will use just one process if the amount of data is quite small or if
# the platform is Cygwin (where named pipes don't work)
def GetCountsMultiProcess(source_int_dir, dest_count_dir, ngram_order, n, num_proc,
                          num_splits = 0):
    try:
        file_size = os.path.getsize('{0}/{1}.txt.gz'.format(source_int_dir, n))
    except:
        ExitProgram('get_counts.py: error getting file size of '
                    '{0}/{1}.txt.gz'.format(source_int_dir, n))

    if IsCygwin() or num_proc <= 1 or file_size < 1000000:
        if num_proc > 1 and file_size >= 1000000:
            # it's only because of Cygwin that we're not using multiple
            # processes this merits a warning.
            print("get_counts.py: cygwin platform detected so named pipes won't work; "
                  "using a single process (will be slower)")
        return GetCountsSingleProcess(source_int_dir, dest_count_dir,
                                      ngram_order, n, num_splits)

    if num_splits == 0:
        int_counts_output = "/dev/null " + " ".join([ "{0}/int.{1}.{2}".format(dest_count_dir, n, o)
                                                      for o in range(2, ngram_order + 1) ])
    else:
        assert num_splits >= 1
        int_counts_output = '/dev/stdout | split-int-counts ' + \
            ' '.join([ "{0}/int.{1}.split{2}".format(dest_count_dir, n, j)
                       for j in range(1, num_splits + 1) ])

    try:
        # we want a temporary directory on a local file system
        # for
        tempdir = tempfile.mkdtemp()
    except Exception as e:
        ExitProgram("Error creating temporary directory: " + str(e))

    # This has several pipes for the internal processing that write to and read
    # from other internal pipes; and we can't do this using '|' in the shell, we
    # need to use mkfifo.  This does not work properly on cygwin.

    log_file = "{dest_count_dir}/log/get_counts.{n}.log".format(
        dest_count_dir = dest_count_dir, n = n)
    test_command = "bash -c 'set -o pipefail; (echo a; echo b) | "\
        "distribute-input-lines /dev/null /dev/null'";
    # We run the following command just to make sure distribute-input-lines is
    # on the path and compiled, since we get hard-to-debug errors if it fails.
    RunCommand(test_command, log_file)

    # we use "bash -c '...'" to make sure it gets run in bash, since
    # for example 'set -o pipefail' would only work in bash.
    command = ("bash -c 'set -o pipefail; set -e; export LC_ALL=C; mkdir -p {0}; ".format(tempdir) +
               ''.join(['mkfifo {0}/{1}; '.format(tempdir, p)
                        for p in range(num_proc) ]) +
               'trap "rm -r {0}" SIGINT SIGKILL SIGTERM EXIT; '.format(tempdir) +
               'gunzip -c {0}/{1}.txt.gz | distribute-input-lines '.format(source_int_dir, n) +
               ' '.join(['{0}/{1}'.format(tempdir, p) for p in range(num_proc)]) + '& ' +
               'sort -m {0}'.format(sort_mem_opt) +
               ' '.join([ '<(get-text-counts {0} <{1}/{2} | sort {3})'.format(ngram_order, tempdir, p, sort_mem_opt)
                                       for p in range(num_proc) ]) +
               '| uniq -c | get-int-counts {0}'.format(int_counts_output) +
               "'") # end the quote from the 'bash -c'.

    RunCommand(command, log_file, args.verbose=='true')


# This function applies the min-counts (it is only called if you supplied the
# --min-counts option to this script).  It reads in the data dumped by
# GetCounts.  It dumps the files into {dest_count_dir}/int.{n}.split{j}.{o}
# for n = 1...num_train_sets j = 1..num_jobs, and o=2..ngram_order.  [note: j is
# supplied to this function].
def EnforceMinCounts(dest_count_dir, formatted_min_counts, ngram_order, num_train_sets, j):
    inputs = ' '.join([ "{0}/int.{1}.split{2}".format(dest_count_dir, n, j)
                        for n in range(1, num_train_sets + 1) ])
    outputs = ' '.join([' '.join([ '{0}/int.{1}.split{2}.{3}'.format(dest_count_dir, n, j, o)
                                   for o in range(2, ngram_order + 1) ])
                        for n in range(1, num_train_sets + 1) ])
    # e.g. suppose j is 2 and ngram_order is 4, outputs would be as follows
    # [assuming brace expansion].:
    # outputs = dir/int.1.split2.{2,3,4} dir/int.2.split2.{2,3,4} ...
    #    dir/int.{num_train_sets}.split2.{2,3,4}

    command = "int-counts-enforce-min-counts {ngram_order} {formatted_min_counts} {inputs} "\
               "{outputs}".format(
        ngram_order = ngram_order, formatted_min_counts = formatted_min_counts,
        inputs = inputs, outputs = outputs, j = j)

    log_file = '{0}/log/enforce_min_counts.{1}.log'.format(dest_count_dir, j)

    RunCommand(command, log_file, args.verbose=='true')


# This function merges counts from multiple jobs, that have been split up by
# most recent history-word (it is only called if you supplied the --min-counts
# option to this script).  It reads in the data dumped by EnforceMinCounts.
# it merges the files into {dest_count_dir}/int.{n}.{o}.
def MergeCounts(dest_count_dir, num_jobs, n, o):
    if num_jobs > 1:
        command = ('merge-int-counts ' +
                   ' '.join(['{0}/int.{1}.split{2}.{3}'.format(dest_count_dir, n, j, o)
                             for j in range(1, num_jobs + 1)]) +
                   '>{0}/int.{1}.{2}'.format(dest_count_dir, n, o))
        log_file = '{0}/log/merge_counts.{1}.{2}.log'.format(dest_count_dir, n, o)
        RunCommand(command, log_file, args.verbose=='true')
    else:
        assert num_jobs == 1
        # we can just move the file if num-jobs == 1.
        try:
            os.remove('{0}/int.{1}.{2}'.format(dest_count_dir, n, o))
        except:
            pass
        os.rename('{0}/int.{1}.split1.{2}'.format(dest_count_dir, n, o),
                  '{0}/int.{1}.{2}'.format(dest_count_dir, n, o))

# we also want to merge the files $dir/int.dev.{2,3,...} into a single file
# that contains all the dev-data's counts; this will be used in likelihood
# evaluation.
def MergeDevData(dest_count_dir, ngram_order):
    command = ("merge-int-counts " + ' '.join([ dest_count_dir + "/int.dev." + str(n)
                                                for n in range(2, ngram_order + 1) ]) +
               ">{0}/int.dev".format(dest_count_dir))
    log_file = dest_count_dir + '/log/merge_dev_counts.log'
    RunCommand(command, log_file, args.verbose=='true')

# this function returns the value and unit of the max_memory
# if max_memory is in format of "integer + letter/%", like  "10G", it returns (10, 'G')
# if max_memory contains no letter, like "10000", it returns (10000, '')
# we assume the input string is not empty since when it is empty we never call this function
def ParseMemoryString(s):
    if not s[-1].isdigit():
        return (int(s[:-1]), s[-1])
    else:
        return (int(s), '')

# make sure 'scripts' and 'src' directory are on the path
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) + "/../src")

if os.system("validate_int_dir.py " + args.source_int_dir) != 0:
    ExitProgram("command validate_int_dir.py {0} failed".format(args.source_int_dir))

if args.ngram_order < 2:
    ExitProgram("ngram-order is {0}; it must be at least 2.  If you "
                "want a unigram LM, do it by hand".format(args.ngram_order))

# read the variable 'num_train_sets'
# from the corresponding file in source_int_dir  This shouldn't fail
# because we just called validate_int-dir.py..
f = open(args.source_int_dir + "/num_train_sets")
num_train_sets = int(f.readline())
f.close()

# set the memory restriction for "sort"
sort_mem_opt = ''
if args.max_memory != '':
    if args.dump_counts_parallel == 'true':
        (value, unit) = ParseMemoryString(args.max_memory)
        sub_memory = value/num_train_sets
        if sub_memory != float(value)/num_train_sets:
            if unit in ['K', '']:
                sub_memory = value*1024/num_train_sets
                unit = 'b'
            if unit == 'M':
                sub_memory = value*1024/num_train_sets
                unit = 'K'
            if unit == 'G':
                sub_memory = value*1024/num_train_sets
                unit = 'M'
            if (unit in ['b', '%']) and (sub_memory == 0):
                ExitProgram("max_memory for each of the {0} train sets is {1}{2}."
                            "Please reset a larger max_memory value".format(
                            num_train_sets, float(value)/num_train_sets, unit))
        sub_max_memory = str(sub_memory) + unit
        sort_mem_opt = ("--buffer-size={0} ".format(sub_max_memory))
    else:
        sort_mem_opt = ("--buffer-size={0} ".format(args.max_memory))

if not os.path.isdir(args.dest_count_dir):
    try:
        os.makedirs(args.dest_count_dir+'/log')
    except:
        ExitProgram("error creating directory " + args.dest_count_dir)


CopyMetaInfo(args.source_int_dir, args.dest_count_dir)

SaveNgramOrder(args.dest_count_dir, args.ngram_order)


if args.min_counts == '':
    # no min-counts specified: use normal pipeline.
    print("get_counts.py: dumping counts", file=sys.stderr)
    threads = []
    for n in [ "dev" ] + range(1, num_train_sets + 1):
        threads.append(threading.Thread(target = GetCountsMultiProcess,
                                        args = [args.source_int_dir, args.dest_count_dir,
                                                args.ngram_order, str(n), args.num_count_jobs] ))
        threads[-1].start()
        if args.dump_counts_parallel == 'false':
            threads[-1].join()

    if args.dump_counts_parallel == 'true':
        for t in threads:
            t.join()

    MergeDevData(args.dest_count_dir, args.ngram_order)
    print("get_counts.py: done", file=sys.stderr)

else:
    # First process the dev data, the min-counts aren't relevant here.
    GetCountsSingleProcess(args.source_int_dir, args.dest_count_dir,
                           args.ngram_order, 'dev')
    MergeDevData(args.dest_count_dir, args.ngram_order)

    num_mc_jobs = args.num_min_count_jobs
    if num_mc_jobs < 1:
        ExitProgram("bad option --num-min-count-jobs={0}".format(num_mc_jobs))
    formatted_min_counts = FormatMinCounts(args.source_int_dir,
                                           num_train_sets,
                                           args.ngram_order,
                                           args.min_counts)

    if not num_mc_jobs >= 1:
        sys.exit("get_counts.py: invalid option --num-jobs={0}".format(num_mc_jobs))

    # First, dump the counts split up by most-recent-history instead of ngram-order.
    print("get_counts.py: dumping counts", file=sys.stderr)
    threads = []
    for n in range(1, num_train_sets + 1):
        threads.append(threading.Thread(target = GetCountsMultiProcess,
                                        args = [args.source_int_dir, args.dest_count_dir,
                                                args.ngram_order, str(n), args.num_count_jobs,
                                                num_mc_jobs] ))
        threads[-1].start()
        if args.dump_counts_parallel == 'false':
            threads[-1].join()
    if args.dump_counts_parallel == 'true':
        for t in threads:
            t.join()

    # Next, apply the min-counts.
    print("get_counts.py: applying min-counts", file=sys.stderr)
    threads = []
    for j in range(1, num_mc_jobs + 1):
        threads.append(threading.Thread(target = EnforceMinCounts,
                                        args = [args.dest_count_dir, formatted_min_counts,
                                                args.ngram_order, num_train_sets, j]))
        threads[-1].start()

    for t in threads:
        t.join()

    if args.cleanup == 'true':
        for n in range(1, num_train_sets + 1):
            for j in range(1, num_mc_jobs + 1):
                os.remove("{0}/int.{1}.split{2}".format(
                        args.dest_count_dir, n, j))

    print("get_counts.py: merging counts", file=sys.stderr)
    threads = []
    for n in range(1, num_train_sets + 1):
        for o in range(2, args.ngram_order + 1):
            threads.append(threading.Thread(target = MergeCounts,
                                            args = [args.dest_count_dir,
                                                    num_mc_jobs, n, o]))
            threads[-1].start()
    for t in threads:
        t.join()

    if args.cleanup == 'true':
        for n in range(1, num_train_sets + 1):
            for j in range(1, args.num_min_count_jobs + 1):
                for o in range(2, args.ngram_order + 1):
                    try:
                        os.remove("{0}/int.{1}.split{2}.{3}".format(
                                args.dest_count_dir, n, j, o))
                    except:
                        pass
    print("get_counts.py: finished.", file=sys.stderr)


if os.system("validate_count_dir.py " + args.dest_count_dir) != 0:
    ExitProgram("command validate_count_dir.py {0} failed".format(args.dest_count_dir))

