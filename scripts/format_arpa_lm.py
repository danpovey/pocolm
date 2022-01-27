#!/usr/bin/env python3

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import os
import argparse
import sys
import subprocess
import glob

# If the encoding of the default sys.stdout is not utf-8,
# force it to be utf-8. See PR #95.
if hasattr(sys.stdout, 'encoding') and sys.stdout.encoding.lower() != "utf-8":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())
    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

# make sure scripts/internal is on the pythonpath.
sys.path = [os.path.abspath(os.path.dirname(sys.argv[0])) + "/internal"
            ] + sys.path

# for ExitProgram
from pocolm_common import ExitProgram

parser = argparse.ArgumentParser(
    description="This script turns a pocolm language model "
    "directory as created by make_lm_dir.py, into an ARPA-format "
    "language model.  The ARPA LM is written to the standard "
    "output and may be redirected as desired.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
    "--temp-dir",
    type=str,
    help="Temporary directory for use by 'sort'; if not provided, "
    "we use the destination directory lm_dir")
parser.add_argument("--max-memory",
                    type=str,
                    default='',
                    help="Memory limitation for sort.")
parser.add_argument("lm_dir",
                    help="Directory of the source language model, as created "
                    "by make_lm_dir.py")

args = parser.parse_args()

if args.temp_dir is None:
    args.temp_dir = args.lm_dir

# Add the script dir and the src dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) +
                      os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])) +
                      "/../src")
# this will affect the program "sort" that we call.
os.environ['LC_ALL'] = 'C'

# this temporary directory will be used by "sort".
os.environ['TMPDIR'] = args.temp_dir

if os.system("validate_lm_dir.py " + args.lm_dir) != 0:
    sys.exit("format_arpa_lm.py: failed to validate input LM directory")

if not os.path.isdir(args.temp_dir):
    sys.exit("format_arpa_lm.py: expected directory {0} to exist.".format(
        args.temp_dir))

# verify the input string max_memory
if args.max_memory != '':
    # valid string max_memory must have at least two items
    if len(args.max_memory) >= 2:
        s = args.max_memory
        # valid string max_memory can be formatted as:
        # "a positive integer + a letter or a '%'" or "a positive integer"
        # the unit of memory size can also be 'T', 'P', 'E', 'Z', or 'Y'. They
        # are not included here considering their rare use in practice
        if s[-1] in ['b', 'B', '%', 'k', 'K', 'm', 'M', 'g', 'G'
                     ] or s[-1].isdigit():
            for x in s[:-1]:
                if not x.isdigit():
                    sys.exit(
                        "format_arpa_lm.py: --max-memory should be formatted as "
                        "'a positive integer' or 'a positive integer appended "
                        "with 'b', 'K', 'M','G', or '%''.")
            # max memory size must be larger than zero
            if int(s[:-1]) == 0:
                sys.exit("format_arpa_lm.py: --max-memory must be > 0 {unit}.".
                         format(unit=s[-1]))
        else:
            sys.exit(
                "format_arpa_lm.py: the format of string --max-memory is not correct."
            )
    else:
        sys.exit(
            "format_arpa_lm.py: the lenght of string --max-memory must >= 2.")
    if args.max_memory[-1] == 'B':  # sort seems not recognize 'B'
        args.max_memory[-1] = 'b'

# this function returns the value and unit of the max_memory
# if max_memory is in format of "integer + letter/%", like  "10G", it returns (10, 'G')
# if max_memory contains no letter, like "10000", it returns (10000, '')
# we assume the input string is not empty since when it is empty we never call this function


def ParseMemoryString(s):
    if not s[-1].isdigit():
        return (int(s[:-1]), s[-1])
    else:
        return (int(s), '')


def DivideMemory(total, n):
    (value, unit) = ParseMemoryString(total)
    sub_memory = value / n
    if sub_memory != float(value) / n:
        if unit in ['K', 'k', '']:
            sub_memory = value * 1024 / n
            unit = 'b'
        elif unit in ['M', 'm']:
            sub_memory = value * 1024 / n
            unit = 'K'
        elif unit in ['G', 'g']:
            sub_memory = value * 1024 / n
            unit = 'M'
        elif (unit in ['B', 'b', '%']) and (sub_memory == 0):
            ExitProgram("max_memory for each of the {0} train sets is {1}{2}."
                        "Please reset a larger max_memory value".format(
                            n,
                            float(value) / n, unit))
        else:
            ExitProgram("Invalid format for max_memory. "
                        "Please 'man sort' to see how to set buffer size.")
    return str(int(sub_memory)) + unit


# read ngram order.
f = open(args.lm_dir + "/ngram_order", encoding="utf-8")
ngram_order = int(f.readline())
f.close()

# work out num_words.  Note: this doesn't count epsilon; it's the
# same as the highest numbered word symbol.
line = subprocess.check_output(['tail', '-n', '1', args.lm_dir + '/words.txt'])
try:
    [last_word, num_words] = line.split()
    num_words = int(num_words)
    assert num_words > 3
except:
    sys.exit(
        "format_arpa_lm.py: error getting num-words from {0}/words.txt".format(
            args.lm_dir))

if not os.path.exists(args.lm_dir + "/num_splits"):
    if args.max_memory == '':
        mem_opt = ''
    else:
        mem_opt = "--buffer-size={0}".format(args.max_memory)
    # LM counts are in one file.
    command = (
        "float-counts-to-pre-arpa {ngram_order} {num_words} {lm_dir}/float.all | sort {mem_opt} |"
        " pre-arpa-to-arpa {lm_dir}/words.txt".format(ngram_order=ngram_order,
                                                      num_words=num_words,
                                                      lm_dir=args.lm_dir,
                                                      mem_opt=mem_opt))
else:
    # reading num_splits shouldn't fail, we validated the directory.
    num_splits = int(
        open(args.lm_dir + "/num_splits", encoding="utf-8").readline())
    if args.max_memory == '':
        mem_opt = ''
    else:
        mem_opt = "--buffer-size={0}".format(
            DivideMemory(args.max_memory, num_splits + 1))

    [
        os.remove(x)
        for x in glob.glob("{lm_dir}/.*.error".format(lm_dir=args.lm_dir))
    ]
    # create command line of the form:
    # sort -m <(command1) <(command2) ... <(commandN) | pre-arpa-to-arpa ...
    # we put it all inside bash -c, because the process substitution <(command)
    # won't always work in /bin/sh.
    command = (
        "bash -c 'sort -m {mem_opt} ".format(mem_opt=mem_opt)
        +  # sort -m merges already-sorted files.
        " ".join([
            "<(float-counts-to-pre-arpa {opt} {ngram_order} {num_words} "
            "{lm_dir}/float.all.{n} | sort {mem_opt} || touch {lm_dir}/.{n}.error)"
            .format(opt=('--no-unigram' if n > 1 else ''),
                    ngram_order=ngram_order,
                    num_words=num_words,
                    lm_dir=args.lm_dir,
                    n=n,
                    mem_opt=mem_opt) for n in range(1, num_splits + 1)
        ]) +
        " | pre-arpa-to-arpa {lm_dir}/words.txt'".format(lm_dir=args.lm_dir))

print("format_arpa_lm.py: running " + command, file=sys.stderr)

ret = os.system(command)

if ret != 0:
    sys.exit("format_arpa_lm.py: command {0} exited with status {1}".format(
        command, ret))

if len(glob.glob("{lm_dir}/.*.error".format(lm_dir=args.lm_dir))) > 0:
    ExitProgram(
        "Something went wrong for the float-counts-to-pre-arpa or sort command."
    )

print("format_arpa_lm.py: succeeded formatting ARPA lm from {0}".format(
    args.lm_dir),
      file=sys.stderr)
