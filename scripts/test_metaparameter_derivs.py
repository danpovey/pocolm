#!/usr/bin/env python

# we're using python 3.x style print but want it to work in python 2.x,
from __future__ import print_function
import re, os, argparse, sys, math, warnings

parser = argparse.ArgumentParser(description="This script, to be used in testing "
                                 "the framework, repeatedly calls get_objf_and_derivs.py "
                                 "and helps you check that the derivatives agree with "
                                 "those computed by the difference method.")


parser.add_argument("--delta", type=float, default=1.0e-04,
                    help="Parameter-change with which to compute derivatives using "
                    "the difference method")

parser.add_argument("metaparameter_file",
                    help="Filename of metaparameters at which to compute derivatives")
parser.add_argument("count_dir",
                    help="Directory in which to find counts")
parser.add_argument("work_dir",
                    help="Directory in which to create temporary files")

args = parser.parse_args()

# put the derivatives and objective functions in directory work/derivs
temp_dir = args.work_dir + os.sep + "temp"
derivs_dir = args.work_dir + os.sep + "derivs"

if not os.path.exists(derivs_dir):
    os.makedirs(derivs_dir)

# Add the script dir to the path.
os.environ['PATH'] = (os.environ['PATH'] + os.pathsep +
                      os.path.abspath(os.path.dirname(sys.argv[0])));

def RunCommand(command):
    # print the command for logging
    print(command, file=sys.stderr)
    if os.system(command) != 0:
        sys.exit("test_metaparameter_derivs.py: error running command: " + command)

# make the first call to get_objf_and_derivs.py, getting the derivatives
# and the objf.
command = ("get_objf_and_derivs.py --derivs-out={derivs}/derivs {counts} "
           "{metaparams} {derivs}/objf {temp} ".format(
        derivs = derivs_dir, temp = temp_dir, counts = args.count_dir,
        metaparams = args.metaparameter_file))

RunCommand(command)

# get baseline objective function (before perturbing metaparameters).
f = open(derivs_dir + "/objf")
baseline_objf = float(f.readline())
f.close()

# let the metaparameters be a list of 2-tuples
# (metaparameter-name, value)
metaparameters = []
f = open(args.metaparameter_file, "r")
for line in f.readlines():
    [ name, value ] = line.split();
    value = float(value)
    metaparameters.append( (name, value) )
f.close()

num_metaparameters = len(metaparameters)

def WriteMetaparameters(metaparameters, file):
    f = open(file, "w")
    for t in metaparameters:
        ( name, value ) = t
        print("{0} {1}".format(name, value), file=f)
    f.close()

modified_objfs = [0] * num_metaparameters
deltas = [0] * num_metaparameters

for i in range(num_metaparameters):
    # always perturb towards 0.5, to avoid exiting the domain [0, 1].
    modified_metaparameters = [ t for t in metaparameters ]
    this_delta = (args.delta if metaparameters[i][1] < 0.5 else -args.delta)
    modified_metaparameters[i] = (metaparameters[i][0],
                                  metaparameters[i][1] + this_delta)
    WriteMetaparameters(modified_metaparameters,
                        derivs_dir + os.sep + "metaparameters.{0}".format(i))

    command = ("get_objf_and_derivs.py {counts} {derivs}/metaparameters.{i} "
               "{derivs}/objf.{i} {temp}".format(
            counts = args.count_dir, derivs = derivs_dir, i = i, temp = temp_dir))
    print("test_metaparameter_derivs.py: running command " + command,
          file=sys.stderr)
    RunCommand(command)
    f = open("{derivs}/objf.{i}".format(derivs = derivs_dir, i = i), "r")
    modified_objfs[i] = float(f.readline())
    deltas[i] = this_delta
    f.close()

# Now compare the computed derivatives with the 'difference-method' derivatives
derivs = []
f = open(derivs_dir + "/derivs", "r")
derivs = [ float(line.split()[1]) for line in f.readlines() ]
f.close()

output_file = derivs_dir + "/derivs_compare";
print ("test_metaparameters_derivs.py: writing the analytical and "
       "difference-method derivatives to " + output_file, file=sys.stderr)
f = open(output_file, "w")
print("#parameter-name    analytical derivative    difference-method derivative",
      file=f)
analytical_sumsq = 0.0
disagreement_sumsq = 0.0
for i in range(num_metaparameters):
    analytical = derivs[i]
    difference_method = (modified_objfs[i] - baseline_objf) / deltas[i]
    print(metaparameters[i][0], analytical,
          difference_method, file=f)
    disagreement = difference_method - analytical
    analytical_sumsq += analytical * analytical
    disagreement_sumsq += disagreement * disagreement
f.close()

percentage_agreement = (100.0 * (math.sqrt(analytical_sumsq) - math.sqrt(disagreement_sumsq)) /
                        math.sqrt(analytical_sumsq))
print("test_metaparameter_derivs.py: analytical and difference-method "
      "derivatives agree {0}%".format(percentage_agreement), file=sys.stderr)
if percentage_agreement < 98.0:
    sys.exit("analytical and difference-method derivatives differ too much.")

