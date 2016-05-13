#!/bin/bash

# this just documents some commands we ran, it's not an automated test.
# run it from this directory.

export PATH=$PATH:../src

( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 2 | sort | uniq -c | get-int-counts int.1gram int.2gram
merge-counts int.2gram,1.0 > merged.2gram
discount-counts 0.8 0.7 0.6 0.4 merged.2gram float.2gram 1gram
discount-counts-1gram 20 <1gram >float.1gram
echo 10 11 12 | get-text-counts 2 | sort | uniq -c | get-int-counts dev.int
merge-float-counts float.2gram float.1gram > float.all
x=$(compute-probs float.all dev.int float_derivs.1gram float_derivs.2gram)
echo $x
prob=$(echo $x | awk '{print $2}')
# 4 -10.0067209

discount-counts-1gram-backward 1gram float.1gram float_derivs.1gram derivs.1gram


print-derivs 1gram derivs.1gram | awk '{for (n=1;n<=NF;n++) print $n;}' | \
  perl -ane 'if (m/\((.+)\),d=\((.+)\)/) { @A = split(",", $1); @B = split(",", $2); $tot = 0; for ($n = 0; $n < 4; $n++) { $tot += $A[$n] * $B[$n]; } print "$tot\n"; } ' | \
    awk '{x+=$1; } END{print x;}'
# again, it's small:
# -1.114e-06

perl -e "print $prob + $(perturb-counts 2 1gram derivs.1gram perturbed.1gram) . \"\n\"; "
#-10.00413492

  compute-probs <(discount-counts-1gram 20 <perturbed.1gram| merge-float-counts float.2gram /dev/stdin) dev.int /dev/null /dev/null
# 4 -10.00414324


# OK, now backprop through the command
#discount-counts 0.8 0.7 0.6 0.4 merged.2gram float.2gram 1gram

discount-counts-backward 0.8 0.7 0.6 0.4 merged.2gram float.2gram float_derivs.2gram 1gram derivs.1gram derivs_merged.2gram


perl -e "print $prob + $(perturb-counts 2 merged.2gram derivs_merged.2gram perturbed_merged.2gram) . \"\n\"; "
# -10.00412881

discount-counts 0.8 0.7 0.6 0.4 perturbed_merged.2gram float.2gram 1gram
discount-counts-1gram 20 <1gram >float.1gram
merge-float-counts float.2gram float.1gram > float.all
compute-probs float.all dev.int /dev/null /dev/null
# 4 -10.00419021
# hmm, pretty close.
# Caution, the above few lines overwrote a bunch of files, we have to run this
# script from the beginning the second time.


print-derivs merged.2gram derivs_merged.2gram | awk '{for (n=1;n<=NF;n++) print $n;}' | \
  perl -ane 'if (m/\((.+)\),d=\((.+)\)/) { @A = split(",", $1); @B = split(",", $2); $tot = 0; for ($n = 0; $n < 4; $n++) { $tot += $A[$n] * $B[$n]; } print "$tot\n"; } ' | \
    awk '{x+=$1; } END{print x;}'
# 4.6e-06


merge-counts-backward merged.2gram derivs_merged.2gram  int.2gram 1.0

