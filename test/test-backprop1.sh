#!/bin/bash

# this just documents some commands we ran, it's not an automated test.
# run it from this directory.

export PATH=$PATH:../src

( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 2 | sort | uniq -c | get-int-counts int.1gram int.2gram
merge-counts int.2gram,1.0 > merged.2gram
discount-counts 0.8 0.7 0.6 merged.2gram float.2gram 1gram
discount-counts-1gram 20 <1gram >float.1gram
echo 10 11 12 | get-text-counts 2 | sort | uniq -c | get-int-counts dev.int
merge-float-counts float.2gram float.1gram > float.all
x=$(compute-probs float.all dev.int float_derivs.1gram float_derivs.2gram)
echo $x
prob=$(echo $x | awk '{print $2}')

# 4 -10.0067209

perl -e "print $prob + $(perturb-float-counts 1 float.1gram float_derivs.1gram float_perturbed.1gram) . \"\n\"; "
# -10.006247441
compute-probs <(merge-float-counts float.2gram float_perturbed.1gram) dev.int /dev/null /dev/null
# 4 -10.00624859

### Now testing 2-gram derivatives.
perl -e "print $prob + $(perturb-float-counts 2 float.2gram float_derivs.2gram float_perturbed.2gram) . \"\n\"; "
# -10.007106366
  compute-probs <(merge-float-counts float_perturbed.2gram float.1gram) dev.int /dev/null /dev/null
# 4 -10.00710714


## checking that the derivs w.r.t scaling any of the sets of counts
## by the same constant, are zero.
print-float-derivs float.2gram float_derivs.2gram | awk '{for (n=1;n<=NF;n++) print $n;}' | perl -ane 'if (m/[=>](\S+),d=(.+)/) { $x = $1 * $2;  print "$x\n"; }' | awk '{x+=$1} END{print x; }'
# good, it's small:
# 1.5e-06

print-float-derivs float.1gram float_derivs.1gram | awk '{for (n=1;n<=NF;n++) print $n;}' | perl -ane 'if (m/[=>](\S+),d=(.+)/) { $x = $1 * $2;  print "$x\n"; }' | awk '{x+=$1} END{print x; }'
# print-float-derivs: printed 1 LM states, with 19 individual n-grams.
# -2.49332e-07 # small; good.

