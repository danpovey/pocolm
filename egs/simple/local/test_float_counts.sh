#!/usr/bin/env bash


export PATH=$PATH:../../src
float-counts-to-float-stats 10 a b c <data/work/float.all
echo "#float stats:"
merge-float-counts a b c > data/work/float_stats.all

print-float-counts <data/work/float_stats.all
# #float stats:
# merge-float-counts: read 1 + 4 + 4 = 9 LM states.
#  [ ]: total=1.54831 discount=0 2->0.266691 3->0.595 4->0.281471 5->0.156985 6->0.248162
#  [ 1 ]: total=1 discount=0.482721 4->0.517279
#  [ 4 ]: total=0.421176 discount=0.421176 5->0 6->0
#  [ 4 1 ]: total=1 discount=0.421176 5->0.251765 6->0.327059
#  [ 5 ]: total=0.246471 discount=0.246471 6->0
#  [ 5 4 ]: total=0.5 discount=0.246471 6->0.253529
#  [ 6 ]: total=0.397941 discount=0.397941 2->0
#  [ 6 4 ]: total=0.5 discount=0.198971 2->0.301029
#  [ 6 5 ]: total=0.5 discount=0.198971 2->0.301029


echo "#float counts:"
print-float-counts <data/work/float.all

# print-float-counts: printed 9 LM states, with 15 individual n-grams.
# #float counts:
#  [ ]: total=2.04 discount=0 2->0.34875 3->0.595 4->0.39875 5->0.22875 6->0.46875
#  [ 1 ]: total=1 discount=0.6 4->0.4
#  [ 4 ]: total=0.8 discount=0.64 5->0.08 6->0.08
#  [ 4 1 ]: total=1 discount=0.8 5->0.1 6->0.1
#  [ 5 ]: total=0.4 discount=0.32 6->0.08
#  [ 5 4 ]: total=0.5 discount=0.4 6->0.1
#  [ 6 ]: total=0.8 discount=0.48 2->0.32
#  [ 6 4 ]: total=0.5 discount=0.4 2->0.1
#  [ 6 5 ]: total=0.5 discount=0.4 2->0.1
# print-float-counts: printed 9 LM states, with 15 individual n-grams.

# using formula to get the  count for [6 5 ] -> 2.
# perl -e 'print 0.1 + (0.4 * ((0.32/0.8) + (0.48/0.8) * (0.34875/2.04))) '
# 0.301029411764706


# OK, now test float-counts-estimate
float-counts-estimate 10 data/work/float.all data/work/float_stats.all a1 b1 c1
#float-counts-estimate: logprob per word was -1.35433 over 3.5 words.
#float-counts-estimate: auxiliary function improvement per word was [ 0 + 5.99452e-08 + -6.81196e-09 ] = 5.31333e-08
merge-float-counts a1 b1 c1 > data/work/float2.all
print-float-counts <  data/work/float2.all
float-counts-estimate 10 data/work/float2.all data/work/float_stats.all a1 b1 c1
#good: it's the same.
#float-counts-estimate: logprob per word was -1.35433 over 3.5 words.
#float-counts-estimate: auxiliary function improvement per word was [ 0 + 1.63487e-08 + 2.38419e-08 ] = 4.01906e-08

# merge-float-counts a1 b1 c1 | print-float-counts
# merge-float-counts: read 1 + 4 + 4 = 9 LM states.
#  [ ]: total=2.04 discount=0 2->0.34875 3->0.595 4->0.39875 5->0.22875 6->0.46875
#  [ 1 ]: total=1 discount=0.6 4->0.4
#  [ 4 ]: total=0.8 discount=0.64 5->0.08 6->0.08
#  [ 4 1 ]: total=1 discount=0.8 5->0.1 6->0.1
#  [ 5 ]: total=0.4 discount=0.32 6->0.08
#  [ 5 4 ]: total=0.5 discount=0.4 6->0.1
#  [ 6 ]: total=0.8 discount=0.48 2->0.32
#  [ 6 4 ]: total=0.5 discount=0.4 2->0.1
#  [ 6 5 ]: total=0.5 discount=0.4 2->0.1
# print-float-counts: printed 9 LM states, with 15 individual n-grams.


