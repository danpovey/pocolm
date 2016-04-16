
# Why we are creating pocolm

Pocolm exists to make it possible to create better interpolated and pruned language models
(i.e. better than from standard toolkits like SRILM and KenLM).

The pruning algorithm we use is better than the entropy pruning used by (for instance)
SRILM because it takes into account the relative occupation counts of LM states, and also
uses a more exact formula that takes into account the change in likelihood of the
backed-off-to-state.

The interpolation algorithm is better because... well it's quite complicated, I'll
explain this later.


