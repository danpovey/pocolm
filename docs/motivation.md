
# Why we are creating pocolm

Pocolm exists to make it possible to create better interpolated and pruned
ARPA-type language models (i.e. better than from standard toolkits like SRILM
and KenLM).  This writeup assumes intimate knowledge of Kneser-Ney discounting,
including the "modified, interpolated" extension by Chen and Goodman; please
read "A Bit of Progress in Language Modeling; extended version" by Joshua
Goodman to understand the background.

The pruning algorithm we use is better than the entropy pruning used by (for instance)
SRILM because it takes into account the relative occupation counts of LM states, and also
uses a more exact formula that takes into account the change in likelihood of the
backed-off-to-state.

## The scenario

The scenario that this toolkit is based on is that you have a small amount of
development data in a "target domain", and you have several data sources that
might be more or less relevant to the "target domain".  You also have decided on
a word list.  You provide the training sources, the dev data and the word list;
Pocolm handles the language model building, interpolation and pruning, and spits
out an ARPA format language model at the end.

## The problems we are trying to solve

The current methods of interpolation and pruning, e.g. as used by SRILM, are a
little sub-optimal, for a couple of reasons:

  - The standard method of interpolation is to first estimate the LMs separately
    and then interpolate them, but this is clearly not optimal in the way it
    interacts with backoff.  Consider the case where the sources would get the
    same weight-- you'd want to combine them and estimate the LM together, to
    get more optimal backoff.  Let's call this the ``estimate-then-interpolate
    problem''.

  - The standard pruning method (e.g. Stolcke entropy pruning) is non-optimal
    because when removing probabilities it doesn't update the backed-off-to
    state.

## Our solution

As part of our solution to the ``estimate-then-interpolate problem'', we
interpolate the data sources at the level of data-counts before we estimate
the LMs.  Think of it as a weighted sum.  At this point, anyone familiar with
how language models are discounted will say ``wait a minute-- that can't work''
because these techniques rely on integer counts.   Our method is to treat a count as a
collection of small pieces of different sizes, where we only care about the total
(i.e. the sum of the size of the pices), and the magnitude of the three largest
pieces.  Recall that modified Kneser-Ney discounting only cares about the first
three data counts when deciding the amount to discount.  We extend
that discounting method to variable-size individual counts, as discounting proportions
D1, D2 and D3 of the top-1, top-2 and top-3 largest individual pieces.

## Computing the gradients

The perceptive reader will notice that there is no very obvious way to estimate
the discounting constants D1, D2 and D3 for each n-gram order, and no very
obvious way to estiate the interpolation weights for the language models.  Our
solution to this is to estimate all these hyperparameters based on the
probability of dev data, using a standard numerical optimization technique
(L-BFGS).  The obvious question is, "how do estimate the gradients?".  The simplest
way to estimate the gradients is by the "difference method".  But if there
are 20 hyperparameters, we expect L-BFGS to take at least, say, 10 iterations
to converge acceptably, and computing the derivative would require estimating
the LM 20 times on each iteration-- in total we're estimating the language model
200 times, which seems like a bit too much.  Our solution to this is to
estimate the gradients directly using reverse-mode automatic differentiation
(which is the general case of neural-net backprop).  This allows us to remove
the factor of 20 (or however many hyperparameters there are) in the gradient
computation.

## Applying reverse-mode automatic differentiation

The simplest way to apply reverse-mode automatic differentiation is to keep all
the intermediate quantities of the computation in memory at once.  In principle
this can be done almost as a mechanical process.  But for language model
estimation the obvious approach is is a bit limiting, as we'd like to cover
cases where the data doesn't all fit in memory at once.  (We're using a version
of Patrick Nguyen's sorting trick, see the ``MSRLM'' report).  Anyway, we have a
solution to this, and the details are extremely tedious.  It involves
decomposing the discounting and probability-estimation processes into simple
individual steps, each of which operates on a pipe (i.e. doesn't hold too much
data in memory), and then working out the automatic differentiation operation
for each individual pipe.  We don't have to go in "backwards order" inside the
individual operations, because operations like summing don't "care about" the
order of operations.


## Pruning

The language model pruning method is something we've done before in a different
context, which is an improved, more-exact version of Stolcke pruning.  It operates
on the same principle, but manages to be more optimal because when it removes
a probability from the LM it assigns the removed data to the backed-off-to-state
and updates its probabilities accordingly.  Of course, we take this change into
account when selecting the n-grams to prune away.

