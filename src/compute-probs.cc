// compute-probs.cc

// Copyright     2016  Johns Hopkins University (Author: Daniel Povey)

// See ../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <math.h>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"
#include "lm-state-derivs.h"


/**
   This program reads discounted float-counts of varying order
   (likely merged by merge-counts) from one input,
   and int-counts of data whose probs we want to compute
   (typically dev data) from another input.

   It computes the appropriately backed-off N-gram probabilities
   of each input count, and prints to the stdout the total count
   and the sum of logprobs.  It prints to stderr some diagnostics.

*/

namespace pocolm {

class ProbComputer {
 public:
  ProbComputer(int argc,
               const char **argv):
      total_log_prob_(0.0),
      total_count_(0) {

    assert(argc >= 3);
    train_input_.open(argv[1], std::ios_base::binary|std::ios_base::in);
    if (train_input_.fail()) {
      std::cerr << "compute-probs: error opening '" << argv[1]
                << "' for reading\n";
      exit(1);
    }
    dev_input_.open(argv[2], std::ios_base::binary|std::ios_base::in);
    if (dev_input_.fail()) {
      std::cerr << "compute-probs: error opening '" << argv[2]
                << "' for reading\n";
      exit(1);
    }
    // we write the training-set derivatives separately for each n-gram order.
    num_train_deriv_outputs_ = argc - 3;
    if (num_train_deriv_outputs_ != 0) {
      train_deriv_outputs_ = new std::ofstream[num_train_deriv_outputs_];
    } else {
      train_deriv_outputs_ = NULL;
    }
    for (int32 i = 0; i < num_train_deriv_outputs_; i++) {
      train_deriv_outputs_[i].open(argv[i + 3],
                                   std::ios_base::binary|std::ios_base::out);
      if (train_deriv_outputs_[i].fail()) {
        std::cerr << "compute-probs: error opening '" << argv[i + 3]
                  << "' for writing\n";
      }
    }
    ProcessInput();
    ProduceOutput();
  }

  ~ProbComputer() {
    for (int32 i = 0; i < num_train_deriv_outputs_; i++) {
      train_deriv_outputs_[i].close();
      if (train_deriv_outputs_[i].fail()) {
        std::cerr << "Failed to close stream for writing train derivs "
            "(full disk?)\n";
      }
    }
  }
 private:

  void ProcessInput() {
    // make sure next_discounted_state_ buffers some input..
    ReadNextDiscountedState();

    while (dev_input_.peek(), !dev_input_.eof()) {
      dev_state_.Read(dev_input_);
      ProcessCurrentDevState();
    }
  }

  // this function first moves next_discounted_state_ (if it
  // contains something) to the appropriate position in discounted_state_,
  // and then tries to read the next 'next_discounted_state_' from
  // train_input_.
  void ReadNextDiscountedState() {
    if (!next_discounted_state_.counts.empty()) {
      // this will move the contents of next_discounted_state_ into the
      // appropriate position in the discounted_state_ array.
      size_t hist_size = next_discounted_state_.history.size();
      if (discounted_state_.size() <= hist_size)
        discounted_state_.resize(hist_size + 1);
      // the next 3 lines move next_discounted_state_
      // into the appropriate position in discounted_state_,
      // then clear next_discounted_state_.
      discounted_state_[hist_size].Swap(&next_discounted_state_);
      if (!next_discounted_state_.counts.empty()) {
        // if the thing we just 'swapped out' was nonempty, we need
        // to clear it, but first we must write any derivatives
        // (if we're writing derivatives).
        if (train_deriv_outputs_ != NULL) {
          assert(size_t(num_train_deriv_outputs_) > hist_size);
          next_discounted_state_.WriteDerivs(train_deriv_outputs_[hist_size]);
        }
        next_discounted_state_.history.clear();
        next_discounted_state_.counts.clear();
      }
    }
    train_input_.peek();
    if (!train_input_.eof())
      next_discounted_state_.Read(train_input_);
  }

  bool NextDiscountedStateValid() {
    return !next_discounted_state_.counts.empty();
  }

  // This function keeps reading input from train_input_ until
  // it reaches a LM state that is lexicographically later than
  // dev_state_.
  void BufferTrainInput() {
    while (NextDiscountedStateValid() &&
           next_discounted_state_.history <= dev_state_.history)
      ReadNextDiscountedState();
    // At this point we've either read all the training-data input, or we've
    // reached a history that is lexicographically later than
    // dev_state_.history-- implying, thanks to the sorting of both the training
    // and dev input, that any training history-states relevant to dev_state_ must
    // be present in discounted_state_.

    assert(discounted_state_.size() > 0 &&
           "compute-probs: read no training-data input.");
  }

  // returns the longest training-set history-length relevant to the current
  // dev_state_'s history, does some checking, and returns it.
  int32 LongestRelevantHistorySize() const {
    size_t dev_history_size = dev_state_.history.size(),
        h = dev_history_size;
    while (h > 0 &&
           (h >= discounted_state_.size() ||
            !std::equal(discounted_state_[h].history.begin(),
                        discounted_state_[h].history.end(),
                        dev_state_.history.begin())))
      h--;
    // do some checking.
    for (size_t i = 1; i <= h; i++) {
      assert(std::equal(discounted_state_[i].history.begin(),
                        discounted_state_[i].history.end(),
                        dev_state_.history.begin()));
    }
    return static_cast<int32>(h);
  }

  // This function computes the probability for word 'word' in a history of
  // length 'hist_size' equal to discounted_state_[hist_size].history, and
  // updates the total_log_prob_ and total_count_ stats appropriately.
  void ProcessWord(int32 hist_size, int32 word, int32 count_of_word) {
    // count_position is only needed for the backprop; for history lengths > 0,
    // it is used to cache the positions in the 'counts' vectors where we found the
    // word.
    std::vector<int32> count_position(hist_size, -1);

    // here, tot_prob will accumulate the total probability for this word in
    // this history, and cur_backoff_prob will be the probability assigned to
    // backoff.. as we back off from higher-order to lower-order states it
    // will get smaller and smaller.  Note, we do backoff 'with interpolation'
    // read "A Bit of Progress in Language Modeling" by Goodman to understand
    // what this means, it comes from Chen and Goodman's work.
    float cur_backoff_prob = 1.0;
    // tot_prob is the probability our model predicts for this word in this
    // history (a sum over all backoff orders, since the model is "with
    // interpolation".
    float tot_prob = 0.0;
    for (int32 h = hist_size; h >= 0; h--) {
      const FloatLmState &lm_state = discounted_state_[h];
      assert(lm_state.total != 0.0);
      if (h == 0) {
        // here we can actually make some assumptions about the counts- that
        // they start from kEosSymbol and have no gaps.  This is because
        // of how discount-counts-1gram works.  This saves us having to do
        // a logarithmic-time lookup.
        assert(word >= kEosSymbol &&
               static_cast<int32>(lm_state.counts.size()) > word - kEosSymbol &&
               lm_state.counts[word - kEosSymbol].first == word);
        double unigram_count = lm_state.counts[word - kEosSymbol].second,
            unigram_total = lm_state.total;
        tot_prob += cur_backoff_prob * unigram_count / unigram_total;
      } else {
        std::pair<int32, float> search_pair(word,
                                            0.0);
        std::vector<std::pair<int32, float> >::const_iterator
            iter = std::lower_bound(lm_state.counts.begin(),
                                    lm_state.counts.end(),
                                    search_pair);
        if (iter != lm_state.counts.end() &&
            iter->first == word) {
          // There is a discounted-count for this word.
          float this_count = iter->second;
          tot_prob += cur_backoff_prob * this_count / lm_state.total;
          count_position[h - 1] = iter - lm_state.counts.begin();
        }
        // update the backoff probability / backoff penalty.
        cur_backoff_prob *= lm_state.discount / lm_state.total;
      }
    }
    assert(tot_prob > 0.0);
    float log_prob = log(tot_prob);
    total_log_prob_ += log_prob * count_of_word;
    total_count_ += count_of_word;

    if (train_deriv_outputs_ == NULL)
      return;

    // The rest of this function is the "backwards computation".
    // dF/dtot_prob [where F is log(tot_prob)] is 1 / tot_prob.
    float tot_prob_deriv = count_of_word / tot_prob,
        cur_backoff_prob_deriv = 0.0;
    // go in the backwards direction.
    for (int32 h = 0; h <= hist_size; h++) {
      FloatLmStateDerivs &lm_state = discounted_state_[h];
      if (h == 0) {
        double unigram_count = lm_state.counts[word - kEosSymbol].second,
            unigram_total = lm_state.total;
        // the forwards computation did:
        // tot_prob += cur_backoff_prob * unigram_count / unigram_total;
        cur_backoff_prob_deriv += tot_prob_deriv * unigram_count / unigram_total;
        float unigram_count_deriv =
            tot_prob_deriv * cur_backoff_prob / unigram_total,
            unigram_total_deriv =
            -(tot_prob_deriv * cur_backoff_prob * unigram_count) /
            (unigram_total * unigram_total);
        lm_state.total_deriv += unigram_total_deriv;
        lm_state.count_derivs[word - kEosSymbol] += unigram_count_deriv;
      } else {
        // 'pos' is the position in the counts array, or -1 if there
        // was no explicit count for this word.
        int32 pos = count_position[h - 1];
        float total = lm_state.total, discount = lm_state.discount;

        // code we're backprop'ing through at this point is:
        // cur_backoff_prob *= discount / total;
        // Mentally, we rearrange it as follows to clarify that there
        // are really two different variables involved:
        // cur_backoff_prob = prev_backoff_prob * discount / total;
        // and then we can write the following as the backprop code:
        //
        // lm_state.discount_deriv += cur_backoff_prob_deriv * prev_backoff_prob / total
        // lm_state.total_deriv -= (cur_backoff_prob_deriv * prev_backoff_prob * discount) / (total * total);
        // prev_backoff_prob_deriv = cur_backoff_prob_deriv * discount / total;
        ///  .. now, note that prev_backoff_prob == cur_backoff_prob * total / discount., so
        //  the statement
        // lm_state.total_deriv -= (cur_backoff_prob_deriv * prev_backoff_prob * discount)  / (total * total);
        // can be simplified to:
        // lm_state.total_deriv -= (cur_backoff_prob_deriv * cur_backoff_prob)  / total;
        //
        lm_state.total_deriv -= (cur_backoff_prob_deriv * cur_backoff_prob) / total;
        // mentally we view the following statement as:
        //  prev_backoff_prob = cur_backoff_prob * total / discount
        // and we view instances of 'cur_backoff_prob' in the lines below as
        // really referring to 'prev_backoff_prob'.
        cur_backoff_prob *= total / discount;
        // view the next statement as
        // lm_state.discount_deriv += cur_backoff_prob_deriv * prev_backoff_prob / total
        lm_state.discount_deriv += cur_backoff_prob_deriv * cur_backoff_prob / total;
        // view the next statement as:
        // prev_backoff_prob_deriv = cur_backoff_prob_deriv * discount / total;
        cur_backoff_prob_deriv *= discount / total;

        if (pos != -1) {
          float this_count = lm_state.counts[pos].second;
          double &this_count_deriv = lm_state.count_derivs[pos];
          // forward code:
          // tot_prob += cur_backoff_prob * this_count / lm_state.total;
          lm_state.total_deriv -=
              (tot_prob_deriv * cur_backoff_prob * this_count) /
              (total * total);
          this_count_deriv += tot_prob_deriv * cur_backoff_prob / total;
          cur_backoff_prob_deriv += tot_prob_deriv * this_count / total;
        }
      }
    }
    assert(fabs(cur_backoff_prob - 1.0) < 0.001);
  }

  void ProcessCurrentDevState() {
    BufferTrainInput();
    int32 hist_size = LongestRelevantHistorySize();

    assert(dev_state_.counts.size() != 0);
    for (std::vector<std::pair<int32, int32> >::const_iterator iter =
             dev_state_.counts.begin(); iter != dev_state_.counts.end();
         ++iter) {
      int32 word = iter->first,
          count_of_word = iter->second;
      assert(word > 0 && word != kBosSymbol &&
             count_of_word > 0);
      ProcessWord(hist_size, word, count_of_word);
    }
  }

  void FlushBuffers() {
    if (train_deriv_outputs_ != NULL) {
      while (!train_input_.eof())
        ReadNextDiscountedState();
      int32 discounted_state_size = discounted_state_.size();
      for (int32 i = 0; i < discounted_state_size; i++) {
        if (!discounted_state_[i].counts.empty()) {
          assert(i < num_train_deriv_outputs_);
          discounted_state_[i].WriteDerivs(train_deriv_outputs_[i]);
          // make sure we never do the same if this is called twice.
          discounted_state_[i].counts.clear();
          discounted_state_[i].history.clear();
        }
      }
    }
  }

  void ProduceOutput() {
    FlushBuffers();
    std::cout << std::setprecision(10)
              << total_count_ << " " << total_log_prob_ << "\n";
    std::cerr << "compute-probs: average log-prob per word was "
              << (total_log_prob_ / total_count_)
              << " (perplexity = "
              << exp(-total_log_prob_ / total_count_) << ") over "
              << total_count_ << " words.\n";
  }

  // train_input_ is the source for reading float-counts into
  // next_discounted_state_ and eventually discounted_state_.
  std::ifstream train_input_;
  // dev_input_ is the source for reading int-counts into dev_state_.
  std::ifstream dev_input_;

  // If the final optional argument is supplied, the derivatives w.r.t. the
  // training float-counts are written to these streams (indexed by
  // history-length).
  int32 num_train_deriv_outputs_;
  std::ofstream *train_deriv_outputs_;

  // dev_state_ is the current set of counts whose probabilities we
  // want to compute.  Since in normal usage this will be dev data,
  // we call it 'dev_state'.
  IntLmState dev_state_;

  // discounted_state_, indexed by history-length (0 for unigram counts, 1 for
  // bigram, etc.), is the counts from the training data.  We store this as
  // class FlloatLmStateDerivs so that it has the capacity to store the
  // derivatives also, in case we were called with the extended usage that
  // requires the derivatives.
  std::vector<FloatLmStateDerivs> discounted_state_;

  // This is a temporary buffer for the next state to be transferred to the
  // appropriate position in the 'discounted_state_' array.
  FloatLmStateDerivs next_discounted_state_;

  double total_log_prob_;
  // total_count_ is the total count of the dev data.
  int64 total_count_;
};

}  // namespace pocolm

int main (int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "usage:\n"
              << "compute-probs <train-float-counts> <dev-int-counts> [<train-float-count-derivs-order1> .. <train-float-count-derivs-orderN>]\n"
              << "This program prints the total count of dev words, followed by\n"
              << "the total log-prob, to stdout on one line.\n"
              << "The <train-float-counts> are discounted float-counts from\n"
              << "training data, obtained by a sequence of steps involving\n"
              << "merging and discounting; and the <dev-int-counts> are\n"
              << "derived from get-int-counts (on dev data).\n"
              << "If the <train-float-count-derivs> argument is supplied, the\n"
              << "derivatives w.r.t. the float-counts are written to that file.\n";
    exit(1);
  }

  // everything happens in the constructor of class ProbComputer.
  pocolm::ProbComputer(argc, argv);

  return 0;
}


/*
 testing:

  ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 2 | sort | uniq -c | get-int-counts int.1gram int.2gram
  merge-counts int.2gram,1.0 > merged.2gram
  discount-counts 0.8 0.7 0.6 merged.2gram float.2gram 1gram
  discount-counts-1gram 20 <1gram >float.1gram
  echo 10 11 12 | get-text-counts 2 | sort | uniq -c | get-int-counts dev.int
  merge-float-counts float.2gram float.1gram > float.all
  compute-probs float.all dev.int float_derivs.1gram float_derivs.2gram
 # produces:
# 4 -10.0067209
# compute-probs: average log-prob per word was -2.50168 (perplexity = 12.203) over 4 words.

perl -e "print -10.0067209 + $(perturb-float-counts 1 float.1gram float_derivs.1gram float_perturbed.1gram) . \"\n\"; "
# -10.006247441
 compute-probs <(merge-float-counts float.2gram float_perturbed.1gram) dev.int /dev/null /dev/null
# 4 -10.00624859

### Now testing 2-gram derivatives.
perl -e "print -10.0067209 + $(perturb-float-counts 2 float.2gram float_derivs.2gram float_perturbed.2gram) . \"\n\"; "
# -10.007106366
  compute-probs <(merge-float-counts float_perturbed.2gram float.1gram) dev.int /dev/null /dev/null
# 4 -10.00710714


# extend the above to propagating backwards through discount-counts-1gram:

# after back-proping through the 1-gram discounting, check that the derivative w.r.t. just scaling
# the 1-gram counts by a constant is zero.
discount-counts-1gram-backward 1gram float.1gram float_derivs.1gram derivs.1gram

print-derivs 1gram derivs.1gram | awk '{for (n=1;n<=NF;n++) print $n;}' | \
  perl -ane 'if (m/\((.+)\),d=\((.+)\)/) { @A = split(",", $1); @B = split(",", $2); $tot = 0; for ($n = 0; $n < 4; $n++) { $tot += $A[$n] * $B[$n]; } print "$tot\n"; } ' | \
    awk '{x+=$1; } END{print x;}'
# again, it's small:
# -1.114e-06

# testing the propagation of 1-gram derivatives backward through discount-counts-1gram-backward:

perl -e "print -10.0067209 + $(perturb-counts 3 1gram derivs.1gram perturbed.1gram) . \"\n\"; "
# -10.00413492

  compute-probs <(discount-counts-1gram 20 <perturbed.1gram| merge-float-counts float.2gram /dev/stdin) dev.int /dev/null /dev/null
# 4 -10.00414324






*/
