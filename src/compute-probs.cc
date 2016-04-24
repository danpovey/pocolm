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
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <map>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"


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

    assert(argc == 3);
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
    ProcessInput();
    ProduceOutput();
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
      next_discounted_state_.history.clear();
      next_discounted_state_.counts.clear();
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
        const FloatLmState &train_counts = discounted_state_[h];
        assert(train_counts.total != 0.0);
        if (h == 0) {
          // here we can actually make some assumptions about the counts- that
          // they start from kEosSymbol and have no gaps.  This is because
          // of how discount-counts-1gram works.  This saves us having to do
          // a logarithmic-time lookup.
          assert(word >= kEosSymbol &&
                 train_counts.counts.size() >= word &&
                 train_counts.counts[word-kEosSymbol].first ==
                 word);
          double unigram_count = train_counts.counts[word-kEosSymbol].second,
              unigram_total = train_counts.total;
          tot_prob += cur_backoff_prob * unigram_count / unigram_total;
        } else {
          std::pair<int32, float> search_pair(word,
                                              0.0);
          std::vector<std::pair<int32, float> >::const_iterator
              iter = std::lower_bound(train_counts.counts.begin(),
                                      train_counts.counts.end(),
                                      search_pair);
          if (iter != train_counts.counts.end() &&
              iter->first == word) {
            // There is a discounted-count for this word.
            float this_count = iter->second;
            tot_prob += cur_backoff_prob * this_count / train_counts.total;
            cur_backoff_prob *= train_counts.discount / train_counts.total;
          }
        }
      }
      assert(tot_prob > 0.0);
      float log_prob = log(tot_prob);
      total_log_prob_ += log_prob * count_of_word;
      total_count_ += count_of_word;
    }
  }

  void ProduceOutput() {
    std::cout << total_count_ << " " << total_log_prob_ << "\n";
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

  // dev_state_ is the current set of counts whose probabilities we
  // want to compute.  Since in normal usage this will be dev data,
  // we call it 'dev_state'.
  IntLmState dev_state_;

  // discounted_state_, indexed by history-length (0 for unigram counts, 1 for
  // bigram, etc.), is the counts from the training data.
  std::vector<FloatLmState> discounted_state_;

  // This is a temporary buffer for the next state to be transferred to the
  // appropriate position in the 'discounted_state_' array.
  FloatLmState next_discounted_state_;

  double total_log_prob_;
  // total_count_ is the total count of the dev data.
  int64 total_count_;
};

}  // namespace pocolm

int main (int argc, const char **argv) {
  if (argc <= 1) {
    std::cerr << "usage:\n"
              << "compute-probs <train-float-counts> <dev-int-counts>\n"
              << "This program prints the total count of dev words, followed by\n"
              << "the total log-prob, to stdout on one line.\n"
              << "The <train-float-counts> are discounted float-counts from\n"
              << "training data, obtained by a sequence of steps involving\n"
              << "merging and discounting; and the <dev-int-counts> are\n"
              << "derived from get-int-counts (on dev data).\n";
    exit(1);
  }

  // everything happens in the constructor of class ProbComputer.
  pocolm::ProbComputer(argc, argv);

  return 0;
}


/*
 testing:

  ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 2 | sort | uniq -c | get-int-counts int.1gram int.2gram
  merge-counts int.2gram,1.0 | discount-counts 0.8 0.7 0.6 float.2gram 1gram
  discount-counts-1gram 20 <1gram >float.1gram
  echo 10 11 12 | get-text-counts 2 | sort | uniq -c | get-int-counts dev.int
  merge-float-counts float.2gram float.1gram > float.all
 compute-probs float.all dev.int

 compute-probs float.all dev.int
4 -9.55299
compute-probs: average log-prob per word was -2.38825 (perplexity = 10.8944) over 4 words.

rm dev.int float.?gram float.all int.?gram

*/
