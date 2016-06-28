// float-counts-to-float-stats.cc

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

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <math.h>
#include <numeric>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"

/*
  This program modifies float-counts to 'float-stats'.  float-stats are
  identical to float-counts in terms of C++ data-structures, but conceptually
  they are a little different in that they represent n-gram stats in a different
  way, more explicitly.  These are useful in computing relative entropies
  between differently pruned LMs, and in the E-M procedure that we use to update
  LMs after pruning, when we want to make them as close as possible to the
  original (un-pruned) LM.

  As an example, suppose we have a 3-gram LM.  Firstly, note that the float-stats
  will always be structurally the same as the float-counts from which they
  were generated, i.e. they contain the same number of states and counts.  This
  is important in the entropy computation.

  Consider an order-3 state (history-length = 2).
  The counts in this history-state represent counts of a particular 3-gram in
  an amount of data generated from the LM that's equal to our original training
  data (after weighting by whatever source-specific weights we used), and with
  the same distribution of history-states as our training data.
  So for instance, a count for "a b c" represents the expected count of the
  sequence "a b c" in the corpus.  We don't have to add any backoff weights to
  this (that is what this program, float-counts-to-float-stats, is for).

  The backoff count for an order-3 state represents the sum-total (expected)
  count of all sequences of the form "a b X" that are not explicitly listed
  by counts of the form "a b c".

  Now consider an order-2 state, say the state for the history "b".
  The count for "b c" equals the total of all counts of the form "X b c"
  that were not accounted for by counts of the form "a b c".
  The count for "b *" equals the total of all counts of the form "b X"
  that were not accounted for by counts of the form "a b c" or "b c".

  Now consider the order-1 state.  The count for "c" equals the total of all
  counts of the form "X Y c" that were not accounted for by counts of the form
  "a b c" or "b c".

  Note that some of the stats may be double counted.  A highest-order
  count like "a b c" will never be counted elsewhere, but supposing
  there is no explicit bigram for "b c", then the total probability for all
  sequences "X b c" where there is no explicit count of the form "a b c",
  will be included in both the backoff-count for history-state "b "(because it
  matches the pattern "b *"), and also in the unigram count for
  word "c" (because it matches the pattern "* c").  However, if we exclude
  the backoff counts, the other stats are all disjoint, i.e. the
  trigram stats like "a b c", the bigram stats like "b c" and the unigram
  stats like "c", these are all disjoint and they should add up to the count
  of training data (after applying corpus weights).

  The output goes to several files, one for each n-gram order.  This is
  because if we were to output them all as one file, the history
  states would get out of order (the lower-order history states would
  get delayed).  It's most convenient to output to several files, and
  then merge those files using merge-float-counts.
 */


namespace pocolm {

class FloatStatsGenerator {
 public:
  // usage is:
  // float-counts-to-float-stats <num-words> <order1-output> ... <orderN-output> < <input>
  // both input and outputs are of float-counts type.
  FloatStatsGenerator(int argc, const char **argv):
      order_(argc - 2), outputs_(NULL), lm_states_(order_),
      work_(order_), total_input_count_(0.0), total_output_count_(0.0) {
    assert(order_ >= 1);
    char *end;
    num_words_ = strtol(argv[1], &end, 10);
    if (num_words_ <= 3 || *end != '\0') {
      std::cerr << "float-counts-to-float-stats: expected num-words as 1st argument, "
                << "got '" << argv[1] << "'\n";
      exit(1);
    }
    word_to_position_map_.resize((num_words_ + 1) * (order_ - 1));
    OpenOutputs(argc, argv);
    ProcessInput();
  }
  ~FloatStatsGenerator() {
    if (total_input_count_ != 0.0) {
      // this helps verify that the code works.
      if (fabs(total_input_count_ - total_output_count_) >
          1.0e-04 * total_input_count_) {
        std::cerr << "warning: float-counts-to-float-stats: total input and output "
                  << "count disagree too much: " << total_input_count_
                  << " vs. " << total_output_count_ << "\n";
      }
    }
    for (int32 o = 0; o < order_; o++) {
      outputs_[o].close();
      if (outputs_[o].fail()) {
        std::cerr << "float-counts-to-float-stats: failed to close an output "
                  << "file.  Disk full?\n";
        exit(1);
      }
    }
    delete [] outputs_;
  }
  private:

  struct FloatLmStateWork {
    double backoff;
    std::vector<double> counts;

    void Init (const FloatLmState &src) {
      counts.resize(src.counts.size());
      backoff = src.discount;
      std::vector<std::pair<int32, float> >::const_iterator
          counts_iter = src.counts.begin(),
          counts_end = src.counts.end();
      std::vector<double>::iterator dest_iter = counts.begin();
      for (; counts_iter != counts_end; ++counts_iter, ++dest_iter)
        *dest_iter = counts_iter->second;
    }
  };

  void OpenOutputs(int argc, const char **argv) {
    outputs_ = new std::ofstream[order_];
    for (int32 i = 0; i < order_; i++) {
      outputs_[i].open(argv[i + 2], std::ios_base::out|std::ios_base::binary);
      if (outputs_[i].fail()) {
        std::cerr << "float-counts-to-float-stats: error opening output file '"
                  << argv[i + 2] << "' for writing.\n";
        exit(1);
      }
    }
  }

  void ProcessInput() {
    while (std::cin.peek(), !std::cin.eof()) {
      FloatLmState lm_state;
      lm_state.Read(std::cin);
      // we don't expect zero counts in these stats and we can't deal with them
      // so we floor them to a small value.
      FloorCounts(1.0e-20, &lm_state);
      int32 history_length = lm_state.history.size();
      assert(history_length < order_ && "float-counts-to-float-stats: the order "
             "of the input counts is more than expected given the number of "
             "command-line arguments.");

      FlushOutput(history_length);
      lm_state.Swap(&(lm_states_[history_length]));
      if (static_cast<int32>(history_length) < order_ - 1)
        PopulateMap(history_length);
      work_[history_length].Init(lm_states_[history_length]);
    }
    FlushOutput(0);
  }

  // we don't expect zero counts in these stats and we can't deal with them so
  // we floor them to a small value.
  static void FloorCounts(float floor, FloatLmState *lm_state) {
    float extra_count = 0.0;
    std::vector<std::pair<int32, float> >::iterator
        iter = lm_state->counts.begin(),
        end = lm_state->counts.end();
    for (; iter != end; ++iter) {
      if (iter->second < floor) {
        extra_count += floor - iter->second;
        iter->second = floor;
      }
    }
    lm_state->total += extra_count;
  }

  void PopulateMap(int32 hist_length) {
    int32 pos = 0, num_words = num_words_;
    assert(word_to_position_map_.size() ==
           static_cast<size_t>((num_words + 1) * (order_ - 1)));
    int32 *map_data = &(word_to_position_map_[0]),
        orderm1 = order_ - 1;
    std::vector<std::pair<int32, float> >::const_iterator
        iter = lm_states_[hist_length].counts.begin(),
        end = lm_states_[hist_length].counts.end();
    for (pos = 0; iter != end; ++iter, pos++) {
      int32 word = iter->first,
          index = word * orderm1 + hist_length;
      assert(word > 0 && word <= num_words);
      map_data[index] = pos;
    }
  }

  // This function does the processing of, and then writes out and destroys, the
  // LM-states of all history lengths >= this history-length.  This is called
  // prior to reading something in of this history length (to make space); and
  // right at the end.
  void FlushOutput(int32 history_length) {
    assert(history_length < order_);
    for (int32 h = order_ - 1; h >= history_length; h--) {
      if (!lm_states_[h].counts.empty()) {
        // the core computations happen in the function call below.
        DoProcessingForLmState(h);
        lm_states_[h].Write(outputs_[h]);
        total_input_count_ += lm_states_[h].total - lm_states_[h].discount;
        // after we make the following call, we treat this history-state
        // as being empty.
        lm_states_[h].counts.clear();
      }
    }
  }

  // This function copies the data from 'work' into 'lm_state' prior to writing
  // it out.
  void FinalizeLmState(int32 history_length) {
    FloatLmState &lm_state = lm_states_[history_length];
    const FloatLmStateWork &work = work_[history_length];
    assert(work.counts.size() == lm_state.counts.size());

    float old_total = lm_state.total;
    lm_state.total = work.backoff +
        std::accumulate(work.counts.begin(), work.counts.end(), 0.0);
    lm_state.discount = work.backoff;
    std::vector<std::pair<int32, float> >::iterator
        counts_iter = lm_state.counts.begin(),
        counts_end = lm_state.counts.end();
    std::vector<double>::const_iterator src_iter = work.counts.begin();
    for (; counts_iter != counts_end; ++counts_iter,++src_iter) {
      float src_count = *src_iter;
      if (src_count < 0.0) {
        // the next line makes sure that if there are any negative values, they
        // are very small in magnitude
        if (src_count > -1.0e-04 * old_total) {
          std::cerr << "float-counts-to-float-stats: warning: possible excessive "
                    << "roundoff: " << src_count << " vs " << old_total << "\n";
        }
        src_count = 0.0;
      }
      counts_iter->second = src_count;
    }
    total_output_count_ += lm_state.total - lm_state.discount;
  }

  void DoProcessingForLmState(int32 history_length) {
    CheckBackoffStatesExist(history_length);
    int32 order_minus_one = order_ - 1;
    FloatLmState &lm_state = lm_states_[history_length];
    FloatLmStateWork &work = work_[history_length];

    if (history_length > 0) {
      // backoff_counts is indexed by history-length; it's the part of the count
      // of this word in this history, that's due to lower-order states than the
      // current one.
      std::vector<float> backoff_counts;
      std::vector<std::pair<int32, float> >::iterator
          count_iter = lm_state.counts.begin(),
          count_end = lm_state.counts.end();
      std::vector<double>::iterator work_count_iter =
          work.counts.begin();
      // backoff_counts will be indexed by history-length;
      // it contains the parts of the expected-count of this word
      // that are due to backoff from each order.
      backoff_counts.resize(history_length);
      for (; count_iter < count_end; ++count_iter, ++work_count_iter) {
        int32 word = count_iter->first;

        // This 'proportion_remaining' deserves some explanation.  It is the
        // proportion of this count that has not already been accounted for by
        // higher-order n-grams than this one.  For instance, if this is the
        // count for "b c" (i.e. this is the state for "b"), and there was a
        // state "a b" that had a count for "c", but there was also a state "x
        // b" that didn't have a count for c (and there were no more states of
        // the form X b), then proportion_remaining is the total count of state
        // "a b" divided by the total counts of states "x b" + "a b".  It would
        // be zero if there was a "c" count in all higher-order states than
        // this.  However, if this LM has already been pruned, it could be that
        // there are other higher-order states that used to exist but that have
        // already been entirely pruned away so we might not be able to identify
        // them by name, but their existence could be inferred from the counts.
        // Note: count_iter->second won't be exactly zero; see FloorCounts().
        float proportion_remaining = *work_count_iter / count_iter->second;

        assert(proportion_remaining > -1.0e-3);
        if (proportion_remaining < 1.0e-05) {
          // save by time by skipping the rest of this block.
          continue;
        }

        // We don't actually need the count in count_iter->second.
        // it's already included in the count in 'work' from where
        // we called FloatLmStateWork::Init().

        // note: 'cur_backoff_weight' equals the current backoff-prob
        // (lm_state.discount / lm_state.total) multiplied by the current
        // state's prior probability (lm_state.total).
        // In order to avoid double-counting, we multiply it
        // by proportion_remaining (which is in the range [0.0, 1.0]).
        float cur_backoff_weight = lm_state.discount * proportion_remaining;

        for (int32 backoff_hlen = history_length - 1;
             backoff_hlen >= 0; backoff_hlen--) {
          FloatLmState &backoff_state = lm_states_[backoff_hlen];
          int32 backoff_pos = word_to_position_map_[word * order_minus_one +
                                                    backoff_hlen];
          assert(static_cast<size_t>(backoff_pos) <
                 backoff_state.counts.size() &&
                 backoff_state.counts[backoff_pos].first == word);
          float backoff_total = backoff_state.total,
              backoff_backoff = backoff_state.discount,
              backoff_count = backoff_state.counts[backoff_pos].second;
          // cur_backoff_count is the contribution to the total expected count
          // for this word in this history, from this level of backoff.
          float cur_backoff_count = cur_backoff_weight * backoff_count /
              backoff_total;
          backoff_counts[backoff_hlen] = cur_backoff_count;
          // Subtract this count from the count in the backoff state,
          // because it is now explicitly part of an N-gram.  The lower-order
          // counts are only supposed to represent the N-grams not already
          // accounted for in higher-order counts.
          work_[backoff_hlen].counts[backoff_pos] -= cur_backoff_count;
          // update the backoff weight for the next-lowest-order state.
          cur_backoff_weight *= backoff_backoff / backoff_total;
        }

        // in the next loop, cur_backoff_tot will contain the sum
        // of all counts from backoff orders <= the current order.
        double cur_backoff_tot = 0.0;
        for (int32 backoff_hlen = 0; backoff_hlen < history_length;
             backoff_hlen++) {
          cur_backoff_tot += backoff_counts[backoff_hlen];
          // For each order, the following line subtracts the total count from
          // all lower backoff orders, from the 'backoff' count in the 'work'.
          // This is saying that since these lower-order counts are already
          // accounted for in an explicit n-gram, we don't want to include
          // them in the backoff count.  This is just how float-stats are defined;
          // it's convenient when we use them.
          work_[backoff_hlen + 1].backoff -= cur_backoff_tot;
        }
        // at this point cur_backoff_tot contains the total count for this word
        // in this context that's due to backoff.  We add this to the count for
        // this word in the current order's 'work' space.  For example, in
        // float-stats, the highest-order n-gram counts also contain the counts
        // due to backoff with interpolation; this is where we add in the part
        // due to backoff with interpolation.
        *work_count_iter += cur_backoff_tot;
      }
    }
    FinalizeLmState(history_length);
  }

  // this function checks that the states in lm_states_[i] for
  // i < hist_length are for histories that are the backoff histories
  // of the state in lm_states_[hist_length].
  void CheckBackoffStatesExist(int32 hist_length) const {
    for (int32 i = 1; i < hist_length; i++) {
      assert(static_cast<int32>(lm_states_[i].history.size()) == i);
      assert(std::equal(lm_states_[i].history.begin(),
                        lm_states_[i].history.end(),
                        lm_states_[hist_length].history.begin()));
    }
  }

  int32 num_words_;
  int32 order_;
  std::ofstream *outputs_;

  // The input LM-states, indexed by history length.  Just before being
  // output and then destroyed, these are temporarily used to store
  // the stats from work_ that we're about to output.
  std::vector<FloatLmState> lm_states_;


  // The output LM-states, indexed by history length.  These contain modified
  // versions of the counts in the input LM-states.
  std::vector<FloatLmStateWork> work_;

  // This maps from word-index to the position in the 'counts' vectors of the LM
  // states.  It exists to help us do rapid lookup of counts in lower-order
  // states when we are doing the backoff computation.  For history-length 0 <=
  // hist_len < order - 1 and word 0 < word <= num_words_,
  // word_to_position_map_[(order - 1) * word + hist_len] contains the position
  // in lm_states_[hist_len].counts that this word exists.  Note: for words that
  // are not in that counts vector, the value is undefined.
  std::vector<int32> word_to_position_map_;

  // total_input_count_ and total_output_count_ are part of a sanity check.
  // They are the total-count of the input and outputs respectively, minus
  // backoff counts.  Each can be interpreted as the (weighted) amount of
  // counts of training data.  They should be the same.
  double total_input_count_;
  double total_output_count_;
};



}  // namespace pocolm


int main (int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: float-counts-float-stats <num-words> <order1-output> ... <orderN-output> < <input>\n"
              << "E.g. float-counts-to-float-stats 20000 stats.1 stats.2 stats.3 < float.all\n"
              << "The outputs is in the same binary format as float-counts, but has a\n"
              << "different interpretation; it is a way of representing the n-gram stats\n"
              << "of the model in a way that's convenient for computing cross-entropies and\n"
              << "for E-M.  Please see the code for details.\n";
    exit(1);
  }

  // everything gets called from the constructor.
  pocolm::FloatStatsGenerator generator(argc, argv);

  return 0;
}

/*
  There is a script to sanity-check this, see egs/simple/run.sh, and
  'local/test_float_counts.sh' in that directory.
 */

