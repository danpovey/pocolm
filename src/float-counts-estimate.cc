// float-counts-estimate.cc

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
  This program does one iteration of E-M in re-estimating float counts to more
  closely approximate another model; it's used after pruning a model, to get
  it closer to the baseline model.

  Please see the usage message for more information.
*/


namespace pocolm {

class FloatCountsEstimator {
 public:
  // usage is:
  // float-counts-estimate <num-words> <float-counts-input> <float-stats-input> <order1-counts-output> ... <orderN-counts-output>
  // both inputs and outputs are of float-counts type.
  FloatCountsEstimator(int argc, const char **argv):
      order_(argc - 4), outputs_(NULL),
      lm_states_(order_), work_(order_),
      total_count_(0.0), total_logprob_(0.0), auxf_impr_(order_, 0.0) {
    assert(order_ >= 1);
    char *end;
    num_words_ = strtol(argv[1], &end, 10);
    if (num_words_ <= 3 || *end != '\0') {
      std::cerr << "float-counts-estimate: expected num-words as 1st argument, "
                << "got '" << argv[1] << "'\n";
      exit(1);
    }
    word_to_position_map_.resize((num_words_ + 1) * (order_ - 1));
    OpenInputs(argc, argv);
    OpenOutputs(argc, argv);
    ProcessInput();
  }
  ~FloatCountsEstimator() {
    for (int32 o = 0; o < order_; o++) {
      outputs_[o].close();
      if (outputs_[o].fail()) {
        std::cerr << "float-counts-estimate: failed to close an output "
                  << "file.  Disk full?\n";
        exit(1);
      }
    }
    delete [] outputs_;
    // produce some output on stdout:
    std::cout << total_count_ << ' ' << total_logprob_ << ' ';
    for (int32 o = 0; o < order_; o++)
      std::cout << auxf_impr_[o] << ' ';
    std::cout << std::endl;

    std::cerr << "float-counts-estimate: logprob per word was "
              << (total_logprob_  / total_count_) << " over "
              << total_count_ << " words." << std::endl;
    std::cerr << "float-counts-estimate: auxiliary function improvement per "
              << "word was [ ";
    for (int32 o = 0; o < order_; o++) {
      std::cerr << (auxf_impr_[o] / total_count_);
      if (o + 1 < order_)
        std::cerr << " + ";
    }
    float total_auxf_impr = std::accumulate(auxf_impr_.begin(),
                                            auxf_impr_.end(), 0.0);
    std::cerr << " ] = " << (total_auxf_impr / total_count_) << std::endl;
  }

 private:

  struct FloatLmStateWork {
    std::vector<double> counts;
    double discount;

    // We initialize all counts to zero; they get
    // added to in the DoExpectation() function.
    void Init(const FloatLmState &src) {
      counts.clear();
      counts.resize(src.counts.size(), 0.0);
      discount = 0.0;
    }
  };

  void OpenInputs(int argc, const char **argv) {
    float_counts_input_.open(argv[2], std::ios_base::in|std::ios_base::binary);
    if (float_counts_input_.fail()) {
      std::cerr << "float-counts-estimate: error opening input file '"
                << argv[2] << "'\n";
      exit(1);
    }
    float_stats_input_.open(argv[3], std::ios_base::in|std::ios_base::binary);
    if (float_stats_input_.fail()) {
      std::cerr << "float-counts-estimate: error opening input file '"
                << argv[3] << "'\n";
      exit(1);
    }
  }

  void OpenOutputs(int argc, const char **argv) {
    outputs_ = new std::ofstream[order_];
    for (int32 i = 0; i < order_; i++) {
      outputs_[i].open(argv[i + 4], std::ios_base::out|std::ios_base::binary);
      if (outputs_[i].fail()) {
        std::cerr << "float-counts-estimate: error opening output file '"
                  << argv[i + 4] << "' for writing.\n";
        exit(1);
      }
    }
  }

  void ProcessInput() {
    while (float_counts_input_.peek(), !float_counts_input_.eof()) {
      FloatLmState lm_state;
      lm_state.Read(float_counts_input_);
      int32 history_length = lm_state.history.size();
      assert(history_length < order_ && "float-counts-estimate: the order "
             "of the input counts is more than expected given the number of "
             "command-line arguments.");
      FlushOutput(history_length);
      lm_state.Swap(&(lm_states_[history_length]));
      if (static_cast<int32>(history_length) < order_ - 1)
        PopulateMap(history_length);
      work_[history_length].Init(lm_states_[history_length]);

      // Re-use the same object to read the float-stats.
      FloatLmState &lm_stats(lm_state);
      lm_stats.Read(float_stats_input_);
      DoExpectation(lm_stats);
    }
    FlushOutput(0);
    float_stats_input_.peek();
    if (!float_stats_input_.eof()) {
      std::cerr << "float-counts-estimate: <float-stats> has more input than "
                << "<float-counts>.  Mismatch?\n";
      exit(1);
    }
  }

  // This does the 'expectation' phase of E-M.
  void DoExpectation(const FloatLmState &stats) {
    int32 history_length = stats.history.size();
    CheckBackoffStatesExist(history_length);
    assert(history_length < order_ && "Bad float-stats input "
           "to float-counts-estimate: order is higher than float-counts.");
    FloatLmState &this_lm_state = lm_states_[history_length];
    FloatLmStateWork &this_work = work_[history_length];
    int32 order_minus_one = order_ - 1;

    if (stats.history != this_lm_state.history) {
      std::cerr << "float-counts-estimate: mismatch in float-counts and "
                << "float-stats inputs (history differs)\n";
      exit(1);
    }

    // Note: to get the total count of all words in the stats we need to add up
    // all the counts in e.g. 'stats.counts', but not the 'discount' numbers,
    // because that would lead to double-counting.  'total' contains the sum of
    // the counts in 'counts' plus 'discount', so we need to subtract
    // 'discount'.
    total_count_ += stats.total - stats.discount;
    // this_total_logprob_ will be the contribution to total_logprob_ that
    // arises from processing this state's worth of stats.
    double this_total_logprob = 0.0;
    if (stats.discount != 0.0) {
      // suppose this is the history-state for "a b *", stats.discount is the
      // total counts of all "a b X" such that there is no explicit n-gram like
      // "a b c".  All of these back off to the lower-order n-gram state
      // (e.g. the state for "b *"), and we need to store stats and update the
      // logprob to reflect this.  This part of the logprob is for counts that
      // will separately be dealt with in the history-states for "b" and "", but
      // when we handle those counts at that time, we won't know the
      // highest-order state from which they came, so this is where we handle
      // the E-M stats for the backoff probabilities.
      this_work.discount += stats.discount;

      this_total_logprob += stats.discount * log(this_lm_state.discount /
                                                 this_lm_state.total);
    }

    std::vector<std::pair<int32, float> >::const_iterator
        stats_counts_iter = stats.counts.begin(),
        stats_counts_end = stats.counts.end();
    std::vector<std::pair<int32, float> >::const_iterator
        lm_counts_iter = this_lm_state.counts.begin();
    std::vector<double>::iterator work_counts_iter =
        this_work.counts.begin();
    float lm_total = this_lm_state.total,
        lm_discount = this_lm_state.discount;
    // for each word we process, 'backoff_probs' will contain all the
    // backoff-derived terms in the probability, indexed by history length of
    // the backoff state
    std::vector<float> backoff_probs(history_length);
    for (; stats_counts_iter != stats_counts_end;
         ++stats_counts_iter,++lm_counts_iter,++work_counts_iter) {
      int32 word = stats_counts_iter->first;
      float stats_count = stats_counts_iter->second;
      float lm_count = lm_counts_iter->second;
      float direct_prob = lm_count / lm_total,
          tot_prob = direct_prob;
      // now we'll add all backoff orders to tot_prob.
      float cur_backoff_prob = lm_discount / lm_total;
      for (int32 backoff_hlen = history_length - 1;
           backoff_hlen >= 0; backoff_hlen--) {
        const FloatLmState &backoff_state = lm_states_[backoff_hlen];
        int32 backoff_pos = word_to_position_map_[word * order_minus_one +
                                                  backoff_hlen];
        assert(static_cast<size_t>(backoff_pos) <
               backoff_state.counts.size() &&
               backoff_state.counts[backoff_pos].first == word);
        float backoff_total = backoff_state.total,
            backoff_backoff = backoff_state.discount,
            backoff_count = backoff_state.counts[backoff_pos].second;
        float this_backoff_prob = cur_backoff_prob * backoff_count /
            backoff_total;
        backoff_probs[backoff_hlen] = this_backoff_prob;
        tot_prob += this_backoff_prob;
        cur_backoff_prob *= backoff_backoff / backoff_total;
      }
      // at this point, 'tot_prob' is the probability of this word in this
      // context.  [if this is not a highest-order LM state, it does not include
      // the discount probability from higher-order states, but that would have
      // been added to the total-logprob while processing higher-order states.
      this_total_logprob += stats_count * log(tot_prob);
      // add something to the count for the current n-gram... we add only
      // the proportion of the count that's due to the direct probability
      // excluding backoff.
      *work_counts_iter += stats_count * direct_prob / tot_prob;

      // in the next loop, cur_backoff_tot will contain the sum of all terms in
      // of 'tot_prob' that arise from backoff orders <= the current order.
      float cur_backoff_tot = 0.0;
      for (int32 backoff_hlen = 0; backoff_hlen < history_length;
           backoff_hlen++) {
        float this_backoff_prob = backoff_probs[backoff_hlen];
        cur_backoff_tot += this_backoff_prob;
        int32 backoff_pos = word_to_position_map_[word * order_minus_one +
                                                  backoff_hlen];
        // update the stats for this word's count in this backoff state.
        work_[backoff_hlen].counts[backoff_pos] +=
            stats_count * this_backoff_prob / tot_prob;
        // Update the stats for the discount-count in the one-more-specific
        // state than this one... all orders of backoff-prob <= the current
        // one are included in that weight since they all require backoff
        // from that state.
        work_[backoff_hlen + 1].discount +=
            stats_count * cur_backoff_tot / tot_prob;
      }
    }
    total_logprob_ += this_total_logprob;
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

  // This function does the "maximization" phase of, and then writes out and
  // destroys, the LM-states of all history lengths >= this history-length.
  // This is called prior to reading something in of this history length (to
  // make space); and right at the end.
  void FlushOutput(int32 history_length) {
    assert(history_length < order_);
    for (int32 h = order_ - 1; h >= history_length; h--) {
      if (!lm_states_[h].counts.empty()) {
        DoMaximizationForLmState(h);
        lm_states_[h].Write(outputs_[h]);
        // after we make the following call, we treat this history-state
        // as being empty.
        lm_states_[h].counts.clear();
      }
    }
  }

  // This function does the 'maximization' phase of E-M based on the stats
  // accumulated in 'work', and updates 'lm_state' accordingly; after this
  // call, 'lm_state' will be written out.
  // This function also accumulates the auxiliary-function improvement from
  // this maximization.
  void DoMaximizationForLmState(int32 history_length) {
    FloatLmState &lm_state = lm_states_[history_length];
    const FloatLmStateWork &work = work_[history_length];
    assert(work.counts.size() == lm_state.counts.size());

    float old_total = lm_state.total,
        work_total = work.discount +
        std::accumulate(work.counts.begin(), work.counts.end(), 0.0);
    if (old_total != 0.0 && work_total == 0.0) {
      std::cerr << "float-counts-estimate: accumulated zero stats "
                   "[unexpected]";
      exit(1);
    }
    if (old_total == 0.0) {
      std::cerr << "float-counts-estimate: had zero stats "
                   "in LM state [unexpected]";
      exit(1);
    }
    // this_auxf_impr will be added to the appopriate entry in
    // the objf_impr_ array.
    double this_auxf_impr = 0.0;
    if (work.discount != 0.0) {
      // first deal with the backoff/discount count.
      float old_backoff_prob = lm_state.discount / old_total,
          new_backoff_prob = work.discount / work_total;
      this_auxf_impr += work.discount * log(new_backoff_prob / old_backoff_prob);
      assert(this_auxf_impr - this_auxf_impr == 0.0); // check for NaN.
    }

    lm_state.total = work_total;
    lm_state.discount = work.discount;
    std::vector<std::pair<int32, float> >::iterator
        counts_iter = lm_state.counts.begin(),
        counts_end = lm_state.counts.end();
    std::vector<double>::const_iterator work_counts_iter = work.counts.begin();
    for (; counts_iter != counts_end; ++counts_iter,++work_counts_iter) {
      float work_count = *work_counts_iter,
          old_prob = counts_iter->second / old_total,
          new_prob = work_count / work_total;
      if (new_prob != 0.0) {
        this_auxf_impr += work_count * log(new_prob / old_prob);
        assert(this_auxf_impr - this_auxf_impr == 0.0); // check for NaN.
      }
      counts_iter->second = work_count;
    }
    auxf_impr_[history_length] += this_auxf_impr;
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

  std::ifstream float_counts_input_;
  std::ifstream float_stats_input_;

  // The input LM-states (from <float-counts-input>, indexed by history length.
  // Just before being output and then destroyed, these are temporarily used to
  // store the stats from work_ that we're about to output.
  std::vector<FloatLmState> lm_states_;

  // The output LM-states, indexed by history length.  These contain stats derived
  // from E-M which eventually will be written back to the LM-states and then
  // written out.
  std::vector<FloatLmStateWork> work_;

  // This maps from word-index to the position in the 'counts' vectors of the LM
  // states.  It exists to help us do rapid lookup of counts in lower-order
  // states when we are doing the backoff computation.  For history-length 0 <=
  // hist_len < order - 1 and word 0 < word <= num_words_,
  // word_to_position_map_[(order - 1) * word + hist_len] contains the position
  // in lm_states_[hist_len].counts that this word exists.  Note: for words that
  // are not in that counts vector, the value is undefined.
  std::vector<int32> word_to_position_map_;

  // total_count_ is the total number of words in the training data, obtained
  //  from the float-stats-input.
  double total_count_;

  // total_logprob_ is the total logprob of all the data; it should
  // be divided by total_count_ to get the logprob per word.
  double total_logprob_;

  // auxf_impr_, indexed by history-length, gives us the auxiliary function
  // improvement for each n-gram order starting with unigrams.  This
  // is worked out when we do E-M; it's the improvement in likelihood (and
  // when summed over all orders, it's a lower bound on the improvement
  // in total_logprob_ that we expect if we were to run this program again
  // using the output of this program as the <float-counts-input>.
  // This should be normalized by dividing by total_count_.
  std::vector<double> auxf_impr_;
};



}  // namespace pocolm


int main (int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: float-counts-estimate <num-words> <float-counts-input> <float-stats-input> <order1-output> ... <orderN-output>\n"
              << "E.g. float-counts-estimate 20000 float.all float_stats.all float.1 float.2 float.3\n"
              << "This can be viewed as a single iteration of E-M (for use after pruning).\n"
              << "To the standard output, this program prints:\n"
              << "<total-count> <total-logprob> <total-logprob-change-order1> .. <total-logprob-change-orderN>,\n"
              << "so the cross-entropy would be <total-logprob>/<total-count>, and\n"
              << "the change in log-prob due to this iteration of E-M is given by\n"
              << "(sum of <total-logprob-change-*>)/<total-count>.\n"
              << "<float-counts-input> will typically be the model (e.g. float.all) after\n"
              << "pruning, and <float-stats-input> will be the result of running\n"
              << "float-counts-to-float-stats on the un-pruned model (and then merging the\n"
              << "orders.\n"
              << "The different orders of output will typically be merged together with\n"
              << "merge-float-counts.\n";
    exit(1);
  }

  // everything gets called from the constructor.
  pocolm::FloatCountsEstimator generator(argc, argv);

  return 0;
}

