// float-counts-stats-remove-zeros.cc

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
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"


namespace pocolm {

class ZeroRemover {
 public:
  ZeroRemover(int argc, const char **argv) {
    assert(argc >= 6);
    order_ = argc - 5;
    num_words_ = ConvertToInt(argv[1]);
    lm_stats_.resize(order_);
    lm_counts_nonzero_.resize(order_);
    num_ngrams_in_.resize(order_, 0);
    num_ngrams_out_.resize(order_, 0);
    word_to_position_map_.resize((num_words_ + 1) * (order_ - 1));
    OpenInputsAndOutputs(argv);
    ProcessInput();
  }

  ~ZeroRemover() {
    counts_out_.close();
    if (counts_out_.fail()) {
      std::cerr << "float-counts-stats-remove-zeros: error closing "
                    "counts file (disk full?)\n";
      exit(1);
    }
    for (int32 h = 0; h < order_; h++) {
      stats_out_[h].close();
      if (stats_out_[h].fail()) {
        std::cerr << "float-counts-stats-remove-zeros: error closing "
                     "stats file (disk full?)\n";
        exit(1);
      }
    }
    delete [] stats_out_;
    std::ostringstream ngrams_in_str, ngrams_out_str;
    int64 tot_in = 0, tot_out = 0;
    for (int32 o = 0; o < order_; o++) {
      ngrams_in_str << num_ngrams_in_[o] << ' ';
      tot_in += num_ngrams_in_[o];
      ngrams_out_str << num_ngrams_out_[o] << ' ';
      tot_out += num_ngrams_out_[o];
    }
    std::cerr << "float-counts-stats-remove-zeros: reduced counts from [ "
              << ngrams_in_str.str() << "] = " << tot_in << " to [ "
              << ngrams_out_str.str() << "] = " << tot_out << '\n';
  }
 private:

  void ProcessInput() {
    while (counts_in_.peek(), !counts_in_.eof()) {
      FloatLmState lm_state;
      lm_state.Read(counts_in_);
      int32 history_length = lm_state.history.size();
      assert(history_length < order_ && "float-counts-stats-remove-zeros: "
             "insufficient command line arguments for order of stats");
      FlushOutput(history_length);

      std::vector<bool> counts_nonzero;
      PruneCounts(&lm_state, &(lm_counts_nonzero_[history_length]));
      if (!lm_state.counts.empty())
        lm_state.Write(counts_out_);
      // Now read in the float-stats.
      lm_state.Read(stats_in_);
      assert(history_length == lm_state.history.size() &&
             "float-counts-stats-remove-zeros: mismatched stats?");
      // Most of the work happens inside FlushOutput().
      lm_stats_[history_length].Swap(&lm_state);
      PopulateMap(history_length);
    }
    FlushOutput(0);
    stats_in_.peek();
    assert(stats_in_.eof() &&
           "float-counts-stats-remove-zeros: more stats than counts.");
  }

  static void PruneCounts(FloatLmState *lm_state,
                          std::vector<bool> *counts_nonzero) {
    counts_nonzero->clear();
    counts_nonzero->resize(lm_state->counts.size(), false);
    std::vector<std::pair<int32, float> >::const_iterator
        counts_in_iter = lm_state->counts.begin(),
        counts_in_end = lm_state->counts.end();
    std::vector<std::pair<int32, float> >::iterator
        counts_out_iter = lm_state->counts.begin();
    std::vector<bool>::iterator nonzero_iter = counts_nonzero->begin();
    for (; counts_in_iter != counts_in_end;
         ++counts_in_iter,++nonzero_iter) {
      if (counts_in_iter->second != 0.0) {
        *nonzero_iter = true;
        *counts_out_iter = *counts_in_iter;
        ++counts_out_iter;
      }
    }
    size_t num_nonzero = counts_out_iter - lm_state->counts.begin();
    lm_state->counts.resize(num_nonzero);
  }

  void RestructureLmStats(int32 history_length) {
    if (history_length == 0)
      return;
    CheckBackoffStatesExist(history_length);
    FloatLmState &lm_stats = lm_stats_[history_length];
    std::vector<bool> &lm_counts_nonzero = lm_counts_nonzero_[history_length];
    FloatLmState &backoff_lm_stats = lm_stats_[history_length - 1];
    const int32 *word_to_position_map_data = &(word_to_position_map_[0]);
    int32 orderm1 = order_ - 1;
    double extra_discount = 0.0;

    assert(lm_stats.counts.size() == lm_counts_nonzero.size() &&
           "float-counts-stats-remove-zeros: mismatched stats and counts input");
    // We process the counts one by one.  We'll only keep the nonzero counts.
    // For each zero count that is removed, the corresponding element
    // of the stats is removed, but added to the lower-order LM-state.
    std::vector<std::pair<int32, float> >::const_iterator
        stats_in_iter = lm_stats.counts.begin(),
        stats_in_end = lm_stats.counts.end();
    std::vector<bool>::const_iterator counts_nonzero_iter =
        lm_counts_nonzero.begin();
    std::vector<std::pair<int32, float> >::iterator
        stats_out_iter = lm_stats.counts.begin();
    for (; stats_in_iter != stats_in_end;
         ++stats_in_iter,++counts_nonzero_iter) {
      if (*counts_nonzero_iter) {
        // nonzero count.
        *stats_out_iter = *stats_in_iter;
        ++stats_out_iter;
      } else {
        // count is pruned.  Add the stats to those of the lower-order LM state.
        int32 word = stats_in_iter->first;
        float stats_count = stats_in_iter->second;
        int32 backoff_pos = word_to_position_map_data[word * orderm1 +
                                                      history_length - 1];
        assert(backoff_pos < backoff_lm_stats.counts.size() &&
               backoff_lm_stats.counts[backoff_pos].first == word);
        backoff_lm_stats.counts[backoff_pos].second += stats_count;
        extra_discount += stats_count;
      }
    }
    size_t new_size = stats_out_iter - lm_stats.counts.begin();
    lm_stats.counts.resize(new_size);
    // the 'discount' term in the stats is defined to include all counts
    // that weren't explicitly accounted for by n-grams of this order...
    lm_stats.discount += extra_discount;
    backoff_lm_stats.total += extra_discount;
  }

  // this function checks that the states in lm_states_[i] for
  // i < hist_length are for histories that are the backoff histories
  // of the state in lm_states_[hist_length].
  void CheckBackoffStatesExist(int32 hist_length) const {
    for (int32 i = 1; i < hist_length; i++) {
      assert(static_cast<int32>(lm_stats_[i].history.size()) == i);
      assert(std::equal(lm_stats_[i].history.begin(),
                        lm_stats_[i].history.end(),
                        lm_stats_[hist_length].history.begin()));
    }
  }

  void FlushOutput(int32 history_length) {
    assert(history_length < order_ &&
           "float-counts-stats-remove-zeros: wrong order "
           "specified on command line");
    for (int32 h = order_ - 1; h >= history_length; h--) {
      if (!lm_stats_[h].counts.empty()) {
        num_ngrams_in_[h] += static_cast<int64>(lm_stats_[h].counts.size());
        RestructureLmStats(h);
        num_ngrams_out_[h] += static_cast<int64>(lm_stats_[h].counts.size());
        if (!lm_stats_[h].counts.empty()) {
          lm_stats_[h].Write(stats_out_[h]);
          lm_stats_[h].counts.clear();
        }
      }
    }
  }

  int32 ConvertToInt(const char *arg) {
    char *end;
    int32 ans = strtol(arg, &end, 10);
    if (end == arg || *end != '\0') {
      std::cerr << "float-counts-to-pre-arpa: command line: expected int, got '"
                << arg << "'\n";
      exit(1);
    }
    return ans;
  }

  void OpenInputsAndOutputs(const char **argv) {
    counts_in_.open(argv[2], std::ios_base::in|std::ios_base::binary);
    if (counts_in_.fail()) {
      std::cerr << "float-counts-stats-remove-zeros: error opening file '"
                << argv[2] << "' for input.\n";
      exit(1);
    }
    stats_in_.open(argv[3], std::ios_base::in|std::ios_base::binary);
    if (stats_in_.fail()) {
      std::cerr << "float-counts-stats-remove-zeros: error opening file '"
                << argv[3] << "' for input.\n";
      exit(1);
    }
    counts_out_.open(argv[4], std::ios_base::out|std::ios_base::binary);
    if (counts_out_.fail()) {
      std::cerr << "float-counts-stats-remove-zeros: error opening file '"
                << argv[4] << "' for output.\n";
      exit(1);
    }
    stats_out_ = new std::ofstream[order_];
    for (int32 h = 0; h < order_; h++) {
      stats_out_[h].open(argv[5 + h],
                         std::ios_base::out|std::ios_base::binary);
      if (stats_out_[h].fail()) {
        std::cerr << "float-counts-stats-remove-zeros: error opening file '"
                  << argv[5 + h] << "' for output.\n";
        exit(1);
      }
    }
  }

  void PopulateMap(int32 hist_length) {
    int32 pos = 0, num_words = num_words_;
    assert(word_to_position_map_.size() ==
           static_cast<size_t>((num_words + 1) * (order_ - 1)));
    int32 *map_data = &(word_to_position_map_[0]),
        orderm1 = order_ - 1;
    std::vector<std::pair<int32, float> >::const_iterator
        iter = lm_stats_[hist_length].counts.begin(),
        end = lm_stats_[hist_length].counts.end();
    for (pos = 0; iter != end; ++iter, pos++) {
      int32 word = iter->first,
          index = word * orderm1 + hist_length;
      assert(word > 0 && word <= num_words);
      map_data[index] = pos;
    }
  }

  std::ifstream counts_in_;
  std::ifstream stats_in_;
  std::ofstream counts_out_;
  // output streams for the stats, indexed by history-length.
  std::ofstream *stats_out_;

  int32 order_;

  int32 num_words_;

  // This is where we store the 'stats' (see float-counts-to-float-stats for
  // explanation of stats vs counts)
  std::vector<FloatLmState> lm_stats_;

  // For each order of the states stored in lm_stats_, vectors saying whether
  // each of its counts was nonzero in the input counts stats.
  std::vector<std::vector<bool> > lm_counts_nonzero_;


  // number of n-grams per order before pruning
  std::vector<int64> num_ngrams_in_;

  // number of n-grams per order after pruning
  std::vector<int64> num_ngrams_out_;

  // This maps from word-index to the position in the 'counts' vectors of the LM
  // states.  It exists to help us do rapid lookup of counts in lower-order
  // states when we are doing the backoff computation.  For history-length 0 <=
  // hist_len < order - 1 and word 0 < word <= num_words_,
  // word_to_position_map_[(order - 1) * word + hist_len] contains the position
  // in lm_states_[hist_len].counts that this word exists.  Note: for words that
  // are not in that counts vector, the value is undefined.
  std::vector<int32> word_to_position_map_;

};

}

int main (int argc, const char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: float-counts-stats-remove-zeros <num-words> "
              << "<float-counts-in> <float-stats-in> <float-counts-out> "
              << "<float-stats-out-order1> ... <float-stats-out-orderN>\n"
              << "This program copies float-counts while removing zero counts, and simultaneously\n"
              << "makes the same structural change to some float-stats.  For the\n"
              << "float-stats, this means adding the removed stats to their backoff states.\n";
    exit(1);
  }

  // everything happens in the constructor.
  pocolm::ZeroRemover zero_remover(argc, argv);
  return 0;
}


