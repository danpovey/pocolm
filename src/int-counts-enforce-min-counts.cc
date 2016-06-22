// int-counts-enforce-min-counts.cc

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
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "pocolm-types.h"
#include "lm-state.h"


/*
   This program enforces the min-counts of different n-gram orders, e.g. suppose
   you have a min-count for orders 1,2,3,4,5 of 1,1,2,2,2, this program
   enforces them.  Note: a min-count of 1 is the same as a min-count of zero,
   as we won't discount anything.

   The effect on the int-counts is that, say, a 5-gram that is below the
   min-count will be backed off to the 4-gram.  For orders lower than
   the highest order, we have to be a bit careful because it's possible
   that the explicit 4-gram count of, say "b c d -> e" could be below
   the min-count, but there still exist 5-grams like "a b c d -> e" that
   were not pruned, so we have to take those into account when deciding whether
   to prune or not.

   Something else that we have to deal with in this program is, there may
   be multiple data sources; and we want to prune away n-grams only if
   the total count, *summed over all the data-sources*, is below the
   specified min-count for that order.

   This program writes it output separately by order-- this is what
   we need anyway in the context in which we use it, and in any case if
   they were written out to the same stream they would be 'out of order'
   (because lower order counts would be delayed relative to higher order
   counts).

*/
namespace pocolm {

class IntCountMinEnforcer {
 public:
  IntCountMinEnforcer(int argc, const char **argv) {
    SetSizes(argc, argv);
    SetMinCounts(argc, argv);
    OpenInputs(argc, argv);
    OpenOutputs(argc, argv);
    InitMembers();
    ProcessData();
  }
  ~IntCountMinEnforcer() {
    for (int32 i = 0; i < (ngram_order_ - 1) * num_data_types_; i++) {
      outputs_[i].close();
      if (outputs_[i].fail()) {
        std::cerr << "int-counts-enforce-min-counts: error closing output "
                  << "(disk full?)";
      }
    }
    delete [] inputs_;
    delete [] outputs_;
  }
 private:
  void ProcessData() {
    for (int32 d = 0; d < num_data_types_; d++)
      ReadStream(d);
    while (!hist_to_data_types_.empty())
      ProcessNextHistoryState();
    while (history_.size() > 0)
      FlushCurrentHistory();
  }

  void InitMembers() {
    lm_states_.resize((ngram_order_ - 1) * num_data_types_);
    // Give the history members of lm_states_ the correct length,
    // which saves having to check their size is correct in the rest of
    // the program
    for (int32 data_type = 0; data_type < num_data_types_; data_type++) {
      for (int32 history_length = 1; history_length < ngram_order_;
           history_length++) {
        int32 index = (history_length - 1) * num_data_types_ + data_type;
        lm_states_[index].history.resize(history_length);
      }
    }
    weighted_total_counts_.resize(ngram_order_ - 1);
    pending_lm_states_.resize(num_data_types_);
  }

  inline void AddToMap(
      int32 word, float weighted_count,
      unordered_map<int32, float> *weighted_total_counts) {
    std::pair<int32, float> pair_to_insert(word, weighted_count);
    std::pair<unordered_map<int32, float>::iterator, bool> ans =
        weighted_total_counts->insert(pair_to_insert);
    if (!ans.second) {  // The pair was not inserted because it already existed.
      // Increase the existing value by 'weighted_count'.
      ans.first->second += weighted_count;
    }
  }

  // This function updates the weighted_total_counts_; it's called after we read
  // in each LM-state.
  inline void AddToWeightedTotalCounts(int32 history_length, int32 data_type) {
    int32 index = (history_length - 1) * num_data_types_ + data_type;

    const IntLmState &lm_state = lm_states_[index];
    float *inverse_min_count = &(inverse_min_counts_[index]);

    unordered_map<int32, float> *weighted_total_counts =
        &(weighted_total_counts_[0]);

    std::vector<std::pair<int32, int32> >::const_iterator
        iter = lm_state.counts.begin(),
        end = lm_state.counts.end();
    for (; iter != end; ++iter) {
      int32 word = iter->first, count = iter->second;
      float *this_inverse_min_count = inverse_min_count;
      // note: h >= 2 means go down as far as trigram (a history-state with two
      // words would be considered a trigram history-state).  For bigram and
      // lower there is no count pruning, so there no point collecting the
      // weighted-total-counts.
      for (int32 h = history_length; h >= 2;
           h--, this_inverse_min_count -= num_data_types_) {
        // note: the weighted_total_counts_ array is indexed
        // by history-length - 1.
        AddToMap(word, count * (*this_inverse_min_count),
                 weighted_total_counts + (h - 1));
      }
    }
  }

  void ProcessNextHistoryState() {
    std::vector<int32> hist = hist_to_data_types_.begin()->first,
        data_types = hist_to_data_types_.begin()->second;
    hist_to_data_types_.erase(hist_to_data_types_.begin());

    // Eventually, for efficiency, we might wan to handle the case where
    // just one source has an LM-state with this history, as a special case.

    FlushConflictingHistories(hist);
    history_ = hist;
    int32 history_length = hist.size();
    for (std::vector<int32>::const_iterator iter = data_types.begin();
         iter != data_types.end(); ++iter) {
      int32 data_type = *iter;
      int32 index = (history_length - 1) * num_data_types_ + data_type;
      assert(lm_states_[index].counts.empty());
      lm_states_[index].Swap(&pending_lm_states_[data_type]);
      AddToWeightedTotalCounts(history_length, data_type);
      // the next statement tries to read something into the now-empty
      // pending_lm_states_[i] (it will do so as long as there's something left
      // in the sream).
      ReadStream(data_type);
    }
  }

  // This function returns true if vec1 is a prefix of vec2,
  // i.e.
  static inline bool IsPrefixOf(const std::vector<int32> &vec1,
                                const std::vector<int32> &vec2) {
    return vec1.size() <= vec2.size() &&
        std::equal(vec1.begin(), vec1.end(), vec2.begin());
  }


  // This flushes any history-states whose histories are are *not* a postfix of
  // 'hist'.  [a post-fix in natural word order, but a prefix if you consider
  // how we actually store them].
  // This is done before setting 'history_' to 'hist'.
  void FlushConflictingHistories(const std::vector<int32> &hist) {
    while (history_.size() > hist.size())
      FlushCurrentHistory();
    while (!IsPrefixOf(history_, hist))
      FlushCurrentHistory();
  }

  // This function sorts counts and combines multiple entries with the
  // same word, into single entries.  It doesn't check for zeros.
  static void CombineSameWordCounts(std::vector<std::pair<int32, int32> > *counts) {
    std::sort(counts->begin(), counts->end());
    // now merge any identical counts and get rid of any zero counts.
    std::vector<std::pair<int32, int32> >::const_iterator
        src = counts->begin(), end = counts->end();
    std::vector<std::pair<int32, int32> >::iterator
        dest = counts->begin();
    while (src != end) {
      int32 cur_word = src->first, cur_count = src->second;
      ++src;
      while (src != end && src->first == cur_word) {
        cur_count += src->second;
        ++src;
      }
      // We don't need to check for zeros at this point.
      // if (cur_count != 0) {
      dest->first = cur_word;
      dest->second = cur_count;
      ++dest;
      // }
    }
    counts->resize(dest - counts->begin());
  }

  // This function removes zero counts (i.e. pairs (word, count) where count == 0).
  static void RemoveZerosFromCounts(
      std::vector<std::pair<int32, int32> > *counts) {
    std::vector<std::pair<int32, int32> >::const_iterator
        src = counts->begin(), end = counts->end();
    std::vector<std::pair<int32, int32> >::iterator
        dest = counts->begin();
    for (; src != end; ++src) {
      if (src->second != 0) {
        if (dest != src)
          *dest = *src;
        ++dest;
      }
    }
    counts->resize(dest - counts->begin());
  }



  /*
    This function writes out, and destroys any LM-states we have in lm_states_
    with nonzero counts, for histories of the current history length
    history_.size().   [It assumes that any LM-states of higher orders have
    already been flushed.]  It also clears the weighted_total_counts_ of this
    history length, and pops something off history_ to reduce the history length.
  */
  void FlushCurrentHistory() {
    int32 history_length = history_.size();
    assert(history_length > 0);
    int32 num_data_types = num_data_types_;
    for (int32 data_type = 0; data_type < num_data_types; data_type++) {
      FlushThisHistory(history_length, data_type);
    }
    weighted_total_counts_[history_length - 1].clear();
    history_.pop_back();
  }

  /*
    This function normalizes, then writes out if nonempty, the counts for this
    history-length and data-type. It assumes that the same function has been
    called for any higher-order histories of the same data-type.
  */
  void FlushThisHistory(int32 history_length, int32 data_type) {
    int32 index = (history_length - 1) * num_data_types_ + data_type;
    IntLmState &lm_state = lm_states_[index];
    if (!lm_state.counts.empty()) {
      if (history_length + 1 < ngram_order_)
        CombineSameWordCounts(&lm_state.counts);
      if (history_length >= 2)
        BackOffLmState(history_length, data_type);
      RemoveZerosFromCounts(&lm_state.counts);
      // Normalizing the counts may have made them empty, so check again.
      if (!lm_state.counts.empty()) {
        // Before writing we have to make sure that lm_state.history is correct.
        // Most of the time lm_state.history is not guaranteed canonically correct,
        // it's this->history_ (or prefixes thereof) that give the 'correct'
        // history for each LM-state.
        assert(history_length == history_.size());
        lm_state.history = history_;
        lm_state.Write(outputs_[index]);
        lm_state.counts.clear();
      }
      lm_state.discount = 0.0;
    }
  }

  /*
    This function checks, for each predicted-word in the LM-state for this
    history-length and data-type, whether the corresponding value in
    weighted_total_counts_ is >= 1.0 (in which case we'll keep the word), or
    less, in which we'll completely discount it to a lower order.
  */
  void BackOffLmState(int32 history_length, int32 data_type) {
    assert(history_length >= 2);
    int32 index = (history_length - 1) * num_data_types_ + data_type,
        backoff_index = (history_length - 2) * num_data_types_ + data_type;
    const unordered_map<int32, float> &weighted_total_counts =
        weighted_total_counts_[history_length - 1];
    IntLmState &lm_state = lm_states_[index],
        &backoff_lm_state = lm_states_[backoff_index];
    std::vector<std::pair<int32, int32> >::iterator
        iter = lm_state.counts.begin(),
        end = lm_state.counts.end();
    // The next line will round down; it's OK.
    int32 min_count = int32(min_counts_[index]);
    // discounted_count will be the total count that we discounted from this
    // state.
    int32 total_discounted_count = 0;
    for (; iter != end; ++iter) {
      int32 word = iter->first, count = iter->second;
      // The next if-statement is an optimization for speed; the
      // real check happens below, see weighted_total.
      if (count >= min_count)
        continue;
      unordered_map<int32, float>::const_iterator map_iter =
          weighted_total_counts.find(word);
      assert(map_iter != weighted_total_counts.end());
      float weighted_total = map_iter->second;
      // Mathematically we're doing if (weighted_total < 1) { prune... }, but we
      // need the cutoff to be slightly less than one in case of roundoff
      // (because we assume that if a number is almost exactly 1, then it is
      // probably mathematically 1).
      if (weighted_total < 0.999) {
        // Add this count to the backoff state (we'll sort and merge later).
        backoff_lm_state.counts.push_back(*iter);
        total_discounted_count += count;
        // and set this count to zero
        iter->second = 0;
      }
    }
    assert(lm_state.discount == 0);  // the LM-state should not already have
    // been discounted.
    lm_state.discount = total_discounted_count;
  }


  // Calling this function will attempt to read a new lm-state from source
  // stream i, and will update hist_to_data_types_ as appropriate.
  void ReadStream(int32 data_type) {
    assert(data_type < num_data_types_);
    inputs_[data_type].peek();
    if (inputs_[data_type].eof())
      return;
    pending_lm_states_[data_type].Read(inputs_[data_type]);
    const std::vector<int32> &history = pending_lm_states_[data_type].history;
    hist_to_data_types_[history].push_back(data_type);
  }


  void SetSizes(int argc, const char **argv) {
    char *endptr = NULL;
    ngram_order_ = strtol(argv[1], &endptr, 10);
    if (!(*endptr == '\0' && ngram_order_ >= 3)) {
      std::cerr << "int-counts-enforce-min-counts: bad ngram-order '"
                << argv[1] << "'\n";
      exit(1);
    }
    /*
      If the n-gram order is n, the expected num-args is:
      argc = 1 [for program name] + 1 [for ngram-order] +
      (n-2) [for min-counts starting with trigram] +
      num_data_types [for the various input files] +
      (n-1) * num_data_types [for the outputs].
      = n * (num_data_types + 1)
    */
    if (argc % ngram_order_ != 0) {
      std::cerr << "int-counts-enforce-min-counts: expected num-args "
                << "to be divisible by n-gram order = "
                << ngram_order_ << "\n";
      exit(1);
    }
    num_data_types_ = (argc / ngram_order_) - 1;
    if (num_data_types_ < 1) {
      std::cerr << "int-counts-enforce-min-counts: too few command-line "
                << "arguments\n";
      exit(1);
    }
  }

  static void ParseCommaSeparatedList(const char *min_count_str,
                                      std::vector<float> *list) {
    const char *orig_str = min_count_str;
    list->clear();
    while (*min_count_str != '\0') {
      char *endptr;
      float f = strtof(min_count_str, &endptr);
      if (endptr == min_count_str || f < 1.0) {
        std::cerr << "int-counts-enforce-min-counts: bad min-counts '"
                  << orig_str << "'\n";
        exit(1);
      }
      list->push_back(f);
      if (endptr[0] == ',' && endptr[1] != '\0')
        endptr++;
      min_count_str = endptr;
    }
  }

  void SetMinCounts(int argc, const char **argv) {
    // note: min-counts is indexed by
    // ((history-length - 1) * num_data_types) + data_type
    min_counts_.resize((ngram_order_ - 1) * num_data_types_, 1);
    inverse_min_counts_.resize(min_counts_.size());

    // note: the min-count for hist_length = 1 (i.e. for order 2) is always 1,
    // and won't actually be accessed.
    for (int32 hist_length = 2; hist_length < ngram_order_; hist_length++) {
      const char *this_min_count_str = argv[hist_length];
      char *endptr;
      // this_min_count_str will either be a single floating-point value
      // (although normally it will be an integer), meaning we have the same
      // min-count for all data-types (although we first add the count across
      // all data-types before testing whether to keep a word); or it will be a
      // comma- separated list of floating point numbers.
      if (strchr(this_min_count_str, ',') == NULL) {
        float this_min_count = strtof(this_min_count_str, &endptr);
        if (!(*endptr == '\0' && this_min_count >= 1)) {
          std::cerr << "int-counts-enforce-min-counts: bad min-count '"
                    << this_min_count_str << "'\n";
          exit(1);
        }
        // set all the min-counts to the same value.
        for (int32 data_type = 0; data_type < num_data_types_; data_type++) {
          int32 index = (hist_length - 1) * num_data_types_ + data_type;
          min_counts_[index] = this_min_count;
        }
      } else {
        std::vector<float> this_list;
        ParseCommaSeparatedList(this_min_count_str, &this_list);
        int32 this_size = this_list.size();
        if (this_size != num_data_types_) {
          std::cerr << "int-counts-enforce-min-counts: bad min-counts '"
                    << this_min_count_str << "' [num-data-types = "
                    << num_data_types_ << "]\n";
          exit(1);
        }
        for (int32 data_type = 0; data_type < num_data_types_; data_type++) {
          int32 index = (hist_length - 1) * num_data_types_ + data_type;
          min_counts_[index] = this_list[data_type];
        }
      }
    }

    // Now check the min-counts.
    for (int32 hist_length = 1; hist_length + 1 < ngram_order_; hist_length++) {
      for (int32 data_type = 0; data_type < num_data_types_; data_type++) {
        int32 next_hist_length = hist_length + 1,
            index = (hist_length - 1) * num_data_types_ + data_type,
            next_index = (next_hist_length - 1) * num_data_types_ + data_type;
        float this_min_count = min_counts_[index],
            next_min_count = min_counts_[next_index];
        if (this_min_count > next_min_count) {
          std::cerr << "int-counts-enforce-min-counts: min-counts must be "
                    << "non-decreasing, but " << this_min_count << " > "
                    << next_min_count << "\n";
          exit(1);
        }
      }
    }
    for (size_t i = 0; i < min_counts_.size(); i++)
      inverse_min_counts_[i] = 1.0 / min_counts_[i];
  }

  void OpenInputs(int argc, const char **argv){
    inputs_ = new std::ifstream[num_data_types_];
    for (int32 data_type = 0; data_type < num_data_types_; data_type++) {
      const char *this_input = argv[ngram_order_ + data_type];
      inputs_[data_type].open(this_input,
                              std::ios_base::in|std::ios_base::binary);
      if (inputs_[data_type].fail()) {
        std::cerr << "int-counts-enforce-min-counts: error opening input '"
                  << this_input << "'\n";
        exit(1);
      }
    }
  }

  void OpenOutputs(int argc, const char **argv){
    outputs_ = new std::ofstream[(ngram_order_ - 1) * num_data_types_];
    for (int32 history_length = 1;
         history_length < ngram_order_;
         history_length++) {
      for (int32 data_type = 0;
           data_type < num_data_types_;
           data_type++) {
        const char *this_output = argv[ngram_order_ + num_data_types_ +
                                       data_type * (ngram_order_ - 1) +
                                       (history_length - 1)];
        int32 index = (history_length - 1) * num_data_types_ + data_type;
        outputs_[index].open(this_output,
                             std::ios_base::out|std::ios_base::binary);
        if (outputs_[index].fail()) {
          std::cerr << "int-counts-enforce-min-counts: error opening output: '"
                    << this_output << "'\n";
          exit(1);
        }
      }
    }
  }

  // The LM-states for each history-length > 0 and each data-type, indexed by
  // ((history_length - 1) * num_data_types_) + data_type.  These are read in
  // from the streams in inputs_, but before writing them out to outputs_, we
  // discount the counts that are below the relevant min-counts, and add them to
  // the one-lower-order state.  The 'counts' data-members may not be properly
  // normalized (meaning: sorted, and free of zero elements), because when we
  // discount counts we just append them to the backoff state's 'counts' vector.
  // These lm-states may contain counts that were backed off from higher-order
  // states.
  // Note: some of the time the 'history' elements of these LM-states may not be
  // valid.  The history in 'history_' is always the canonical history, and
  // we'll set the history members in the LM-states before writing them out; see
  // the comment for 'history_.'
  std::vector<IntLmState> lm_states_;

  // The current history-state we're processing.  The counts and discounts in
  // the lm_states_ for history-lengths greater than this history's length must
  // be empty/zero, and the counts and discounts for history-lengths <= this
  // history's length must correspond to postfixes of this history (in the
  // natural word-order; or prefixes in the order we store them).
  // Note: it may be that the actual 'history' members of the lm_states_ may
  // differ from the prefixes of this vector; this vector is canonical.
  std::vector<int32> history_;

  // Indexed first by history-length - 1, and then by word-id, this
  // contains the weighted-total-count, which is a summation over all
  // data-data_types and over all history-lengths g >= h, of the input count for
  // this word times the inverse_min_counts for this (h, data-source).
  // If the weighted-total-count for a word is < 1, it means we can prune
  // it away.
  // weighted_total_counts_[0] (i.e. for history-length == 1) is not
  // ever accessed, because
  std::vector<unordered_map<int32, float> > weighted_total_counts_;

  int32 ngram_order_;  // highest ngram order of counts we'll process
  // (this program processes all counts at once).
  int32 num_data_types_;

  // Indexed by ((history-length - 1) * num_data_types) + data_type, this vector
  // contains the min-counts for each data-type and order.  Note: the min-count
  // for order 2 is hard-coded at 1 and will not be accessed (we don't support a
  // min-count for bigrams, to simplify other parts of the toolkit design).
  std::vector<float> min_counts_;

  // This vector contains the inverse of min_counts_.
  std::vector<float> inverse_min_counts_;

  // inputs, indexed by data-type.
  std::ifstream *inputs_;

  // outputs, indexed by ((history_length - 1) * num_data_types_) + data_type.
  // There is no output for history-length 0 (order 1) because we don't support
  // min-counts for order 2 and the input should have no n-grams for order 1
  // (the toolkit does not support estimating 1-grams, and when estimating
  // higher orders, there is no way for a 1-gram count to appear in the raw
  // stats).
  std::ofstream *outputs_;

  // This is a map from the history vector to the list of source indexes that
  // currently have an LM-state with that history-vector, that needs to be
  // processed.  The keys of this map are the set of history-vectors
  // that are currently sitting, waiting to be processed, in the vectors
  // int_lm_states_ and general_lm_states_; the number of such keys will
  // not exceed the number of inputs (== num_data_types_).
  std::map<std::vector<int32>, std::vector<int32> > hist_to_data_types_;

  // This vector, indexed by data-source, is the LM-states we've just
  // read in from the input and that we have not started processing;
  // the indexes of pending LM-states that are valid (i.e. are really
  // pending and not just junk data) will appear as the values of
  // the 'hist_to_data_types_' map.
  std::vector<IntLmState> pending_lm_states_;
};

} // namespace pocolm

int main (int argc, const char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: int-count-enforce-min-counts <ngram-order> <min-counts-order3> .. <min-counts-orderN> \\\n"
              << "   <input-int-counts1> ... <input-int-countsX> \\\n"
              << "   <output-int-counts1-order2> ... <output-int-counts1-orderN> ... \\\n"
              << "   <output-int-countsX-order2> ... <output-int-countsX-orderN>\n"
              << "We don't support min-counts for orders 2 and fewer; this simplifies other\n"
              << "aspects of the toolkit.\n"
              << "The min-counts may be integers (in which case the interpretation is\n"
              << "obvious, except that they apply to the sum of the counts over all the\n"
              << "data sources), or they may be comma-separated lists of integers or floating\n"
              << "point values, one per data-source.  Suppose, for a particular order, the\n"
              << "min-counts are m1, m2 and m3.  Then if the counts for a particular word in\n"
              << "a particular history are c1, c2 and c3, we completely discount it if\n"
              << " c1/m1 + c2/m2 + c3/m3 < 0.999.  This is the same as saying that we discount if\n"
              << " [total-count] < min-count if there is a single min-count, but allows you to\n"
              << "incorporate dataset-specific weighting factors if you want.\n"
              << "min-counts may not decrease from one order to the next.\n";

    exit(1);
  }

  // Everything happens in the constructor.
  pocolm::IntCountMinEnforcer enforcer(argc, argv);

  return 0;
}

/*

  ( for n in $(seq 4); do echo 11 12 13; done; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/stdout | print-int-counts

get-text-counts: processed 5 lines, with (on average) 5.2 words per line.
get-int-counts: processed 5 LM states, with 6 individual n-grams.
 [ 1 ]: 11->5
 [ 11 1 ]: 12->5
 [ 12 11 ]: 13->5
 [ 13 12 ]: 2->4 14->1
 [ 14 13 ]: 2->1
print-int-counts: printed 5 LM states, with 6 individual n-grams.


  ( for n in $(seq 4); do echo 11 12 13; done; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/stdout | int-counts-enforce-min-counts 3 2.0 /dev/stdin foo.{2,3}

merge-int-counts foo.{2,3} | print-int-counts

merge-int-counts: read 3 + 3 = 6 LM states.
 [ 1 ]: 11->5
 [ 11 1 ]: 12->5
 [ 12 11 ]: 13->5
 [ 13 ]: 14->1
 [ 13 12 ]: discount=1 2->4
 [ 14 ]: 2->1
print-int-counts: printed 6 LM states, with 6 individual n-grams.


 */
