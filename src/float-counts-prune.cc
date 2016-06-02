// float-counts-prune.cc

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
  This program prunes away counts from a 'float-counts' representation
  of a language model using a divergence-based criterion... a threshold
  is supplied, and if the expected log-like of data generated from
  the original model and rescored using the new model, would decrease by
  less than the threshold, if we remove a single extra parameter, we
  remove that parameter and reassign the count to the lower-order state.
  Note: structurally the LM will be the same, it will just have more zero
  counts.  Later on, after re-estimating the parameters, we'll structurally
  remove the un-needed counts and states from the model.

*/


namespace pocolm {

class NullCountsReader {
 public:
  // Note: 'order' is actually the highest expected order of the history-states
  // in the null-counts, which is one less than the order of the model
  // we're dealing with.
  NullCountsReader(std::istream &input,
                   int32 order,
                   int32 num_words):
      input_(input), order_(order), num_words_(num_words),
      lm_states_(order), word_to_position_map_(order * (num_words + 1)) { }


  // This function uses the 'null-counts' to determine whether the n-gram of
  // 'word' in history-state 'history' is protected because there exists a
  // history-state with an idential word-sequence as this n-gram.  In order for
  // an ARPA file to be convertible to an FST, we need to make sure that n-grams
  // that coincide with history-states are not pruned, as we need a transition
  // to that history state; also, the ARPA format itself assumes this.
  // Note: for this function to work as intended, you have to call these in the
  // normal sorted order of history-states.
  bool NgramIsProtected(const std::vector<int32> &history,
                        int32 word) {
    if (history.empty()) {
      return true;  // all unigrams are protected.
    }
    int32 history_size = history.size();
    if (history_size >= order_) {
      return false;  // the highest-order N-grams are not protected.  note
                     // this->order_ is one less than the order of the language
                     // model we're pruning.
    }

    while ((lm_states_[history_size].predicted.empty() ||
            history > lm_states_[history_size].history) &&
           !input_.eof())
      ReadNext();

    // If the history-state in 'history_in' does not even exist in the protected
    // counts (e.g. this history-state is "a b" and there is no history-state of
    // the form "a b x"), return false.
    // Note: we can infer that it doesn't exist from the fact that
    // these history-states are being called in sorted order and the
    //
    if (lm_states_[history_size].history != history)
      return false;

    int32 pos = word_to_position_map_[word * order_ + history_size];
    return (pos < static_cast<int32>(lm_states_[history_size].predicted.size()) &&
            lm_states_[history_size].predicted[pos] == word);
  }
 private:
  void ReadNext() {
    input_.peek();
    if (input_.eof())
      return;
    NullLmState lm_state;
    lm_state.Read(input_);
    int32 history_length = lm_state.history.size();
    assert(history_length < order_);
    // The next line checks that the input is in sorted order.  To work at the
    // start of the file, relies on the fact that the empty vector always comes
    // first in the lexicographical ordering.
    assert(lm_states_[history_length].history <= lm_state.history);
    lm_states_[history_length].Swap(&lm_state);
    PopulateMap(history_length);
  }

  std::istream &input_;
  int32 order_;
  int32 num_words_;
  std::vector<NullLmState> lm_states_;
  // indexed: word * order_ + history-length.
  std::vector<int32> word_to_position_map_;

  void PopulateMap(int32 hist_length) {
    int32 pos = 0, num_words = num_words_, order = order_;
    assert(word_to_position_map_.size() ==
           static_cast<size_t>((num_words + 1) * order));
    int32 *map_data = &(word_to_position_map_[0]);
    std::vector<int32>::const_iterator
        iter = lm_states_[hist_length].predicted.begin(),
        end = lm_states_[hist_length].predicted.end();
    for (pos = 0; iter != end; ++iter, pos++) {
      int32 word = *iter,
          index = word * order + hist_length;
      assert(word > 0 && word <= num_words);
      map_data[index] = pos;
    }
  }
};


class FloatCountsPruner {
 public:
  // usage is:
  // float-counts-prune <threshold> <num-words> <float-counts-input> <protected-counts-input> <order1-counts-output> ... <orderN-counts-output>

  FloatCountsPruner(int argc, const char **argv):
      order_(argc - 5), outputs_(NULL),
      null_counts_reader_(NULL),
      lm_states_(order_), count_shadowed_(order_),
      total_count_(0.0), total_logprob_change_(0.0),
      num_ngrams_(0), num_ngrams_shadowed_(0), num_ngrams_protected_(0),
      num_ngrams_pruned_(0) {
    assert(order_ >= 1);
    SetThresholdAndNumWords(argv);
    word_to_position_map_.resize((num_words_ + 1) * (order_ - 1));
    OpenInputs(argc, argv);
    OpenOutputs(argc, argv);
    null_counts_reader_ = new NullCountsReader(protected_counts_input_,
                                               order_ - 1,
                                               num_words_);
    ProcessInput();
  }

  ~FloatCountsPruner() {
    delete null_counts_reader_;
    for (int32 o = 0; o < order_; o++) {
      outputs_[o].close();
      if (outputs_[o].fail()) {
        std::cerr << "float-counts-prune: failed to close an output "
                  << "file.  Disk full?\n";
        exit(1);
      }
    }
    delete [] outputs_;
    // produce some output on stdout:
    std::cout << total_count_ << ' ' << total_logprob_change_ << ' ' << '\n';

    std::cerr << "float-counts-prune: logprob change per word was "
              << (total_logprob_change_  / total_count_) << " over "
              << total_count_ << " words.\n";

    std::cout << num_ngrams_ << ' ' << num_ngrams_shadowed_ << ' '
              << num_ngrams_protected_ << ' ' << num_ngrams_pruned_ << '\n';

    std::cout << "float-counts-prune: aside from unigram there were "
              << num_ngrams_ << " nonzero n-grams.\n";
    int64 num_ngrams_eligible = num_ngrams_ - num_ngrams_shadowed_ -
        num_ngrams_protected_;

    std::cerr << "Of these " << num_ngrams_shadowed_ << " were not pruned "
              << "because they were shadowed by a higher-order n-gram, and\n"
              << num_ngrams_protected_ << " because they lead to an existing "
              << "LM-state (according to <protected-counts-input>).\n"
              << "Of the " << num_ngrams_eligible << " n-grams eligible for "
              << "pruning, " << num_ngrams_pruned_ << " were actually pruned.\n";
  }

 private:

  void SetThresholdAndNumWords(const char **argv) {
    char *end;
    threshold_ = strtod(argv[1], &end);
    if (*end != '\0' || end == argv[1] || threshold_ <= 0.0 ||
        threshold_ - threshold_ != 0.0) {
      std::cerr << "float-counts-prune: invalid threshold: '"
                << argv[1] << "'\n";
      exit(1);
    }

    num_words_ = strtol(argv[2], &end, 10);
    if (num_words_ <= 3 || *end != '\0') {
      std::cerr << "float-counts-prune: expected num-words as 2nd argument, "
                << "got '" << argv[1] << "'\n";
      exit(1);
    }
  }

  void OpenInputs(int argc, const char **argv) {
    float_counts_input_.open(argv[3], std::ios_base::in|std::ios_base::binary);
    if (float_counts_input_.fail()) {
      std::cerr << "float-counts-prune: error opening input file '"
                << argv[3] << "'\n";
      exit(1);
    }
    protected_counts_input_.open(argv[4],
                                 std::ios_base::in|std::ios_base::binary);
    if (protected_counts_input_.fail()) {
      std::cerr << "float-counts-prune: error opening input file '"
                << argv[4] << "'\n";
      exit(1);
    }
  }

  void OpenOutputs(int argc, const char **argv) {
    outputs_ = new std::ofstream[order_];
    for (int32 i = 0; i < order_; i++) {
      outputs_[i].open(argv[i + 5], std::ios_base::out|std::ios_base::binary);
      if (outputs_[i].fail()) {
        std::cerr << "float-counts-prune: error opening output file '"
                  << argv[i + 5] << "' for writing.\n";
        exit(1);
      }
    }
  }

  void ProcessInput() {
    while (float_counts_input_.peek(), !float_counts_input_.eof()) {
      FloatLmState lm_state;
      lm_state.Read(float_counts_input_);
      int32 history_length = lm_state.history.size();
      assert(history_length < order_ && "float-counts-prune: the order "
             "of the input counts is more than expected given the number of "
             "command-line arguments.");
      // the actual pruning is called from FlushOutput().
      FlushOutput(history_length);
      lm_state.Swap(&(lm_states_[history_length]));
      if (static_cast<int32>(history_length) < order_ - 1)
        PopulateMap(history_length);
      InitializeCountShadowed(history_length);
    }
    FlushOutput(0);
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

  void InitializeCountShadowed(int32 hist_length) {
    count_shadowed_[hist_length].clear();
    count_shadowed_[hist_length].resize(lm_states_[hist_length].counts.size(),
                                        false);
  }

  inline void check_divergence_params(double *c_a_h1, double *c_all_h1, double *c_bo_h1,
                                      double *c_a_hbo, double *c_all_hbo) {
    assert(*c_a_h1 > 0.0);
    assert(*c_all_h1 > 0.0);
    assert(*c_bo_h1 > 0.0);
    assert(*c_a_hbo >= 0.0);
    if (*c_a_hbo == 0.0) *c_a_hbo = 1.0e-20; // Avoids NaN's.
    assert(*c_all_hbo > 0.0);
    if (*c_all_h1 - *c_bo_h1 - *c_a_h1 < -0.0) {
      if (*c_all_h1 - *c_bo_h1 - *c_a_h1 < -0.05) {
        fprintf(stderr, "Remaining probability mass is <-0.05 in this state\n");
      }
      *c_all_h1 = *c_bo_h1 + *c_a_h1;
    }
    if (*c_all_hbo <= *c_a_hbo) { // Should be some mass left for backoff:
      // this is not right.
      fprintf(stderr, "No probability mass left in backoff state\n");
      exit(1);
    }
    if (*c_all_hbo < 0.98*(*c_bo_h1 - 0.2)) { // If this later causes problems, we
      // can make it a warning.
      fprintf(stderr, "Error: backoff mass in history-state is more than total mass "
              "in backoff state: %f < %f\n",
              *c_all_hbo, *c_bo_h1);
      exit(1);
    }
  }

  // This function computes the log-likelihood change (<= 0) from backing of
  // a particular symbol to the lower-order state.
  //
  //  'count' is the count of the word (suppose 'a') in this state
  //  'discount'is the discount-count in this state (discount/total == backoff prob).
  //  [note: we don't care about the total-count in this state.]
  //  'backoff_count' is the count of word 'a' in the lower-order state.
  //                 [actually its the augmented count, treating any
  //                  extra probability from even-lower-order states as
  //                  if it were a count].
  //  'backoff_total' is the total count in the lower-order state.
  float PruningLogprobChange(float count, float discount,
                             float backoff_count, float backoff_total) {

    // check that backoff_total > backoff_count and
    // that backoff_total >= discount [while allowing for roundoff].
    assert(count > 0.0 && discount > 0.0 &&
           backoff_total > backoff_count &&
           backoff_total >= 0.99 * discount);

    // augmented_count is like 'count', but with the extra count for symbol 'a'
    // due to backoff included.
    float augmented_count = count + discount * backoff_count / backoff_total;

    // We imagine a phantom symbol 'b' that represents all symbols other than
    // 'a' appearing in this history-state that are accessed via backoff.  We
    // treat these as being distinct symbols from the same symbol if accessed
    // not-via-backoff.  (Treating same symbols as distinct gives an upper bound
    // on the divergence).  We also treat them as distinct from the same symbols
    // that are being accessed via backoff from other states.  b_count is the
    // observed count of symbol 'b' in this state (the backed-off count is
    // zero).  b_count is also the count of symbol 'b' in the backoff state.
    float b_count = discount * ((backoff_total - backoff_count) / backoff_total);
    // b_count should not be negative.
    assert(b_count >= -0.001 * backoff_total);

    // We imagine a phantom symbol 'c' that represents all symbols other than
    // 'a' and 'b' appearing in the backoff state, which got there
    // from backing off other states (other than 'this' state).  Again, we
    // imagine the symbols are distinct even though they may not be, and this
    // gives us an upper bound on the divergence.
    float c_count = backoff_total - backoff_count - b_count;
    // c_count should not be negative.
    assert(c_count >= -0.001 * backoff_total);

    // backoff_other is the count of all symbols in the backoff state except 'a'.
    // it equals b_count + c_count.
    float backoff_other = backoff_total - backoff_count;
    // backoff_other should not be negative.
    assert(backoff_other >= -0.001 * backoff_total);

    // a_other is the count of 'a' in the backoff state that comes from
    // 'other sources', i.e. it was backed off from history-states other
    // than the current history state.
    float a_other_count = backoff_count - discount * backoff_count / backoff_total;
    // a_other_count should not be negative.
    assert(a_other_count >= -0.001 * backoff_count);

    // the following sub-expressions are the 'new' versions of certain
    // quantities after we assign the total count 'count' to backoff.
    // it increases the discount/backoff count in 'this' state, and also
    // the total count in the backoff state, and the count of symbol
    // 'a' in the backoff state.
    float new_backoff_count = backoff_count + count,  // new count of symbol 'a' in
                                                      // backoff state
        new_backoff_total = backoff_total + count,  // new total count in
                                                    // backoff state.
        new_discount = discount + count;  // new discount-count in 'this' state.


    /* all the loglike changes below are of the form
        count-of-symbol * log(new prob / old prob)
       which can be more conveniently written (by canceling the denominators),
        count-of-symbol * log(new count / old count). */

    // this_a_change is the log-like change of symbol 'a' coming from 'this'
    // state.  bear in mind that
    // augmented_count = count + discount * backoff_count / backoff_total,
    // and the 'count' term is zero in the numerator part of the log expression,
    // because symbol 'a' is completely backed off in 'this' state.
    float this_a_change = augmented_count *
        logf((new_discount * new_backoff_count / new_backoff_total) /
             augmented_count);

    // other_a_change is the log-like change of symbol 'a' coming from all other
    // states than 'this'.  For speed reasons we don't examine the direct
    // (non-backoff) counts of symbol 'a' in all other states than 'this' that
    // back off to the backoff state.  Instead we just treat the direct part
    // of the prob for symbol 'a' as a distinct symbol when it comes from those
    // other states... as usual, doing so gives us an upper bound on the
    // divergence.
    float other_a_change =
        a_other_count * logf((new_backoff_count / new_backoff_total) /
                             (backoff_count / backoff_total));

    // b_change is the log-like change of phantom symbol 'b' coming from
    // 'this' state (and note: it only comes from this state, that's how we
    // defined it).
    // note: the expression below could be more directly written as a
    // ratio of counts
    //  b_count * logf((new_discount * b_count / new_backoff_total) /
    //                 (discount * b_count / backoff_total),
    // but we cancel b_count to give us the expression below.
    float b_change = b_count * logf((new_discount / new_backoff_total) /
                                    (discount / backoff_total));

    // c_change is the log-like change of phantom symbol 'c' coming from
    // all other states that back off to the backoff sate (and all prob. mass of
    // 'c' comes from those other states).  The expression below could be more
    // directly written as a ratio of counts, as c_count * logf((c_count /
    // new_backoff_total) / (c_count / backoff_total)), but we simplified it to
    // the expression below.
    float c_change = c_count * logf(backoff_total / new_backoff_total);


    float ans = this_a_change + other_a_change + b_change + c_change;
    // ans should be negative.
    assert(ans < 0.0001 * count);
    return ans;
  }


  // This function does the pruning for, and then writes out and
  // destroys, the LM-states of all history lengths >= this history-length.
  // This is called prior to reading something in of this history length (to
  // make space); and also right at the end.
  void FlushOutput(int32 history_length) {
    assert(history_length < order_);
    for (int32 h = order_ - 1; h >= history_length; h--) {
      if (!lm_states_[h].counts.empty()) {
        DoPruningForLmState(h);
        UpdateCountShadowed(h);
        lm_states_[h].Write(outputs_[h]);
        // after we make the following call, we treat this history-state
        // as being empty.
        lm_states_[h].counts.clear();
      }
    }
  }

  // This function does the pruning of this LM state, and it assumes that the
  // pruning has already been done for all higher-order LM states.
  // Note: counts are discounted by setting them to zero and moving
  // the count to the backoff state.  Later on we'll structurally
  // remove the pruned counts.
  void DoPruningForLmState(int32 history_length) {
    if (history_length == 0)
      return;  // we don't prune the unigram state.
    CheckBackoffStatesExist(history_length);
    FloatLmState &lm_state = lm_states_[history_length],
        &backoff_state = lm_states_[history_length - 1];
    // update total_count_ before pruning... it would be the same before or
    // after, as long as we were consistent.
    total_count_ += lm_state.total - lm_state.discount;
    float threshold = threshold_;
    assert(count_shadowed_[history_length].size() == lm_state.counts.size());
    std::vector<std::pair<int32, float> >::iterator
        counts_iter = lm_state.counts.begin(),
        counts_end = lm_state.counts.end();
    std::vector<bool>::const_iterator
        shadowed_iter = count_shadowed_[history_length].begin();
    for (; counts_iter != counts_end; ++counts_iter,++shadowed_iter) {
      int32 word = counts_iter->first;
      float count = counts_iter->second;
      if (count == 0.0)
        continue;  // already pruned.
      num_ngrams_++;
      if (*shadowed_iter) {
        num_ngrams_shadowed_++;
        continue;  // We can't prune because there is a count for this word in a
                   // history state that backs off to this one.
      }
      if (null_counts_reader_->NgramIsProtected(lm_state.history, word)) {
        num_ngrams_protected_++;
        continue;  // We can't prune because there is a history-state with the
                   // same word-sequence as this n-gram (and there needs to be a
                   // path to get there); this is also a requirement to be able
                   // to format as ARPA.
      }
      float backoff_count = backoff_state.total * ProbForWord(word, history_length - 1);
      // likelike_change will be negative
      float logprob_change = PruningLogprobChange(count, lm_state.discount,
                                                  backoff_count, backoff_state.total);
      if (logprob_change > -threshold) {
        // get position of 'word' in the lower-order state.
        int32 pos = word_to_position_map_[word * (order_ - 1) +
                                          history_length - 1];
        counts_iter->second = 0.0;  // set this count to zero.
        lm_state.discount += count;  // assign it to backoff in this state..
        backoff_state.counts[pos].second += count;  // and move it to the
                                                    // backoff state
        backoff_state.total += count;  // update total of the backoff state.
        total_logprob_change_ += logprob_change;
        num_ngrams_pruned_++;
      }
    }
  }

  // This function checks which counts are still nonzero in the lm-state
  // in lm_states_[history_length], and for all nonzero ones, sets the
  // appropriate count_shadowed_ array element to true.
  void UpdateCountShadowed(int32 history_length) {
    if (history_length == 0)
      return;
    FloatLmState &lm_state = lm_states_[history_length];
    std::vector<std::pair<int32, float> >::iterator
        counts_iter = lm_state.counts.begin(),
        counts_end = lm_state.counts.end();
    for (; counts_iter != counts_end; ++counts_iter) {
      if (counts_iter->second != 0.0) {
        int32 word = counts_iter->first;
        // get position of 'word' in the lower-order state.
        int32 pos = word_to_position_map_[word * (order_ - 1) +
                                          history_length - 1];
        assert(lm_states_[history_length - 1].counts[pos].first == word);
        count_shadowed_[history_length][pos] = true;
      }
    }
  }

  // This function returns the probability of word 'word' in the
  // history-state of length 'hist_length'.  It is an error if this
  // history-state has no count for this word, or the count is zero.  The
  // probability includes backoff terms.
  float ProbForWord(int32 word, int32 hist_length) const {
    int32 pos = word_to_position_map_[word * (order_ - 1) + hist_length];
    const FloatLmState &lm_state = lm_states_[hist_length];
    assert(static_cast<size_t>(pos) < lm_state.counts.size() &&
           lm_state.counts[pos].first == word);
    float count = lm_state.counts[pos].second;
    assert(count > 0.0);
    if (hist_length > 0)
      count += lm_state.discount * ProbForWord(word, hist_length - 1);
    return count / lm_state.total;
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

  float threshold_;
  int32 num_words_;
  int32 order_;
  // outputs, one for each order of N-gram.
  std::ofstream *outputs_;

  std::ifstream float_counts_input_;
  std::ifstream protected_counts_input_;

  NullCountsReader *null_counts_reader_;

  // The input LM-states (from <float-counts-input>, indexed by history length.
  // Just before being output and then destroyed, these are temporarily used to
  // store the stats from work_ that we're about to output.
  std::vector<FloatLmState> lm_states_;

  // This vector, of size order_ - 1 tells us for each of the currently loaded
  // LM states (in lm_states_), and for each count, whether there is a
  // higher-order count for the same word that (after pruning) is nonzero.  If
  // this is the case, we say that a count is 'shadowed' and we won't consider
  // pruning it.  The reason we apply this rule (that a lower-order count can't
  // be pruned if a higher-order count exists), is for software compatibility.
  // We expect that quite a lot of software that deals with ARPA files makes
  // this assumption, even though it is not really necessary, so we enforce this
  // rule; we expect that the effect on performance will be quite small.  Note:
  // we also ask whether a count is 'protected' (via null_counts_reader_) and a
  // count being 'protected' (meaning: there is a history-state corresponding to
  // that count) also would disallow pruning.
  std::vector<std::vector<bool> > count_shadowed_;


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

  // total_logprob_chnage_ is the total change in logprob (of data generated
  // according to the distribution of the LM) due to pruning; divide it by
  // total_count_ to get the logprob-change per word.  This will be <= 0.
  double total_logprob_change_;


  // the total number of nonzero n-gram counts in the input LM, excluding
  // unigram
  int64 num_ngrams_;

  // the total number of nonzero n-gram counts in the input LM that could not be
  // considered for pruning because they were 'shadowed' (i.e. a higher-order
  // n-gram for the same word, with an extension of the same history existed).
  int64 num_ngrams_shadowed_;

  // the total number of nonzero n-gram counts in the input LM, that
  // could not be considered for pruning because they were 'protected'
  // (i.e. they are required because they lead to a state that hasn't
  // been completely pruned away).  this excludes shadowed n-grams.
  int64 num_ngrams_protected_;

  // the number of n-grams that were pruned away.
  int64 num_ngrams_pruned_;
};



}  // namespace pocolm


int main (int argc, const char **argv) {
  if (argc < 6) {
    std::cerr << "Usage: float-counts-prune <threshold> <num-words> <float-counts-input> <protected-counts-input> <order1-output> ... <orderN-output>\n"
              << "E.g. float-counts-prune 1.6 20000 float.all protected.all float.1 float.2 float.3\n"
              << "This program does entropy pruning of a language model.  Any count that is\n"
              << "not listed in <protected-counts-input> (which will probably be the output\n"
              << "of histories-to-null-counts) will be pruned if the data-weighted perplexity change\n"
              << "from backing off the count to its lower-order history state would be less than\n"
              << "the threshold.\n"
              << "The output is written separately per order, for later\n"
              << "merging.\n";
    exit(1);
  }

  // everything gets called from the constructor.
  pocolm::FloatCountsPruner pruner(argc, argv);

  return 0;
}

