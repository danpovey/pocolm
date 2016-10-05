// discount-counts-backward.cc

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
#include "lm-state-derivs.h"

/*
   This is the backprop program corresponding to the 'discount-counts' program,
   that computes the derivatives w.r.t. its inputs.
*/


namespace pocolm {

class CountDiscounterBackward {
 public:
  CountDiscounterBackward(int argc,
                          const char **argv):
      d1_deriv_(0.0), d2_deriv_(0.0), d3_deriv_(0.0), d4_deriv_(0.0),
      num_lm_states_processed_(0) {
    // see usage message for expected usage.
    assert(argc == 11);
    ReadArgs(argv);
    ProcessInput();
  }

  ~CountDiscounterBackward() {
    deriv_stream_.close();
    if (deriv_stream_.fail()) {
      std::cerr << "discount-counts-backward: failed to close output (disk full?)\n";
      exit(1);
    }
  }
 private:


  void ProcessInput() {

    while (count_stream_.peek(), !count_stream_.eof()) {
      GeneralLmStateDerivs input_lm_state;
      input_lm_state.Read(count_stream_);

      FloatLmStateDerivs discounted_lm_state;
      discounted_lm_state.Read(discounted_count_stream_);
      discounted_lm_state.ReadDerivs(discounted_deriv_stream_);

      if (backoff_lm_state_.history.size() + 1 !=
          input_lm_state.history.size() ||
          ! std::equal(backoff_lm_state_.history.begin(),
                       backoff_lm_state_.history.end(),
                       input_lm_state.history.begin()) ||
          backoff_lm_state_.counts.empty()) {
        CheckDerivsUsed(backoff_lm_state_);
        backoff_lm_state_.Read(backoff_count_stream_);
        backoff_lm_state_.ReadDerivs(backoff_deriv_stream_);
        UpdateWordMap();
      }
      ProcessLmState(discounted_lm_state,
                     &input_lm_state);
      input_lm_state.WriteDerivs(deriv_stream_);
    }
    CheckDerivsUsed(backoff_lm_state_);

    std::cerr << "discount-counts-backward: processed "
              << num_lm_states_processed_ << " LM states\n";
    std::cout << d1_deriv_ << " " << d2_deriv_ << " " << d3_deriv_
              << " " << d4_deriv_ << "\n";
  }


  // ensures that the top1, top2, top3 derivs in this GeneralLmStateDerivs class
  // instance are all zero, implying those derivs have been propagated
  // correctly.
  // note: it's OK for one the derivs to be nonzero if the corresponding value
  // is zero, since the derivatives around zero are not very well defined
  // and their propagation won't make a difference in the final application.
  void CheckDerivsUsed(const GeneralLmStateDerivs &state) const {
    std::vector<Count>::const_iterator
        iter = state.count_derivs.begin(),
        end = state.count_derivs.end();
    std::vector<std::pair<int32,Count> >::const_iterator pair_iter =
        state.counts.begin();
    for (; iter != end; ++iter, ++pair_iter) {
      assert((iter->top1 == 0.0 || pair_iter->second.top1 == 0.0) &&
             (iter->top2 == 0.0 || pair_iter->second.top2 == 0.0) &&
             (iter->top3 == 0.0 || pair_iter->second.top3 == 0.0) &&
             "some derivatives were not accounted for.");
    }
  }

  void UpdateWordMap() {
    for (size_t i = 0; i < backoff_lm_state_.counts.size(); i++) {
      int32 word = backoff_lm_state_.counts[i].first;
      assert(word > 0);
      // note, we leave the new indexes undefined, it doesn't matter.
      if (word_map_.size() <= static_cast<size_t>(word))
        word_map_.resize(static_cast<size_t>(word) + 1);
      word_map_[word] = i;
    }
  }
  /*
    This does the core of the backwards differentiation.  'discounted_lm_state'
    is the discounted version of this LM-state (i.e. with discount amounts removed),
    with derivatives.
    'lm_state' is the original LM-state before discounting (in the form of
    GeneralCounts), together with its derivatives; and we're backprop-ing through
    the discounting operation.  The output of this function is the derivatives
    stored in 'lm_state'.
    A 'hidden' input to this function is 'backoff_lm_state_', which is the
    lower-order version of this LM state, and which stores the derivatives
    w.r.t. the backed-off parts.
   */
  void ProcessLmState(const FloatLmStateDerivs &discounted_lm_state,
                      GeneralLmStateDerivs *lm_state) {
    assert(discounted_lm_state.counts.size() ==
           lm_state->counts.size());
    num_lm_states_processed_++;

    std::vector<std::pair<int32,Count> >::const_iterator count_iter =
        lm_state->counts.begin(), count_end = lm_state->counts.end();
    std::vector<double>::const_iterator discounted_deriv_iter =
        discounted_lm_state.count_derivs.begin();
    std::vector<Count>::iterator deriv_iter = lm_state->count_derivs.begin();
    double d1_deriv_part = 0.0,
        d2_deriv_part = 0.0,
        d3_deriv_part = 0.0,
        d4_deriv_part = 0.0;
    // The derivative of the objective function w.r.t. the 'total backoff count'
    // discounted_lm_state.discount_deriv.
    float total_backoff_count_deriv = discounted_lm_state.discount_deriv;
    // lm_state->discount_deriv is the derivative w.r.t. lm_state->discount,
    // which is nonzero only if we applied min-counts to the stats, and which
    // gets added into discounted_lm_state.discount.
    lm_state->discount_deriv = discounted_lm_state.discount_deriv;
    for (; count_iter != count_end;
         ++count_iter, ++discounted_deriv_iter, ++deriv_iter) {
      int32 word = count_iter->first;
      const Count &count = count_iter->second;
      // 'discounted_deriv' is the derivative of the objective function
      // w.r.t. the discounted count we're currently processing.
      float discounted_deriv = *discounted_deriv_iter;
      Count &deriv = *deriv_iter;
      assert(static_cast<size_t>(word) < word_map_.size() &&
             static_cast<size_t>(word_map_[word]) <
             backoff_lm_state_.counts.size() &&
             backoff_lm_state_.counts[word_map_[word]].first == word);

      int32 backoff_pos = word_map_[word];
      const Count &backoff_count = backoff_lm_state_.counts[backoff_pos].second;
      Count &backoff_deriv = backoff_lm_state_.count_derivs[backoff_pos];
      // mark these quantities volatile to avoid compiler optimization, so that
      // we can ensure they will be exactly the same value in discount-counts
      // and discount-counts-backward (since the backprop relies on exact
      // floating-point comparisons).
      volatile float top4plus = count.total - count.top1 - count.top2 - count.top3,
          d1 = d1_ * count.top1, d2 = d2_ * count.top2, d3 = d3_ * count.top3,
          d4 = d4_ * top4plus, d = d1 + d2 + d3 + d4;
      // d_deriv is the derivative of the objective function w.r.t. d...
      // this is because we set, in the forward pass,
      // discounted_count = count.total - d.
      // The following two statements are doing backprop through the
      // statements
      // "discounted_count = count.total - d".
      //  and "discount_total += d".
      float d_deriv = total_backoff_count_deriv - discounted_deriv;
      deriv.total = discounted_deriv;

      if (POCOLM_SEPARATE_COUNTS) {
        // 'discount' is the part that we discount and that gets added to the
        // backoff state.
        Count discount;
        discount.total = d;
        discount.top1 = d1;
        discount.top2 = d2;
        discount.top3 = d3;
        Count discount_deriv(0.0f);
        // In the forward pass we'd be doing 'backoff_count.Add(discount)'.
        backoff_count.AddBackward(discount, &backoff_deriv, &discount_deriv);

        // after getting discount_deriv we need to bear in mind that
        // discount.total = d1 + d2 + d3 + d4, so the derivative w.r.t.
        // discount.total (==discount_deriv.total + d_deriv) has to be
        // propagated bck to d1, d2, d3 and d4.
        float d1_deriv = discount_deriv.top1 + discount_deriv.total + d_deriv,
            d2_deriv = discount_deriv.top2 + discount_deriv.total + d_deriv,
            d3_deriv = discount_deriv.top3 + discount_deriv.total + d_deriv,
            d4_deriv = discount_deriv.total + d_deriv;

        // the following backprops through the statements
        // d1 = d1_ * count.top1 (etc.)
        d1_deriv_part += count.top1 * d1_deriv;
        d2_deriv_part += count.top2 * d2_deriv;
        d3_deriv_part += count.top3 * d3_deriv;
        d4_deriv_part += top4plus * d4_deriv;

        // note, 'deriv' is the derivative of the objf w.r.t. 'count'.
        float top4plus_deriv = d4_deriv * d4_;
        deriv.top1 = d1_deriv * d1_ - top4plus_deriv;
        deriv.top2 = d2_deriv * d2_ - top4plus_deriv;
        deriv.top3 = d3_deriv * d3_ - top4plus_deriv;
        deriv.total += top4plus_deriv;
      } else {
        // the forward code is just:
        // backoff_count.Add(d);

        // the next statement will add something to 'd_deriv'.
        backoff_count.AddBackward(d, &backoff_deriv, &d_deriv);

        // the following backprops through the statements
        // d1 = d1_ * count.top1 (etc.)

        d1_deriv_part += count.top1 * d_deriv;
        d2_deriv_part += count.top2 * d_deriv;
        d3_deriv_part += count.top3 * d_deriv;
        d4_deriv_part += top4plus * d_deriv;

        float top4plus_deriv = d_deriv * d4_;
        deriv.top1 = d_deriv * d1_ - top4plus_deriv;
        deriv.top2 = d_deriv * d2_ - top4plus_deriv;
        deriv.top3 = d_deriv * d3_ - top4plus_deriv;
        deriv.total += top4plus_deriv;
      }
    }
    d1_deriv_ += d1_deriv_part;
    d2_deriv_ += d2_deriv_part;
    d3_deriv_ += d3_deriv_part;
    d4_deriv_ += d4_deriv_part;
  }

  // this is the backoff LM-state and its derivatives, both read from disk.
  GeneralLmStateDerivs backoff_lm_state_;

  // this is a lookup table that allows us to quickly find the position of a
  // word in the 'counts' array of 'backoff_lm_state_'.  It maps from the word
  // index to the position in the 'counts' array.  Note: words currently present
  // in the 'counts' array of 'backoff_lm_state' are guaranteed to have an entry
  // in this map, but other words may also have entries, so you have to check
  // that an entry is correct before trusting that it is.
  std::vector<int32> word_map_;


  void ReadArgs(const char **argv) {
    d1_ = ConvertToFloat(argv[1]);
    d2_ = ConvertToFloat(argv[2]);
    d3_ = ConvertToFloat(argv[3]);
    d4_ = ConvertToFloat(argv[4]);
    assert(1.0 > d1_ && d1_ >= d2_ && d2_ >= d3_ && d3_ >= d4_ && d4_ >= 0);

    OpenStream(argv[5], &count_stream_);
    OpenStream(argv[6], &discounted_count_stream_);
    OpenStream(argv[7], &discounted_deriv_stream_);
    OpenStream(argv[8], &backoff_count_stream_);
    OpenStream(argv[9], &backoff_deriv_stream_);
    OpenStream(argv[10], &deriv_stream_);
  }

  float ConvertToFloat(const char *str) const {
    char *end;
    float ans = strtod(str, &end);
    if (!(*end == 0.0)) {
      std::cerr << "discount-counts: expected float, got '" << str << "'\n";
    }
    if (!(ans >= 0.0 and ans <= 1.0)) {
      std::cerr << "discount-counts-backward: discounting values must be "
                << ">=0.0 and <= 1.0: " << str << "\n";
      exit(1);
    }
    return ans;
  }

  void OpenStream(const char *filename,
                  std::ifstream *stream) {
    stream->open(filename, std::ios_base::binary|std::ios_base::in);
    if (!stream->is_open()) {
      std::cerr << "discount-counts-backward: failed to open '"
                << filename << "' for reading\n";
      exit(1);
    }
  }
  void OpenStream(const char *filename,
                  std::ofstream *stream) {
    stream->open(filename, std::ios_base::binary|std::ios_base::out);
    if (!stream->is_open()) {
      std::cerr << "discount-counts-backward: failed to open '"
                << filename << "' for writing\n";
      exit(1);
    }
  }

  float d1_;
  float d2_;
  float d3_;
  float d4_;
  double d1_deriv_;
  double d2_deriv_;
  double d3_deriv_;
  double d4_deriv_;

  std::ifstream count_stream_;  // original counts.
  std::ofstream deriv_stream_;  // the derivatives we write, w.r.t. the original
                                // counts.

  std::ifstream discounted_count_stream_;
  std::ifstream discounted_deriv_stream_;
  // the following 2 are for the one-lower order discount stats.
  std::ifstream backoff_count_stream_;
  std::ifstream backoff_deriv_stream_;

  int64 num_lm_states_processed_;
};

}

int main (int argc, const char **argv) {
  if (argc != 11) {
    std::cerr << "discount-counts-backward: expected usage:\n"
              << "discount-counts-backward <D1> <D2> <D3> <D4> <counts-in>\\\n"
              << "  <discounted-float-counts-in> <discounted-float-derivs-in> \\\n"
              << "  <backoff-counts-in> <backoff-derivs-in> <derivs-out>\n"
              << "This program prints to its stdout the derivatives w.r.t. D1, D2, D3 and D4.\n";
    exit(1);
  }

  // everything happens in the constructor.
  pocolm::CountDiscounterBackward discounter(argc, argv);
  return 0;
}


/*
  See some testing commands in the comments at the bottom of compute-probs.cc
*/
