// discount-counts.cc

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

/*
   This program discounts n-gram stats (of order >1).  It outputs the discounted
   stats and also the one-lower-order "discount" stats (by which we mean subtracted
   part of the stats, aggregated by removing the leftmost part of the history).

   The discounting method is a generalization of modified Kneser-Ney discounting (see
   Goodman, "A Bit of Progress in Language Modeling"), in which we discount a
   specified proportion of the (first, second and third, and 4+) counts, according to
   three values 1 > D1 > D2 > D3 > D4 > 0.  Note, D4 would be zero in standard
   modified Kneser-Ney, but in our method we allow it to be nonzero (in standard Kneser-Ney
  there would be no way to estimate it, but in our framework it's possible to do so based
   on dev-data probabilities.).

   Our method is a generalization of modified Kneser-Ney because in our
   framework, these counts are not automatically equal to one.  I.e. instead of
   dealing with counts-of-counts, we keep track of the discounted pieces and
   their exact magnitudes.  Note, be careful because our values D1, D2 and D3
   (and D4) are defined differently than in the original modified Kneser-Ney.
   */


namespace pocolm {

class CountDiscounter {
 public:
  CountDiscounter(int argc,
                  const char **argv): num_lm_states_processed_(0) {
    // args are: program name, D1, D2, D3, D4, counts-input-filename,
    // discounted-float-counts-filename, discount-counts-filename.
    assert(argc == 8);
    ReadArgs(argv);
    ProcessInput();
  }

  ~CountDiscounter() {
    discounted_output_.close();
    backoff_output_.close();
    if (discounted_output_.fail() || backoff_output_.fail()) {
      std::cerr << "discount-counts: failed to close output (disk full?)\n";
      exit(1);
    }
  }
 private:
  void ProcessInput() {
    bool first_time = true;
    GeneralLmState input_lm_state;
    while (true) {
      input_.peek();
      if (input_.eof())
        break;
      input_lm_state.Read(input_);
      ProcessLmState(first_time, input_lm_state);
      first_time = false;
    }
    if (first_time) {
      std::cerr << "discount-counts: processed no data\n";
      exit(1);
    } else {
      // flush the last state's discount stats.
      OutputDiscountStats();
      std::cerr << "discount-counts: processed "
                << num_lm_states_processed_ << " LM states\n";
    }
  }

  // Note: we expect to process input of a single n-gram order.
  void ProcessLmState(bool first_time, const GeneralLmState &lm_state) {
    num_lm_states_processed_++;
    if (backoff_history_.size() + 1 != lm_state.history.size()) {
      if (first_time) {
        assert(lm_state.history.size() > 0 && "discount-counts should not be "
               "applied to 1-gram input");
        size_t backoff_history_size = lm_state.history.size() - 1;
        backoff_history_.resize(backoff_history_size);
        std::copy(lm_state.history.begin(),
                  lm_state.history.begin() + backoff_history_size,
                  backoff_history_.begin());
      } else {
        std::cerr << "discount-counts: input seems to have differing "
                  << "n-gram orders\n";
        exit(1);
      }
    }
    // 'discounted_state' is what's left of the counts in 'lm_state' after we've
    // removed the discounted pieces.
    FloatLmState discounted_state;
    discounted_state.history = lm_state.history;
    discounted_state.counts.resize(lm_state.counts.size());

    if (!std::equal(backoff_history_.begin(), backoff_history_.end(),
                    lm_state.history.begin())) {
      // the history of the backoff state has changed.
      // (remember that histories are reversed in these vectors, so
      // to back the history off we remove the right-most element.)
      OutputDiscountStats();
      size_t backoff_history_size = backoff_history_.size();
      // update the backoff history to corresond to the current input.
      std::copy(lm_state.history.begin(),
                lm_state.history.begin() + backoff_history_size,
                backoff_history_.begin());
    }

    std::vector<std::pair<int32, Count> >::const_iterator in_iter =
        lm_state.counts.begin(), in_end = lm_state.counts.end();
    std::vector<std::pair<int32, float> >::iterator out_iter =
        discounted_state.counts.begin();
    double lm_state_total = lm_state.discount,
        discount_total = lm_state.discount;
    for (; in_iter != in_end; ++in_iter,++out_iter) {
      int32 word = in_iter->first;
      const Count &count = in_iter->second;
      out_iter->first = word;
      // mark these quantities volatile to avoid compiler optimization, so that
      // we can ensure they will be exactly the same value in discount-counts
      // and discount-counts-backward (since the backprop relies on exact
      // floating-point comparisons).
      volatile float top4plus = count.total - count.top1 - count.top2 - count.top3,
          d1 = d1_ * count.top1, d2 = d2_ * count.top2, d3 = d3_ * count.top3,
          d4 = d4_ * top4plus, d = d1 + d2 + d3 + d4;
      // we can set separate_counts to true or false.. it's a design decision.
      if (POCOLM_SEPARATE_COUNTS) {
        // the up to 3 discounted pieces will remain separate in the lower-order
        // state..  I think this will likely perform better, but we can try both
        // ways.
        Count discount;  // the part removed, which will go to the lower-order state.
        discount.top1 = d1;
        discount.top2 = d2;
        discount.top3 = d3;
        discount.total = d;
        backoff_builder_.AddCount(word, discount);
      } else {
        // the up to 3 discounted pieces will be merged at the time we discount
        // them.
        backoff_builder_.AddCount(word, d);
      }
      lm_state_total += count.total;
      discount_total += d;
      // store the discounted count.
      out_iter->second = count.total - d;
    }
    discounted_state.total = lm_state_total;
    discounted_state.discount = discount_total;
    discounted_state.Write(discounted_output_);
  }


  void OutputDiscountStats() {
    // calling this function causes the history and stats in
    // (backoff_history_, backoff_builder_) to be written to
    // backoff_output_.

    GeneralLmState backoff_state;
    backoff_builder_.Output(backoff_history_, &backoff_state);

    backoff_state.Write(backoff_output_);
    // clear the stats that we just wrote.  We'll later modify the history from
    // outside this function.
    backoff_builder_.Clear();
  }

  void ReadArgs(const char **argv) {
    d1_ = ConvertToFloat(argv[1]);
    d2_ = ConvertToFloat(argv[2]);
    d3_ = ConvertToFloat(argv[3]);
    d4_ = ConvertToFloat(argv[4]);
    assert(1.0 >= d1_ && d1_ >= d2_ && d2_ >= d3_ && d3_ >= d4_ && d4_ >= 0);

    input_.open(argv[5], std::ios_base::binary|std::ios_base::in);
    if (input_.fail()) {
      std::cerr << "discount-counts: failed to open '"
                << argv[5] << "' for reading.\n";
      exit(1);
    }

    discounted_output_.open(argv[6], std::ios_base::binary|std::ios_base::out);
    if (discounted_output_.fail()) {
      std::cerr << "discount-counts: failed to open '"
                << argv[6] << "' for writing.\n";
      exit(1);
    }
    backoff_output_.open(argv[7], std::ios_base::binary|std::ios_base::out);
    if (backoff_output_.fail()) {
      std::cerr << "discount-counts: failed to open '"
                << argv[7] << "' for writing.\n";
      exit(1);
    }
  }

  float ConvertToFloat(const char *str) const {
    char *end;
    float ans = strtod(str, &end);
    if (!(*end == 0.0)) {
      std::cerr << "discount-counts: expected float, got '" << str << "'\n";
    }
    if (!(ans >= 0.0 and ans <= 1.0)) {
      // note: we really expect discount < 1.0, but once it gets close to 1,
      // due to rounding it can look like exactly 1.0.
      std::cerr << "discount-counts: discounting values must be "
                << ">=0.0 and <= 1.0: " << str << "\n";
      exit(1);
    }
    return ans;
  }


  float d1_;
  float d2_;
  float d3_;
  float d4_;

  std::ifstream input_;
  std::ofstream discounted_output_;
  std::ofstream backoff_output_;


  // backoff_builder_ and backoff_history_ keep track of the
  // stats that we discounted from the input, and aggregates them
  // over the lower-order history state.
  std::vector<int32> backoff_history_;
  GeneralLmStateBuilder backoff_builder_;

  int64 num_lm_states_processed_;
};

}

int main (int argc, const char **argv) {
  if (argc != 8) {
    std::cerr << "discount-counts: expected usage: discount-counts <D1> <D2> <D3> <D4> <counts-in> <discounted-float-counts-out> <backoff-counts-out>\n"
              << "e.g.: discount-counts 0.8 0.5 0.2 0.1 dir/merged/3.ngram dir/discounted/3.ngram dir/discounts/3.ngram\n"
              << "(note: <discounted-float-counts-out> are written as float-counts, <backoff-counts-out> are written as\n"
              << "general counts (where we keep track of top1, top2, top3)\n";
    exit(1);
  }

  // everything happens in the constructor.
  pocolm::CountDiscounter discounter(argc, argv);
  return 0;
}


/*

  some testing:
  # print the counts after discounting.
( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | merge-counts /dev/stdin,0.5 | discount-counts 0.8 0.7 0.6 /dev/stdin /dev/stdout /dev/null | print-float-counts

get-int-counts: processed 5 LM states, with 6 individual n-grams.
merge-counts: wrote 4 LM states.
 [ 11 1 ]: total=1 discounted=0.75 12->0.25
 [ 12 11 ]: total=1 discounted=0.75 13->0.25
 [ 13 12 ]: total=1 discounted=0.8 2->0.1 14->0.1
 [ 14 13 ]: total=0.5 discounted=0.4 2->0.1
print-float-counts: printed 4 LM states, with 5 individual n-grams.

# print the lower-order, discounted part.

( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | merge-counts /dev/stdin,0.5 | discount-counts 0.8 0.7 0.6 /dev/stdin /dev/null /dev/stdout | print-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
merge-counts: wrote 4 LM states.
 [ 11 ]: 12->(0.75,0.4,0.35)
 [ 12 ]: 13->(0.75,0.4,0.35)
 [ 13 ]: 2->(0.4,0.4) 14->(0.4,0.4)
print-counts: printed 3 LM states, with 4 individual n-grams.


 */
