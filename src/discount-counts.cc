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
   specified proportion of the (first, second and third) counts, according to
   three values 0 < D1 < 1, 0 < D2 < 1, and 0 < D3 < 1.  Normally we'll have D1
   > D2 > D3, but this is not required.

   Our method is a generalization of modified Kneser-Ney because in our
   framework, these counts are not automatically equal to one.  I.e. instead of
   dealing with counts-of-counts, we keep track of the discounted pieces and
   their exact magnitudes.  Note, be careful because our values D1, D2 and D3
   are defined differently than in the original modified Kneser-Ney.
*/


namespace pocolm {

class CountDiscounter {
 public:
  CountDiscounter(int argc,
                  const char **argv): num_lm_states_processed_(0) {
    // args are: program name, D1, D2, D3, discounted-counts-filename,
    // discount-counts-filename.
    assert(argc == 6);
    ReadArgs(argv);
    ProcessInput();
  }

  ~CountDiscounter() {
    discounted_output_.close();
    discount_output_.close();
    if (discounted_output_.fail() || discount_output_.fail()) {
      std::cerr << "discount-counts: failed to close output (disk full?)\n";
    }
  }
 private:
  void ProcessInput() {
    bool first_time = true;
    GeneralLmState input_lm_state;
    while (true) {
      std::cin.peek();
      if (std::cin.eof())
        break;
      input_lm_state.Read(std::cin);
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

  void ProcessLmState(bool first_time, const GeneralLmState &lm_state) {
    num_lm_states_processed_++;
    if (discount_history_.size() + 1 != lm_state.history.size()) {
      if (first_time) {
        assert(lm_state.history.size() > 0 && "discount-counts should not be "
               "applied to 1-gram input");
        size_t backoff_history_size = lm_state.history.size() - 1;
        discount_history_.resize(backoff_history_size);
        std::copy(lm_state.history.begin(),
                  lm_state.history.begin() + backoff_history_size,
                  discount_history_.begin());
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

    if (!std::equal(discount_history_.begin(), discount_history_.end(),
                    lm_state.history.begin())) {
      // the history of the backoff state has changed.
      // (remember that histories are reversed in these vectors, so
      // to back the history off we remove the right-most element.)
      OutputDiscountStats();
      size_t backoff_history_size = discount_history_.size();
      // update the backoff history to corresond to the current input.
      std::copy(lm_state.history.begin(),
                lm_state.history.begin() + backoff_history_size,
                discount_history_.begin());
    }


    std::vector<std::pair<int32, Count> >::const_iterator in_iter =
        lm_state.counts.begin(), in_end = lm_state.counts.end();
    std::vector<std::pair<int32, float> >::iterator out_iter =
        discounted_state.counts.begin();
    double lm_state_total = 0.0,
        discount_total = 0.0;
    for (; in_iter != in_end; ++in_iter,++out_iter) {
      int32 word = in_iter->first;
      const Count &count = in_iter->second;
      out_iter->first = word;
      float d1 = d1_ * count.top1, d2 = d2_ * count.top2, d3 = d3_ * count.top3,
          d = d1 + d2 + d3;
      // we can set separate_counts to true or false.. it's a design decision.
      bool separate_counts = true;
      if (separate_counts) {
        // the up to 3 discounted pieces will remain separate in the lower-order
        // state..  I think this will likely perform better, but we can try both
        // ways.
        Count discount;  // the part removed, which will go to the lower-order state.
        discount.top1 = d1;
        discount.top2 = d2;
        discount.top3 = d3;
        discount.total = d;
        discount_builder_.AddCount(word, discount);
      } else {
        // the up to 3 discounted pieces will be merged at the time we discount
        // them.
        discount_builder_.AddCount(word, d);
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
    // (discount_history_, discount_builder_) to be written to
    // discount_output_.

    GeneralLmState backoff_state;
    backoff_state.history = discount_history_;
    discount_builder_.Output(&backoff_state.counts);
    backoff_state.Write(discount_output_);
    // clear the stats that we just wrote.  We'll later modify the history from
    // outside this function.
    discount_builder_.Clear();
  }

  void ReadArgs(const char **argv) {
    char *end;
    d1_ = strtod(argv[1], &end);
    if (!(d1_ > 0.0 && d1_ < 1.0 && *end == '\0')) {
      std::cerr << "discount-counts: d1 must be >0.0 and <1.0\n";
      exit(1);
    }
    d2_ = strtod(argv[2], &end);
    if (!(d2_ > 0.0 && d2_ < 1.0 && *end == '\0')) {
      std::cerr << "discount-counts: d2 must be >0.0 and <1.0\n";
      exit(1);
    }
    d3_ = strtod(argv[3], &end);
    if (!(d3_ > 0.0 && d3_ < 1.0 && *end == '\0')) {
      std::cerr << "discount-counts: d3 must be >0.0 and <1.0\n";
      exit(1);
    }
    assert(1.0 > d1_ && d1_ >= d2_ && d2_ >= d3_ && d3_ > 0);
    discounted_output_.open(argv[4], std::ios_base::binary|std::ios_base::out);
    if (discounted_output_.fail()) {
      std::cerr << "discount-counts: failed to open '"
                << argv[4] << "' for writing.\n";
      exit(1);
    }
    discount_output_.open(argv[5], std::ios_base::binary|std::ios_base::out);
    if (discount_output_.fail()) {
      std::cerr << "discount-counts: failed to open '"
                << argv[5] << "' for writing.\n";
      exit(1);
    }
  }


  float d1_;
  float d2_;
  float d3_;

  std::ofstream discounted_output_;
  std::ofstream discount_output_;


  // discount_builder_ and discount_history_ keep track of the
  // stats that we discounted from the input, and aggregates them
  // over the lower-order history state.
  std::vector<int32> discount_history_;
  GeneralLmStateBuilder discount_builder_;

  int64 num_lm_states_processed_;
};

}

int main (int argc, const char **argv) {
  if (argc != 6) {
    std::cerr << "discount-counts: expected usage:  discount-counts <D1> <D2> <D3> <discounted-counts-out> <discount-counts-out>  <counts>a\n"
              << "e.g.: discount-counts 0.8 0.5 0.2 dir/discounted/3.fngram dir/discounts/3.ngram <dir/merged.3.ngram\n"
              << "(note: <discounted-counts-out> are written as float-counts, <discount-counts-out> are written as\n"
              << "general counts (where we keep track of top1, top2, top3)\n";
  }

  // everything happens in the constructor.
  pocolm::CountDiscounter discounter(argc, argv);
  return 0;
}


/*

  some testing:
  # print the counts after discounting.
( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | merge-counts /dev/stdin,0.5 | discount-counts 0.8 0.7 0.6 /dev/stdout /dev/null | print-float-counts

get-int-counts: processed 5 LM states, with 6 individual n-grams.
merge-counts: wrote 4 LM states.
 [ 11 1 ]: total=1 discounted=0.75 12->0.25
 [ 12 11 ]: total=1 discounted=0.75 13->0.25
 [ 13 12 ]: total=1 discounted=0.8 2->0.1 14->0.1
 [ 14 13 ]: total=0.5 discounted=0.4 2->0.1
print-float-counts: printed 4 LM states, with 5 individual n-grams.

# print the lower-order, discounted part.

( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | merge-counts /dev/stdin,0.5 | discount-counts 0.8 0.7 0.6 /dev/null /dev/stdout | print-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
merge-counts: wrote 4 LM states.
 [ 11 ]: 12->(0.75,0.4,0.35)
 [ 12 ]: 13->(0.75,0.4,0.35)
 [ 13 ]: 2->(0.4,0.4) 14->(0.4,0.4)
print-counts: printed 3 LM states, with 4 individual n-grams.


 */
