// discount-counts-1gram.cc

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
   This program discounts n-gram stats of order 1.  Because the process of choosing
   vocabularies is often against the assumptions required to properly estimate
   discounting factors, it's safest to just use fixed discounting factors for
   order 1, and that's what we do here.

   The following rule is very arbitrarily chosen.

   We subtract 0.75, 0.25 and 0.1 of the first, second and third counts.   Half of
  this mass is divided equally among all non-<unk>/<s> words in the vocabulary, and
  the remaining half is assigned to <unk>, in addition to any data-derived counts
  for <unk> (which will have happened if our training data contained words that
  were not in the vocabulary, which would have been turned into <unk> during
  data preparation.
*/


namespace pocolm {

class UnigramCountDiscounter {
 public:
  UnigramCountDiscounter(int argc,
                         const char **argv) {
    assert(argc == 2);
    char *end;
    vocab_size_ = strtol(argv[1], &end, 10);
    // vocab size must be at least 3 because it includes the special symbols
    // <s>, </s> and <unk>.
    if (!(vocab_size_ > 3 && *end == '\0')) {
      std::cerr << "discount-counts-1gram: invalid vocabulary size '"
                << vocab_size_ << "'\n";
      exit(1);
    }
    ProcessInput();
  }

 private:
  void ProcessInput() {
    // there should be only one lm-state to process, as we're dealing
    // with the unigram counts.
    GeneralLmState input_lm_state;
    input_lm_state.Read(std::cin);
    std::cin.peek();
    if (!std::cin.eof()) {
      std::cerr << "discount-counts-1gram: too much input\n";
      exit(1);
    }
    FloatLmState discounted_unigram_state;
    DiscountUnigramState(input_lm_state,
                         &discounted_unigram_state);
    discounted_unigram_state.Write(std::cout);
    exit(0);
  }

  void DiscountUnigramState(const GeneralLmState &input_lm_state,
                            FloatLmState *output_lm_state) {
    // We fix by hand the discounting factors for unigram, because setting them
    // automatically is not likely to be very robust in the unigram case
    // e.g. consider cases like when the vocabulary includes all words of dev
    // data, or only words with count >2 in training data, etc.

    // 'unigram_counts' is indexed by word index... index 0 is not used as that
    // is not a valid word index.
    std::vector<float> unigram_counts(vocab_size_ + 1, 0.0);

    // We don't expect the unigram counts to have a nonzero 'discount'
    // value [note: this is only nonzero due to enforcing a min-count,
    // and this is not done for bigram or unigram].
    assert(input_lm_state.discount == 0);

    double total_count = 0.0, total_discount = input_lm_state.discount;

    std::vector<std::pair<int32, Count> >::const_iterator
        iter = input_lm_state.counts.begin(),
        end = input_lm_state.counts.end();
    for (; iter != end; ++iter) {
      int32 word = iter->first;
      assert(word != kBosSymbol && "<s> should never be predicted.");
      if (!(word > 0 && word <= vocab_size_)) {
        std::cerr << "discount-counts-1gram: invalid word index "
                  << word << " (vs. specified vocabulary size "
                  << vocab_size_ << "\n";
      }
      const Count &count = iter->second;
      float discount = POCOLM_UNIGRAM_D1 * count.top1 +
          POCOLM_UNIGRAM_D2 * count.top2 +
          POCOLM_UNIGRAM_D3 * count.top3;
      assert(discount < count.total);
      float this_count = count.total,
          discounted_count = this_count - discount;
      total_count += this_count;
      total_discount += discount;
      unigram_counts[word] = discounted_count;
    }


    // the general discount is distributed to all words except <unk> and <s>.
    float extra_count = total_discount * (1.0 - POCOLM_UNK_PROPORTION) /
        (vocab_size_ - 2),
        extra_unk_count = POCOLM_UNK_PROPORTION * total_discount;

    std::cerr << "discount-counts-1gram: total count is " << total_count
              << ", total discount is " << total_discount
              << ", increasing unk count from "
              << unigram_counts[kUnkSymbol] << " to "
              << (unigram_counts[kUnkSymbol] + extra_unk_count)
              << " and adding " << extra_count << " to each unigram count.\n";

    unigram_counts[kUnkSymbol] += extra_unk_count;

    for (int32 i = 1; i <= vocab_size_; i++)
      if (i != kBosSymbol && i != kUnkSymbol)
        unigram_counts[i] += extra_count;

    output_lm_state->history.clear();  // it's the unigram state.
    output_lm_state->total = total_count;
    // we explicitly give the counts to all the words in the vocab, there is no
    // need to record the discounted amount as we never back off from unigram.
    output_lm_state->discount = 0.0;
    // we don't have a count for <s>, but we do for all other words.
    assert(kBosSymbol == 1 && kEosSymbol == 2);
    output_lm_state->counts.resize(vocab_size_ - 1);
    for (int32 i = kEosSymbol; i <= vocab_size_; i++) {
      output_lm_state->counts[i - kEosSymbol].first = i;
      float count = unigram_counts[i];
      assert(count > 0.0);
      output_lm_state->counts[i - kEosSymbol].second = count;
    }
  }

  int32 vocab_size_;
};

}

int main (int argc, const char **argv) {

  if (argc != 2) {
    std::cerr << "discount-counts-1gram: expected usage:\n"
              << "discount-counts-1gram <vocab-size>  <counts >float_counts\n"
              << "e.g.: merge-counts ... | discount-counts-1gram 50000 > dir/discounted/1.ngram\n";
    exit(1);
  }

  // everything happens inside the constructor below.
  pocolm::UnigramCountDiscounter(argc, argv);

  return 0;
}


/*

  some testing:

( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | merge-counts /dev/stdin,0.5 | discount-counts 0.8 0.7 0.6 /dev/null /dev/stdout | discount-counts 0.8 0.7 0.6 /dev/null /dev/stdout

print-counts

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
