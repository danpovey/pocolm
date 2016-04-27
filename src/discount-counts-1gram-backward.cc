// discount-counts-1gram-backward.cc

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
   This program is the 'backprop' program for discount-counts-1gram.  To
   understand it, first understand discount-counts-1gram.  It propagates
   derivatives backward through that program.
*/


namespace pocolm {

class UnigramCountDiscounterBackward {
 public:
  UnigramCountDiscounterBackward(int argc,
                                 const char **argv) {
    assert(argc == 5);
    GeneralLmStateDerivs input_lm_state;
    FloatLmStateDerivs output_lm_state;
    ReadInputs(argv, &input_lm_state, &output_lm_state);
    DoBackprop(output_lm_state, &input_lm_state);
    WriteOutput(argv, input_lm_state);
  }
 private:
  void ReadInputs(const char **argv, GeneralLmStateDerivs *input_lm_state,
                  FloatLmStateDerivs *output_lm_state) {
    std::ifstream input_counts(argv[1],
                               std::ios::in|std::ios::binary);
    if (!input_counts.is_open()) {
      std::cerr << "discount-counts-1gram-backward: error opening '"
                << argv[1] << "' for reading.\n";
      exit(1);
    }
    input_lm_state->Read(input_counts);

    std::ifstream output_counts(argv[2],
                                std::ios::in|std::ios::binary);
    if (!output_counts.is_open()) {
      std::cerr << "discount-counts-1gram-backward: error opening '"
                << argv[2] << "' for reading.\n";
      exit(1);
    }
    output_lm_state->Read(output_counts);
    std::ifstream output_derivs(argv[3],
                                std::ios::in|std::ios::binary);
    if (!output_derivs.is_open()) {
      std::cerr << "discount-counts-1gram-backward: error opening '"
                << argv[3] << "' for reading.\n";
      exit(1);
    }
    output_lm_state->ReadDerivs(output_derivs);
  }
  void WriteOutput(const char **argv,
                   const GeneralLmStateDerivs &input_lm_state) {
    std::ofstream input_derivs(argv[4],
                               std::ios::out|std::ios::binary);
    if (!input_derivs.is_open()) {
      std::cerr << "discount-counts-1gram-backward: error reading '"
                << argv[3] << "' for reading.\n";
      exit(1);
    }
    input_lm_state.WriteDerivs(input_derivs);
  }


  void DoBackprop(const FloatLmStateDerivs &output_lm_state,
                  GeneralLmStateDerivs *input_lm_state) {
    int32 vocab_size = output_lm_state.counts.size() + 1;
    assert(vocab_size > 0);

    double extra_count_deriv = 0.0, extra_unk_count_deriv = 0.0;
    for (int32 i = kEosSymbol; i <= vocab_size; i++) {
      float output_deriv = output_lm_state.count_derivs[i - kEosSymbol];
      assert(output_lm_state.counts[i - kEosSymbol].first == i);
      if (i != kUnkSymbol) {
        extra_count_deriv += output_deriv;
      } else {
        extra_unk_count_deriv = output_deriv;
      }
    }

    // in the forward computation we did
    //float extra_count = total_discount * (1.0 - POCOLM_UNK_PROPORTION) /
    //    (vocab_size_ - 2),
    //    extra_unk_count = POCOLM_UNK_PROPORTION * total_discount;
    // the backprop for this is as follows:
    double total_discount_deriv =
        extra_count_deriv * (1.0 - POCOLM_UNK_PROPORTION) / (vocab_size - 2) +
        POCOLM_UNK_PROPORTION * extra_unk_count_deriv;

    assert(input_lm_state->counts.size() == input_lm_state->count_derivs.size());
    int32 num_counts = input_lm_state->counts.size();
    for (int32 i = 0; i < num_counts; i++) {
      int32 word = input_lm_state->counts[i].first;
      Count &count_deriv = input_lm_state->count_derivs[i];
      float output_deriv = output_lm_state.count_derivs[word - kEosSymbol];
      assert(output_lm_state.counts[word - kEosSymbol].first == word);

      // diff_deriv is just a subexpression that we repeatedly need.
      float diff_deriv = total_discount_deriv - output_deriv;

      // backprop through the following code:
      // float discount = POCOLM_UNIGRAM_D1 * count.top1
      //   + POCOLM_UNIGRAM_D2 * count.top2
      //   + POCOLM_UNIGRAM_D3 * count.top3;
      // total_discount += discount;
      // unigram_counts[word] = count.total - discount;
      count_deriv.top1 = POCOLM_UNIGRAM_D1 * diff_deriv;
      count_deriv.top2 = POCOLM_UNIGRAM_D2 * diff_deriv;
      count_deriv.top3 = POCOLM_UNIGRAM_D3 * diff_deriv;
      count_deriv.total = output_deriv;
    }
  }
};

}

int main (int argc, const char **argv) {

  if (argc != 5) {
    std::cerr << "discount-counts-1gram-backward: expected usage:\n"
              << "discount-counts-1gram-backward <counts-in> <float-counts-in> <float-derivs-in> <derivs-out>\n"
              << "This program is the 'backprop' counterpart of discount-counts-1gram.\n"
              << "The arguments <counts-in> and <float-counts-in> are the input and output\n"
              << "respectively of discount-counts-1gram; <float-derivs-in> are the derivatives\n"
              << "corresponding to <float-counts-in>, and <derivs-out> are the backprop'ed\n"
              << "derivatives w.r.t. <counts-in>.\n";
    exit(1);
  }

  // everything happens inside the constructor below.
  pocolm::UnigramCountDiscounterBackward(argc, argv);

  return 0;
}


/*
  there are some testing commands for this in a comment at the bottom of
  compute-probs.cc.
*/
