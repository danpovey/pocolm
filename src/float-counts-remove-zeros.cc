// float-counts-remove-zeros.cc

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
void RemoveZeroCounts(FloatLmState *lm_state) {
  std::vector<std::pair<int32, float> >::const_iterator
      in_iter = lm_state->counts.begin(),
      in_end = lm_state->counts.end();
  std::vector<std::pair<int32, float> >::iterator
      out_iter = lm_state->counts.begin();
  for (; in_iter != in_end; ++in_iter) {
    if (in_iter->second != 0.0) {
      *out_iter = *in_iter;
      out_iter++;
    }
  }
  size_t new_size = out_iter - lm_state->counts.begin();
  lm_state->counts.resize(new_size);
}
}

/*
  This program copies float-counts while removing zero-valued counts
  and LM states that have no counts.
*/


int main (int argc, char **argv) {
  if (argc != 1) {
    std::cerr << "Usage: float-counts-remove-zeros  < <float-counts> > <float-counts>\n"
              << "This program copies float-counts while removing zero counts and\n"
              << "LM-states that have no counts.\n";
    exit(1);
  }

  int64 num_lm_states_in = 0,
      num_lm_states_out = 0,
      num_counts_in = 0,
      num_counts_out = 0;

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (std::cin.peek(), !std::cin.eof()) {
    pocolm::FloatLmState lm_state;
    lm_state.Read(std::cin);
    num_lm_states_in++;
    num_counts_in += static_cast<int64>(lm_state.counts.size());
    pocolm::RemoveZeroCounts(&lm_state);
    if (!lm_state.counts.empty()) {
      num_lm_states_out++;
      num_counts_out += static_cast<int64>(lm_state.counts.size());
      lm_state.Print(std::cout);
    }
  }

  std::cerr << "float-counts-remove-zeros: reduced LM states from "
            << num_lm_states_in << " to " << num_lm_states_out
            << " and counts from "
            << num_counts_in << " to " << num_counts_out << ".\n";

  return 0;
}

// see egs/simple/local/test_float_counts.sh for a a command line example of using this.

