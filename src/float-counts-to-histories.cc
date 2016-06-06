// float-counts-to-histories.cc

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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"


/*
   This program reads in float-counts, and prints out just the histories of the
   LM-states it reads in (if the histories are not empty).  If there is a
   history-state "a b c" (written in the natural word order; it would actually
   be represented as "c b a" in the vector), then this program prints out a line
   " b a\tc" (note, they are all actually integers), with spacing such that
   string sort will coincide with integer-vector sort.  The use of the tab just
   before the count is a kind of hack to manipulate the order in which these
   things appear, so that states of different orders can be properly separated.

   The reason for this odd order relates to how we'll process them; we'll sort
   these lines and pipe them into get-null-counts.  That program will represent
   them as LM-states, e.g. in this example the lm-state for "a b" would have
   a predicted-word "c".

   Note: the reason for processing these 'null-counts' is to identify which
   n-grams are "protected" in the pruning operation.
*/


inline void PrintNumber(int32 i) {
  assert(i < 10000000 &&
         "To deal with vocabularies over 10 million, change setw(7) to setw(8)"
         "or more.");
  std::cout << std::setfill(' ') << std::setw(7) << i;
}

int main (int argc, char **argv) {
  if (argc != 1) {
    std::cerr << "float-counts-to-histories: expected usage: print-float-counts <float_counts >histories.txt\n"
              << "You'll typically pipe this into sort and then into get-null-counts.\n";
    exit(1);
  }

  int64 num_histories_total = 0,
      num_histories_printed = 0;

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (std::cin.peek(), !std::cin.eof()) {
    pocolm::FloatLmState lm_state;
    lm_state.Read(std::cin);
    const std::vector<int32> &history = lm_state.history;
    // histories will only be 'protected' if there is at least one
    // nonzero n-gram count in that state.
    bool found_nonzero_count = false;
    std::vector<std::pair<int32, float> >::const_iterator
        counts_iter = lm_state.counts.begin(),
        counts_end = lm_state.counts.end();
    for (; counts_iter != counts_end; ++counts_iter) {
      if (counts_iter->second != 0.0) {
        found_nonzero_count = true;
        break;
      }
    }

    if (found_nonzero_count && !history.empty()) {
      for (size_t i = 1; i < history.size(); i++) {
        std::cout << ' ';
        PrintNumber(history[i]);
      }
      std::cout << "\t";
      PrintNumber(history[0]);
      std::cout << '\n';
      num_histories_printed++;
    }
    num_histories_total++;
  }

  std::cerr << "float-counts-to-histories: printed "
            << num_histories_printed << " histories (out of "
            << num_histories_total << " in total.\n";
  return 0;
}

// see discount-counts.cc for a command-line example that was used to test this.

