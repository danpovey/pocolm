// histories-to-null-counts.cc

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
#include <string.h>
#include <errno.h>
#include "pocolm-types.h"
#include "lm-state.h"



/*
   This program is used to compile text-form counts representing sets of
   histories that have been seen (used in pruning, to disallow pruning of
   those n-grams), into null-counts (type NullLmState).

   Note: the representation is such that the farthest-away word in the
   history (i.e. the left-most word) appears in the 'predicted' position,
   in the same place that the next-word would normally appear.
*/

int main (int argc, const char **argv) {

  if (argc != 1) {
    std::cerr << "histories-to-null-counts: expected usage:\n"
              << "histories-to-null-counts < <histories> > <null-counts>\n"
              << "E.g. float-counts-to-histories float.all | LC_ALL=C sort | histories-to-null-counts >protected.counts\n"
              << "This is used to keep track of n-grams that cannot be pruned away, when pruning LMs.\n";
    exit(1);
  }
  errno = 0;

  int32 num_states_written = 0;
  int64 num_predicted = 0;

  bool first_time = true;

  pocolm::NullLmState lm_state;

  std::vector<int32> wseq;
  std::string line;

  // each line of input will look something like:
  // "  12    11    13"
  // which is interpreted as: "  <reversed-history> <predicted-word>\n"
  while (getline(std::cin, line)) {
    const char *str = line.c_str(), *cur_pos = str;
    int32 predicted_word;
    wseq.clear();
    errno = 0;

    // This code assumes there is no space at the end of the line.
    // This is safe, since the data is produced by our own binary.
    while (*cur_pos != '\0') {
      int base = 10;
      errno = 0;
      wseq.push_back(strtol(cur_pos, const_cast<char**>(&cur_pos), base));
      if (errno) goto fail;
    }
    if (wseq.empty()) goto fail;

    predicted_word = wseq.back();
    wseq.pop_back();

    if (lm_state.history != wseq || first_time) {
      if (!first_time) {
        lm_state.Check();
        lm_state.Write(std::cout);
        num_states_written++;
      }
      lm_state.history = wseq;
      lm_state.predicted.clear();
      first_time = false;
    }
    lm_state.predicted.push_back(predicted_word);
    num_predicted++;
    continue;
 fail:
    std::cerr << "histories-to-null-counts: " << strerror(errno) << "\n";
    std::cerr << "histories-to-null-counts: bad input line '" << line << "'\n";
    exit(1);
  }

  if (!first_time) {
    lm_state.Check();
    lm_state.Write(std::cout);
    num_states_written++;
  } else {
    std::cerr << "histories-to-null-counts: processed no data\n";
    exit(1);
  }
  std::cerr << "histories-to-null-counts: processed "
            << num_states_written << " LM states, with "
            << num_predicted << " individual n-grams.\n";

  return 0;
}

/*
  see testing example in egs/simple/local/test_float_counts.sh
 */
