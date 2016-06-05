// split-float-counts.cc

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
   This program operates on 'float-counts'; it splits them up by taking the most
   recent word in the history modulo the number of outputs.  The empty
   history-state is written to all outputs.
*/


int main (int argc, const char **argv) {
  if (argc < 3) {
    std::cerr << "split-float-counts: expected usage:\n"
              << "split-float-counts  <output1> <output2> ... <outputN>  < <input-float-counts>\n"
              << "This program reads float-counts from its stdin, and distributes them\n"
              << "among the provided outputs by taking the most recent word in the history\n"
              << "modulo the number of outputs.\n"
              << "The -d option takes an integer argument N > 0; if supplied, we will\n"
              << "divide the most-recent-word by N before taking it modulo the number of\n"
              << "outputs.  This is useful in splitting counts that have already been\n"
              << "split.";
    exit(1);
  }

  int num_outputs = argc - 1;

  std::ofstream *outputs = new std::ofstream[num_outputs];

  for (int32 i = 0; i < num_outputs; i++) {
    outputs[i].open(argv[i + 1], std::ios_base::binary|std::ios_base::out);
    if (!outputs[i]) {
      std::cerr << "split-float-counts: Failed to open '" << argv[i + 1] << "' for output.\n";
      exit(1);
    }
  }
  errno = 0;

  int32 num_states_written = 0;
  std::vector<int64> counts_written_per_output(num_outputs, 0);

  pocolm::FloatLmState lm_state;

  while (std::cin.peek(),!std::cin.eof()) {
    lm_state.Read(std::cin);
    if (lm_state.history.empty()) {
      // write to all outputs.
      for (int i = 0; i < num_outputs; i++) {
        lm_state.Write(outputs[i]);
        counts_written_per_output[i] += lm_state.counts.size();
      }
    } else {
      int32 most_recent_history_word = lm_state.history[0];
      assert(most_recent_history_word > 0);
      int32 output = most_recent_history_word % num_outputs;
      counts_written_per_output[output] += lm_state.counts.size();
      lm_state.Write(outputs[output]);
    }
  }

  std::ostringstream info_string;
  for (int32 i = 0; i < num_outputs; i++) {
    outputs[i].close();
    if (outputs[i].fail()) {
      std::cerr << "split-float-counts: failed to close file "
                << argv[i + 1] << " (disk full?)\n";
      exit(1);
    }
    info_string << " " << counts_written_per_output[i];
  }
  delete [] outputs;


  std::cerr << "split-float-counts: processed "
            << num_states_written << " LM states, with the counts "
            << "for each output respectively as: "
            << info_string.str() << "\n";

  return 0;
}

