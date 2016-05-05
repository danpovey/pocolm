// get-int-counts.cc

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
#include "errno.h"

/*
   This program is used to compile text-form counts into binary integer counts.
   The pipeline would be something like the following (for a single sentence of
   3 words, although you'd run it with multiple sentences):

   ngram_order=2

   echo 11 12 13 | get-text-counts $ngram_order | sort | uniq -c | get-int-counts dir/order1.int dir/order2.int

   This would put the order-1 counts (i.e. with empty history) in dir/order1.int
   and the order-2 counts (i.e. with history of length 1) in dir/order2.int.
   Note: dir/order1.int will be empty if the ngram-order is >1, but we require
   this anyway.

*/


// paul hsu maxent interpolation

int main (int argc, const char **argv) {
  int num_outputs = argc - 1;
  if (num_outputs <= 0) {
    std::cerr << "get-int-counts: expected usage:\n"
              << "get-int-counts <order1-output> <order2-output> ... < <text-counts>\n"
              << " or: get-int-counts <all-output> < <text-counts>\n"
              << "(the first method gives you counts divided by order, the second\n"
              << "gives you all orders of counts together.. note that typically\n"
              << "the only reason there are multiple orders is end effects.\n"
              << "e.g.:\n"
              << " cat data | get-text-counts <ngram-order> | sort |\\\n"
              << "   uniq -c | get-int-counts <order1-output> <order2-output>\n";
    exit(1);
  }
  // note: if num_outputs is 1 and the input is of higher order, all data just goes
  // to the same file.
  std::ofstream *outputs = new std::ofstream[num_outputs];

  for (int32 i = 0; i < num_outputs; i++) {
    outputs[i].open(argv[i + 1], std::ios_base::binary|std::ios_base::out);
    if (!outputs[i]) {
      std::cerr << "get-int-counts: Failed to open '" << argv[i + 1] << "' for output.";
      exit(1);
    }
  }
  errno = 0;

  int32 num_states_written = 0;
  int64 num_counts = 0;

  bool first_time = true;

  pocolm::IntLmState int_lm_state;

  std::vector<int32> wseq;
  std::string line;

  // each line of input will look something like:
  // "   90 12 11 13"
  // which is: "   <count> <reversed-history> <predicted-word>\n"
  while (getline(std::cin, line)) {
    const char *str = line.c_str(), *cur_pos;
    int32 count, predicted_word;
    wseq.clear();
    errno = 0;
    count = strtol(str, const_cast<char**>(&cur_pos), 10);

    if (count <= 0 || errno != 0)
      goto fail;
    // note, this code assumes there is no space at the end of the line.
    // safe, since the data is produced by our own binary.
    while (*cur_pos != '\0') {
      int base = 10;
      errno = 0;
      wseq.push_back(strtol(cur_pos, const_cast<char**>(&cur_pos), base));
      if (errno) goto fail;
    }
    if (wseq.empty()) goto fail;
    if (wseq.size() > static_cast<size_t>(num_outputs) && num_outputs != 1) {
      std::cerr << "get-int-counts: bad line for n-gram-order="
                << num_outputs << ": '" << line << "'\n";
      exit(1);
    }

    predicted_word = wseq.back();
    wseq.pop_back();

    if (int_lm_state.history != wseq) {
      if (!first_time) {
        int32 output_index = (num_outputs == 1 ? 0 :
                              int_lm_state.history.size());
        int_lm_state.Write(outputs[output_index]);
        num_states_written++;
      }
      first_time = false;
      int_lm_state.Init(wseq);
    }
    int_lm_state.AddCount(predicted_word, count);
    num_counts++;

    continue;
 fail:
    std::cerr << "get-int-counts: " << strerror(errno) << "\n";
    std::cerr << "get-int-counts: bad input line '" << line << "'\n";
    exit(1);
  }

  if (!first_time) {
    int32 output_index = (num_outputs == 1 ? 0 :
                          int_lm_state.history.size());
    int_lm_state.Write(outputs[output_index]);
    num_states_written++;
  } else {
    std::cerr << "get-int-counts: processed no data\n";
    exit(1);
  }
  std::cerr << "get-int-counts: processed "
            << num_states_written << " LM states, with "
            << num_counts << " individual n-grams.\n";

  for (int32 i = 0; i < num_outputs; i++) {
    outputs[i].close();
    if (outputs[i].fail()) {
      std::cerr << "get-int-counts: failed to close file "
                << argv[i + 1] << " (disk full?)\n";
      exit(1);
    }
  }
  delete [] outputs;
  return 0;
}

/*
  testing

 ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort -n | uniq -c | get-int-counts /dev/null /dev/null /dev/null

 */
