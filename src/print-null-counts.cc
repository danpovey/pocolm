// print-null-counts.cc

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
   This program exists to enable human inspection of the files produced by
   histories-to-null-counts.  It reads such counts from its stdin, and writes them
   in human-readable text form to the stdout.
*/


int main (int argc, char **argv) {
  if (argc != 1) {
    std::cerr << "print-null-counts: expected usage: print-null-counts <counts.int >counts.txt\n";
        exit(1);
  }

  int64 num_lm_states = 0;
  int64 num_counts = 0;

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (std::cin.peek(), !std::cin.eof()) {
    pocolm::NullLmState lm_state;
    lm_state.Read(std::cin);
    lm_state.Print(std::cout);
    num_lm_states++;
    num_counts += lm_state.predicted.size();
  }

  std::cerr << "print-null-counts: printed "
            << num_lm_states << " LM states, with "
            << num_counts << " individual n-grams.\n";
  return 0;
}

/*
  see testing example in egs/simple/local/test_float_counts.sh
*/
