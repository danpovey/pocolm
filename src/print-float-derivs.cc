// print-float-derivs.cc

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
   This program exists to enable human inspection of float-count files and their
   associated derivatives.
*/


int main (int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "print-float-counts: expected usage:\n"
              << "print-float-derivs <float-counts> <float-derivs> >counts.txt\n"
              << "e.g.:\n"
              << "print-float-derivs float.1gram derivs.1gram\n";
    exit(1);
  }

  int64 num_lm_states = 0;
  int64 num_counts = 0;

  std::ifstream counts_input(argv[1],
                             std::ios_base::in|std::ios_base::binary),
      derivs_input(argv[2],
                   std::ios_base::in|std::ios_base::binary);
  if (!counts_input.is_open()) {
    std::cerr << "print-float-derivs: error opening '" << argv[1]
              << "' for reading\n";
    exit(1);
  }
  if (!derivs_input.is_open()) {
    std::cerr << "print-float-derivs: error opening '" << argv[2]
              << "' for reading\n";
    exit(1);
  }

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (counts_input.peek(), !counts_input.eof()) {
    pocolm::FloatLmStateDerivs lm_state;
    lm_state.Read(counts_input);
    lm_state.ReadDerivs(derivs_input);
    lm_state.Print(std::cout);
    num_lm_states++;
    num_counts += lm_state.counts.size();
  }

  std::cerr << "print-float-derivs: printed "
            << num_lm_states << " LM states, with "
            << num_counts << " individual n-grams.\n";
  return 0;
}

// see compute-probs.cc for a command-line example that was used to test this.

