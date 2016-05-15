// sum-count-derivs.cc

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
#include <map>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state-derivs.h"


/**
   This program is for summing 'general-count' derivatives-- for the case
   when the derivatives have the same structure (e.g. correspond to the same
   counts file).  This is useful for the order-1 count's derivatives in
   the derivative computation for split directories (it's called from
   get_objf_and_derivs_split.py). */


int main (int argc, const char **argv) {
  if (argc < 4) {
    std::cerr << "sum-count-derivs: expected usage: <general-counts-file> <derivs1> <derivs2>  > <summed-derivs>\n"
              << "This program sums derivatives for general-counts; the derivatives must all\n"
              << "correspond to the same counts file.  The summed derivatives are written\n"
              << "to the standard output.\n";
    exit(1);
  }


  std::ifstream counts_input;
  counts_input.open(argv[1], std::ios_base::in|std::ios_base::binary);
  if (counts_input.fail()) {
    std::cerr << "sum-count-derivs: error opening counts file "
              << argv[1] << "\n";
    exit(1);
  }

  int32 num_deriv_inputs = argc - 2;
  std::ifstream *deriv_inputs = new std::ifstream[num_deriv_inputs];
  for (int32 i = 0; i < num_deriv_inputs; i++) {
    deriv_inputs[i].open(argv[2 + i], std::ios_base::in|std::ios_base::binary);
    if (deriv_inputs[i].fail()) {
      std::cerr << "sum-count-derivs: error opening derivatives file "
                << argv[2 + i] << "\n";
      exit(1);
    }
  }

  int32 num_lm_states = 0;

  while (counts_input.peek(), !counts_input.eof()) {
    pocolm::GeneralLmStateDerivs lm_state;
    lm_state.Read(counts_input);
    lm_state.ReadDerivs(deriv_inputs[0]);
    for (int32 i = 1; i < num_deriv_inputs; i++)
      lm_state.ReadDerivsAdding(deriv_inputs[i]);
    lm_state.WriteDerivs(std::cout);
    num_lm_states++;
  }

  std::cerr << "sum-count-derivs: summed derivatives for " << num_lm_states
            << " LM states.\n";

  return 0;
}
