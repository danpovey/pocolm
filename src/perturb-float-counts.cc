// perturb-float-counts.cc

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
  This function perturbs the count 'count' by a small random offset
  ('delta' controls the range of the relative change in 'count').
  It returns the predicted change in the overall objective that results
  from this random change, which equals the change times 'deriv';
  'deriv' is the compute derivative of the objective function w.r.t.
  this count.
*/
float PerturbCount(float delta,
                   float deriv,
                   float *count) {
  float relative_change = delta * (((rand() % 100) - 50) / 100.0),
      this_change = *count * relative_change;
  *count += this_change;
  return deriv * this_change;
}


/*
   For use in testing derivatives, this program perturbs float-counts by a small
   (relative) amount; it also reads the derivatives and prints the predicted
   change in objective function to the standard output.
*/


int main (int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "perturb-float-counts: expected usage:\n"
              << "perturb-float-counts <srand-seed> <float-counts-in> <float-derivs-in> <float-counts-out>\n"
              << "This program prints to the standard output the objective function change\n"
              << "that is predicted to result from the perturbation (based on the\n"
              << "derivatives).\n";
    exit(1);
  }

  srand(atoi(argv[1]));

  std::ifstream counts_input(argv[2],
                             std::ios_base::in|std::ios_base::binary),
      derivs_input(argv[3],
                   std::ios_base::in|std::ios_base::binary);
  if (!counts_input.is_open()) {
    std::cerr << "print-float-derivs: error opening '" << argv[2]
              << "' for reading\n";
    exit(1);
  }
  if (!derivs_input.is_open()) {
    std::cerr << "print-float-derivs: error opening '" << argv[3]
              << "' for reading\n";
    exit(1);
  }
  std::ofstream counts_output(argv[4],
                              std::ios_base::out|std::ios_base::binary);
  if (!counts_output.is_open()) {
    std::cerr << "print-float-derivs: error opening '" << argv[4]
              << "' for writing\n";
    exit(1);
  }

  // delta controls how much we perturb the counts.  It's hard-coded.
  const float delta = 5.0e-03;

  int64 num_lm_states = 0;
  int64 num_counts = 0;

  double tot_objf_change = 0.0;

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (derivs_input.peek(), !derivs_input.eof()) {
    pocolm::FloatLmStateDerivs lm_state;
    lm_state.Read(counts_input);
    lm_state.ReadDerivs(derivs_input);
    assert(lm_state.total_deriv == 0.0);

    tot_objf_change += PerturbCount(delta, lm_state.discount_deriv,
                                    &lm_state.discount);
    size_t num_counts = lm_state.counts.size();
    for (size_t i = 0; i < num_counts; i++)
      tot_objf_change += PerturbCount(delta, lm_state.count_derivs[i],
                                      &lm_state.counts[i].second);
    lm_state.ComputeTotal();  // adjust the total count.
    lm_state.Write(counts_output);

    num_lm_states++;
    num_counts += lm_state.counts.size();
  }

  counts_output.close();
  if (counts_output.fail()) {
    std::cerr << "perturb-float-counts: error closing stream "
              << argv[3] << " (disk full?)\n";
    exit(1);
  }

  std::cerr << "perturb-float-counts: perturbed "
            << num_lm_states << " LM states, with "
            << num_counts << " individual n-grams; delta = " << delta
            << ", predicted-objf-change = " << tot_objf_change << "\n";
  std::cout << tot_objf_change << "\n";
  return 0;
}


// see discount-counts.cc for a command-line example that was used to test this.

