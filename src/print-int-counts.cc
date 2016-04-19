// print-int-counts.cc

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
   get-int-counts.  It reads such counts from its stdin, and writes them
   in human-readable text form to the stdout.
*/


int main (int argc, char **argv) {
  if (argc != 1) {
    std::cerr << "print-int-counts: expected usage: print-int-counts <counts.int >counts.txt\n";
        exit(1);
  }

  int32 num_lm_states = 0;
  int64 num_counts = 0;

  // we only get EOF after trying to read past the end of the file,
  // so first call peek().
  while (std::cin.peek(), !std::cin.eof()) {
    pocolm::IntLmState lm_state;
    lm_state.Read(std::cin);
    lm_state.Print(std::cout);
    num_lm_states++;
    num_counts += lm_state.counts.size();
  }

  std::cerr << "print-int-counts: printed "
            << num_lm_states << " LM states, with "
            << num_counts << " individual n-grams.\n";
  return 0;
}


/*

  some testing:
( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/stdout /dev/null | print-int-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
 [ 1 ] -> 11->2
 print-int-counts: printed 1 LM states, with 1 individual n-grams.

 ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | print-int-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
 [ 11 1 ] -> 12->2
 [ 12 11 ] -> 13->2
 [ 13 12 ] -> 14->1 2->1
 [ 14 13 ] -> 2->1
print-int-counts: printed 4 LM states, with 5 individual n-grams.

 */
