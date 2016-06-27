// distribute-input-lines.cc

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
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>

/*
   This program reads lines from the standard input and echoes the lines
   round robin to the output files.
 */


int main (int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: distribute-input-lines <output1> <output2> ... <outputN> < <input-lines>\n"
              << "Writes input lines round-robin to the output files.\n";
    exit(1);
  }

  int num_outputs = argc - 1;
  std::ofstream *outputs = new std::ofstream[num_outputs];

  for (int i = 0; i < num_outputs; i++) {
    outputs[i].open(argv[i + 1]);
    if (outputs[i].fail()) {
      std::cerr << "distribute-input-lines: failed to open output to '"
                << argv[i + 1] << "'\n";
      exit(1);
    }
  }

  size_t count = 0;
  std::string line;
  while (std::getline(std::cin, line)) {
    if (!(outputs[count % num_outputs] << line << '\n').good()) {
      std::cerr << "distribute-input-lines: failed to write to '"
                << argv[(count % num_outputs) + 1] << "'\n";
      exit(1);
    }
    count++;
  }

  for (int i = 0; i < num_outputs; i++) {
    outputs[i].close();
    if (outputs[i].fail()) {
      std::cerr << "distribute-input-lines: failed to close output to '"
                << argv[i + 1] << "'\n";
      exit(1);
    }
  }
  return 0;
}
