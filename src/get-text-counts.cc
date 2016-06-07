// get-text-counts.cc

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
#include <vector>
#include <iomanip>
#include <stdlib.h>

/*
   This standalone C++ program is intended to turn integerized text into
   strings that identify the n-gram counts that we need to count up.
   As the simplest example, if we do

   echo 11 12 13 | get-text-counts 2

   (2 is interpreted as the n-gram order), the program would output

   1 11
   11 12
   12 13
   13 2

   ... note that 1 is "special" and refers to the start-of-sentence symbol
   normally written <s>, and 2 is also "special" and refers to the end-of-sentence
   symbol </s>.  This data will be piped into sort -c.

   Note that for n-gram orders higher than 2-gram, we reverse the "history"
   part of the count, so the command

      echo 11 12 13 | get-text-counts 3

   would produce the following output:

   1 11
   11 1 12
   12 11 13
   13 12 2

   (read: MSRLM: a scalable language modeling toolkit, to see why.. it's
   all about sort order).

 */


int main (int argc, char **argv) {
  int ngram_order;
  if (argc != 2) {
    std::cerr << "Expected usage: get-text-counts <ngram-order>\n";
    exit(1);
  }
  ngram_order = atoi(argv[1]);
  if (!(ngram_order > 0)) {
    std::cerr << "Expected usage: get-text-counts <ngram-order>\n"
              << "ngram-order must be > 0\n";
    exit(1);
  }

  long int num_lines_processed = 0, num_words_processed = 0;
  std::string line;
  std::vector<int> line_ints;

  while (std::getline(std::cin, line)) {
    num_lines_processed++;
    std::istringstream str(line);
    line_ints.clear();
    line_ints.push_back(1);  // <-- 1 is a special symbol representing
                             // beginning-of-sentence (BOS, or <s>)
    int i;
    while (str >> i) {
      assert(i > 2);
      line_ints.push_back(i);
    }
    line_ints.push_back(2);  // <-- 2 is a special symbol representing
                             // ebd-of-sentence (EOS, or <s>)
    int size = line_ints.size();
    num_words_processed += size;
    // 'count' will contain the reversed history and then the predicted word.

    // the '<< std::setfill(' ') << setw(7)' is to left pad with spaces.
    // this allows us to get the correct sorting order.  This width allows
    // for vocabulary sizes up to 10 million - 1... should be enough for
    // a while.
    for (int pos = 1; pos < size; pos++) {
      for (int h = pos - 1; h >= 0 && h > pos - ngram_order; h--)
        std::cout << std::setfill(' ') << std::setw(7) << line_ints[h] << " ";
      std::cout << std::setfill(' ') << std::setw(7) << line_ints[pos] << "\n";
      assert(line_ints[pos] < 10000000 &&
             "To deal with vocabularies over 10 million, change setw(7) to setw(8)"
             "or more.");
    }
  }
  std::cerr << "get-text-counts: processed " << num_lines_processed
            << " lines, with (on average) "
            << (num_words_processed * 1.0 / num_lines_processed)
            << " words per line.\n";
  return (num_lines_processed > 0 ? 0 : 1);
}
