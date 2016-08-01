// pre-arpa-to-arpa.cc

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
#include "pocolm-types.h"
#include "lm-state.h"


namespace pocolm {

/*
   This program is used together with float-counts-to-arpa to produce an ARPA
   format language model; you pipe the output of float-counts-to-arpa into
   'sort' and then into this program.
*/


class PreArpaProcessor {
 public:
  // The only command line arg is <vocab-file>.
  //
  // Input is read from stdin.  The following describes the format of input:
  // N-gram probability lines: <n-gram-order> <word1> ... <wordN> <log10-prob>, e.g.:
  // ' 3 162 82 978 -2.843283\n'
  // Backoff probability lines: <n-gram-order> <word1> ... <wordN>\t<log10-prob>, e.g:
  // ' 3 162 82 978 \t-1.72432\n'
  // Lines (beginning with 0) that announce the counts of n-grams for a
  // particular n-gram order, e.g.:
  // ' 0  3 894121\n'
  // announces that there are 894121 3-grams.
  // Note: in places in the example inputs there may seem to be extra spaces.
  // We pad n-gram orders with space to ensure that sorting on string order
  // coincides with numerical order (we don't allow n-gram order >99).

  PreArpaProcessor (int argc,
                    const char **argv) {
    assert(argc == 2);
    ReadVocabulary(argv[1]);
    ProcessNgramCountLines();
    ProcessNgrams();
  }

 private:
  // this function processes those lines of the input that start with " 0" and
  // represent the numbers of n-grams of different orders, e.g. the line "0 3
  // 5431423" means that there are 5431423 order-3 ngrams.  In order for this to
  // work in the pipeline where we're working from a split-up LM, we have to
  // aggregate counts across multiple lines, so for instance if we see two lines
  // "0 3 12" and "0 3 13", we add up 12 + 13 = 25 and output a line like
  // "ngram 3=25".
  void ProcessNgramCountLines() {
    int32 current_order = -1;
    int64 current_ngram_count = 0;
    std::cout << "\\data\\\n";
    while (1) {
      std::cin >> std::ws;  // eat up whitespace.
      int32 marker = -1, this_order = -1;
      if (std::cin.peek() == '0') {
        // this is a line with a number-of-ngrams, like "0 1 500".  at this
        // point we read in the 0 (which identifies it as a line that has a
        // number of ngrams), and the order (the 1 in the example).
        std::cin >> marker >> std::ws >> this_order;
        if (std::cin.fail() || marker != 0 || this_order < 1) {
          std::cerr << "pre-arpa-to-arpa, error at file position "
                    << std::cin.tellg() << ", expected int: marker = "
                    << marker << ", order = " << this_order << '\n';
          exit(1);
        }
      }
      if (this_order != current_order) {
        if (current_order != -1) {
          // we have now seen all the lines of a particular order, e.g.  all
          // lines of the form "0 2 X" (for order 2).  We need to print out the
          // 'ngram' line with the accumulated count.
          std::cout << "ngram " << current_order << '='
                  << current_ngram_count << '\n';
        }
        current_order = this_order;
        current_ngram_count = 0;
      }
      if (this_order == -1) {
        // the line does not begin with '0' (the peek() didn't find '0').  we're
        // done with the header, so return.  Before we return, unget the most
        // recent character which would have been a space.  Its absence would
        // confuse the main loop in ProcessNgrams(), which compares the strings
        // of successive lines.
        std::cin.unget();
        assert(std::cin.peek() == ' ');
        return;
      }
      int64 this_count = -1;
      std::cin >> this_count;
      if (std::cin.fail() || this_count < 0) {
        std::cerr << "pre-arpa-to-arpa, error at file position "
                  << std::cin.tellg() << ", expected int.";
        exit(1);
      }
      current_ngram_count += this_count;
    }
  }

  void ProcessNgrams() {
    std::string *vocab_data = &(vocab_[0]);
    int32 vocab_size = vocab_.size(),
        cur_order = -1;
    std::string line_str,
        extra_line_str;

    while (std::getline(std::cin, line_str)) {
      std::ostringstream words;  // will contain the words in the n-gram (in
                                 // text form), separated by spaces.
      const char *line = line_str.c_str();
      int32 order = strtol(line, const_cast<char**>(&line), 10);
      if (!(*line == ' ' && order > 0))
        goto fail;
      line++;  // consume the ' '
      if (order != cur_order) {
        // new order.  Print the separators in the ARPA file.
        // e.g. print "\n\\2-grams".
        std::cout << "\n\\" << order << "-grams:\n";
        cur_order = order;
      }
      // the next block prints out to a temporary string, the string form of
      // each of the words.  e.g. if this line is " 3 891 22 81 -4.43142", the
      // next block takes the entries for 891, 22 and 81 from the vocabulary
      // file and prints each of them into "words", separated by spaces.
      for (int32 i = 0; i < order; i++) {
        int32 word = strtol(line, const_cast<char**>(&line), 10);
        if ((*line != ' ' && *line != '\t') || word < 0)
          goto fail;
        if (word >= vocab_size) {
          std::cerr << "pre-arpa-to-arpa: word " << word << " is > "
                    << "the vocabulary size: line is " << line_str;
          exit(1);
        }
        words << vocab_data[word];
        if (i + 1 < order)
          words << ' ';
      }
      if (*line == ' ') {
        // We reach this position if we're processing an n-gram line that
        // was not preceded by a line showing the backoff probability
        // for that sequence.  In this case we just won't print out the
        // backoff log-prob, it defaults to zero if not printed.
        // E.g. at this point we might have line == " -1.84292",
        // and we print "-1.84292".
        std::cout << (line + 1) << '\t' << words.str() << "\n";
      } else {
        if (*line != '\t')
          goto fail;
        // we assume we're processing a line showing the backoff
        // probability (the tab identifies it as such; the tab also ensures
        // that the line is printed before the corresponding line
        // with the actual n-gram probability for the same sequence.
        // Before we output the backoff prob we want to output the
        // n-gram prob, and to do so we need to read the next line.

        // Handle an edge case, where the n-gram is "<s>".  In this
        // case there is a backoff probability but no direct n-gram
        // probability.  We print out -99 as the log-prob (this is
        // normal in ARPA-format LMs, e.g. SRILM does it).
        // The following code relies on the fact that kBosSymbol == 1,
        // documented in pocolm-types.h.
        if (order == 1 && !strncmp(line_str.c_str(), " 1 1\t", 5)) {
          std::cout << "-99\t" << words.str() << ' ' << (line + 1) << "\n";
          continue;
        }
        // Each line with a backoff prob (except the edge case with <s>)
        // should followed by a line with the same n-gram and the n-gram
        // prob.  We now read and try to parse that following line.
        std::getline(std::cin, extra_line_str);
        if (!std::cin.good()) {
          std::cerr << "pre-arpa-to-arpa: expected to read another line after "
                    << "this line [file truncated or bad counts?]: " << line_str;
          exit(1);
        }
        // As an example, imagine that
        // line_str == " 3 531 432 8901\t-1.43123"
        // extra_line_str == " 3 531 432 8901 -2.984312"
        size_t extra_line_size = extra_line_str.size(),
            this_line_consumed = line - line_str.c_str();
        // check that the initial parts of the lines are identical.
        if (extra_line_size <= this_line_consumed ||
            strncmp(line_str.c_str(), extra_line_str.c_str(),
                    this_line_consumed) != 0 ||
            extra_line_str[this_line_consumed] != ' ') {
          std::cerr << "pre-arpa-to-arpa: read confusing sequence of lines: '" << line_str
                    << "' followed by: '" << extra_line_str << "'... bad counts?\n";
          exit(1);
        }
        // extra_line_float will point to "-2.984312" in the example.  We print
        // this out.
        const char *extra_line_float = extra_line_str.c_str() +
            this_line_consumed + 1;
        std::cout << extra_line_float << '\t' << words.str();
        // the next line will print out "\t-1.43123\n" in the example,
        // which is the log-base-10 backoff prob.
        std::cout << line << "\n";
      }
      continue;
   fail:
      std::cerr << "pre-arpa-to-arpa: could not process line " << line << "\n";
      exit(1);
    }
    if (cur_order == -1) {
      std::cerr << "pre-arpa-to-arpa: read no input\n";
      exit(1);
    }
    // print a newline and \end\ to terminate the data section.
    std::cout << "\n\\end\\\n";
    if (std::cout.fail()) {
      std::cerr << "pre-arpa-to-arpa: failure to write output (disk full?)\n";
      exit(1);
    }
  }

  // this reads a file like 'words.txt', that should look a bit like
  // <eps> 0
  // <s> 1
  // </s> 2
  // <unk> 3
  // the 4
  // .. and so on.
  void ReadVocabulary(const char *vocab_filename) {
    std::ifstream vocab_stream(vocab_filename);
    if (vocab_stream.fail()) {
      std::cerr << "pre-arpa-to-arpa: error opening vocabulary file '"
                << vocab_filename << "'\n";
      exit(1);
    }
    std::string line;
    while (std::getline(vocab_stream, line)) {
      std::istringstream is(line);
      int32 i = -1;
      std::string word;
      // read 'word' then 'i' then eat up whitespace.
      // note: this approach should work for UTF-8 encoded text, as
      // (I believe) it's encoded in such a way that no character
      // could be interpreted as an (ASCII) space.
      is >> word >> i >> std::ws;
      is.peek();  // so it will register as EOF.
      if (i == -1 || !is.eof()) {
        std::cerr << "pre-arpa-to-arpa: could not interpret the following line "
                  << "(line " << (vocab_.size() + 1) << ") of the file "
                  << vocab_filename << ": " << line;
        exit(1);
      }
      if (static_cast<size_t>(i) != vocab_.size()) {
        std::cerr << "pre-arpa-to-arpa: expected the vocab file "
                  << vocab_filename << " to have lines in order: unexpected "
                  << (vocab_.size() + 1) << "'th line " << line;
        exit(1);
      }
      vocab_.push_back(word);
    }
  }

  // vocab_[i] is the printed form of symbol i, e.g. vocab_[3] = "<unk>"
  // normally.
  std::vector<std::string> vocab_;
};

}  // namespace pocolm


int main (int argc, const char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: pre-arpa-to-arpa <vocab-file>  < <pre-arpa-lines> > <arpa-file>\n"
              << "e.g.:  float-counts-to-pre-arpa 3 40000 <float.all | sort | \\\n"
              << "    pre-arpa-to-arpa words.txt | gzip -c > arpa.gz\n"
              << "Note: this program will also work if you start from several 'split' files\n"
              << "of a language model (float.all.1, float.all.2), split by most recent\n"
              << "history state, and do sort and merge-sort after float-counts-to-pre-arpa.\n";
    exit(1);
  }

  // everything happens in the constructor.
  pocolm::PreArpaProcessor processor(argc, argv);

  std::cerr << "pre-arpa-to-arpa: success\n";
}



