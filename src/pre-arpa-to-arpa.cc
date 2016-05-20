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
#include "pocolm-types.h"
#include "lm-state.h"


namespace pocolm {

/*
   This program is used together with float-counts-to-arpa to produce and ARPA
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
    ProcessInput();
  }

 private:
  // this function gets called if we find a line starting
  // with " 0".  It gets the rest of the line.  An example
  // would be a line " 0  3 54132143\n",
  // and what would get passed into this function would be the part
  // "  3 54132143\n".
  inline void ProcessNgramCountLine(const char *line_in) {
    const char *line = line_in;
    while (isspace(*line)) line++;
    int64 ngram_order = strtol(line, const_cast<char**>(&line), 10);
    if (*line != ' ' || ngram_order <= 0) {
      std::cerr << "pre-arpa-to-arpa: can't process line: 0 " << line_in;
      exit(1);
    }
    int64 num_ngrams = strtol(line, const_cast<char**>(&line), 10);
    if (*line != '\n' || num_ngrams < 0) {
      std::cerr << "pre-arpa-to-arpa: can't process line: 0 " << line_in;
      exit(1);
    }
    // produce a line like
    // 'ngram 1=40000'
    std::cout << "ngram " << ngram_order << "=" << num_ngrams << "\n";
  }

  void ProcessInput() {
    std::string *vocab_data = &(vocab_[0]);
    int32 vocab_size = vocab_.size(),
        cur_order = -1;
    std::cout << "\\data\\\n";
    std::string line_str,
        extra_line_str;
    while (std::getline(std::cin, line_str)) {
      const char *line = line_str.c_str();
      int32 order = strtol(line, const_cast<char**>(&line), 10);
      if (!(*line == ' ' && order >= 0))
        goto fail;
      line++;  // consume the ' '
      if (order == 0) {
        ProcessNgramCountLine(line);
        continue;
      }
      if (order != cur_order) {
        // new order.  Print the separators in the ARPA file.
        // e.g. print "\n\\2-grams".
        std::cout << "\n\\" << cur_order << "-grams:\n";
        cur_order = order;
      }
      // the next block prints out the string form of each of the words.
      // e.g. if this line is " 3 891 22 81 -4.43142", the next block takes the
      // entries for 891, 22 and 81 from the vocabulary file and prints each of
      // them out followed by a space.
      for (int32 i = 0; i < order; i++) {
        int32 word = strtol(line, const_cast<char**>(&line), 10);
        if ((*line != ' ' && *line != '\t') || word < 0)
          goto fail;
        if (word >= vocab_size) {
          std::cerr << "pre-arpa-to-arpa: word " << word << " is > "
                    << "the vocabulary size: line is " << line_str;
          exit(1);
        }
        std::cout << vocab_data[word] << ' ';
      }
      if (*line == ' ') {
        // We reach this position if we're processing an n-gram line that
        // was not preceded by a line showing the backoff probability
        // for that sequence.  In this case we just won't print out the
        // backoff log-prob, it defaults to zero if not printed.
        // E.g. at this point we might have line == " -1.84292\n",
        // and we print "-1.84292\n".
        std::cout << (line + 1);
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
        if (order == 1 && !strncmp(line_str.c_str(), " 1 1\t", 4)) {
          std::cout << " -99 " << (line + 1);
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
        // line_str == " 3 531 432 8901\t-1.43123\n"
        // extra_line_str == " 3 531 432 8901 -2.984312\n"
        size_t extra_line_size = extra_line_str.size(),
            this_line_consumed = line - line_str.c_str();
        // check that the initial parts of the lines are identical.
        if (extra_line_size <= this_line_consumed ||
            strncmp(line_str.c_str(), extra_line_str.c_str(),
                    this_line_consumed) != 0 ||
            extra_line_str[this_line_consumed] != ' ') {
          std::cerr << "pre-arpa-to-arpa: read confusing sequence of lines: " << line_str
                    << "followed by: " << extra_line_str << "... bad counts?\n";
          exit(1);
        }
        // extra_line_float will point to "-2.984312\n" in the example.  we work
        // out the length of the floating-point string- we need it in order to
        // avoid printing out the terminating newline, because after this we'll
        // print out the backoff prob and we don't want the line to end before
        // that.
        const char *extra_line_float = extra_line_str.c_str() +
            this_line_consumed + 1;
        size_t extra_line_float_length = extra_line_str.size() -
            this_line_consumed - 2;
        if (!(extra_line_float_length > 0)) {
          std::cerr << "pre-arpa-to-arpa: read confusing sequence of lines: " << line_str
                    << "followed by: " << extra_line_str << "... bad counts?\n";
          exit(1);
        }
        std::cout.write(extra_line_float, extra_line_float_length);
        // the next line will print out " -1.43123\n" in the example,
        // which is the log-base-10 backoff prob.
        std::cout << ' ' << (line + 1);
      }
   fail:
      std::cerr << "pre-arpa-to-arpa: could not process line " << line;
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
      std::cerr << "float-counts-to-arpa: error opening vocabulary file '"
                << vocab_filename << "'\n";
      exit(1);
    }
    std::string line;
    while (std::getline(vocab_stream, line)) {
      std::istringstream is(line);
      int32 i;
      std::string word;
      // read 'i' and 'word' then eat up whitespace.
      // note: this approach should work for UTF-8 encoded text, as
      // (I believe) it's encoded in such a way that no character
      // could be interpreted as an (ASCII) space.
      is >> i >> word >> std::ws;
      if (is.fail() || !is.eof()) {
        std::cerr << "float-counts-to-arpa: could not interpret the following line "
                  << "(line " << (vocab_.size() + 1) << ") of the file "
                  << vocab_filename << ": " << line;
        exit(1);
      }
      if (i != vocab_.size()) {
        std::cerr << "float-counts-to-arpa: expected the vocab file "
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
              << "    pre-arpa-to-arpa words.txt | gzip -c > arpa.gz\n";
    exit(1);
  }

  // everything happens in the constructor.
  pocolm::PreArpaProcessor processor(argc, argv);

  std::cerr << "pre-arpa-to-arpa: success\n";
}



