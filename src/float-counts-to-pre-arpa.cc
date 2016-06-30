// float-counts-to-pre-arpa.cc

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

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <sstream>
#include <fstream>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include "pocolm-types.h"
#include "lm-state.h"


namespace pocolm {

class PreArpaGenerator {
 public:
  PreArpaGenerator(int argc, const char **argv) {
    // note: we checked the following condition already in main(), this
    // documents it locally.
    assert(argc == 4 || (argc == 5 && !strcmp(argv[1], "--no-unigram")));
    if (argc == 5) {
      print_unigrams_ = false;
      argc--;
      argv++;
    } else {
      print_unigrams_ = true;
    }
    order_ = ConvertToInt(argv[1]);
    num_words_ = ConvertToInt(argv[2]);
    assert(order_ >= 2 && num_words_ >= 4);
    lm_states_.resize(order_);
    num_ngrams_.resize(order_, 0);
    // we add one to the number of n-grams for order 1 (history-length 0),
    // because the BOS (<s>) will have its backoff printed although it has no
    // n-gram probability, which adds one line to the file that wouldn't
    // otherwise be counted.
    num_ngrams_[0] += 1;
    word_to_position_map_.resize((num_words_ + 1) * (order_ - 1));
    // set fill character to space and precision to 6.
    std::cout << std::setfill(' ') << std::setprecision(6);
    ProcessInput(argv[3]);
    OutputNumNgrams();
  }

  int32 ConvertToInt(const char *arg) {
    char *end;
    int32 ans = strtol(arg, &end, 10);
    if (end == arg || *end != '\0') {
      std::cerr << "float-counts-to-pre-arpa: command line: expected int, got '"
                << arg << "'\n";
      exit(1);
    }
    return ans;
  }

 private:

  void ProcessInput(const char *float_counts) {
    std::ifstream input;
    input.open(float_counts, std::ios_base::in|std::ios_base::binary);
    if (input.fail()) {
      std::cerr << "float-counts-to-pre-arpa: error opening float-counts file "
                << float_counts << "\n";
      exit(1);
    }
    while (input.peek(), !input.eof()) {
      FloatLmState lm_state;
      lm_state.Read(input);
      size_t hist_length = lm_state.history.size();
      assert(hist_length < lm_states_.size());
      lm_states_[hist_length].Swap(&lm_state);
      if (static_cast<int32>(hist_length) < order_ - 1)
        PopulateMap(hist_length);
      if (hist_length == 0)
        assert(lm_states_[0].total > 0 &&
               "Zero count for 1-gram history state (something went wrong?)");
      if (hist_length > 0 || print_unigrams_)
        OutputLmState(hist_length);
    }
  }


  void PopulateMap(int32 hist_length) {
    int32 pos = 0, num_words = num_words_;
    assert(word_to_position_map_.size() ==
           static_cast<size_t>((num_words + 1) * (order_ - 1)));
    int32 *map_data = &(word_to_position_map_[0]),
        orderm1 = order_ - 1;
    std::vector<std::pair<int32, float> >::const_iterator
        iter = lm_states_[hist_length].counts.begin(),
        end = lm_states_[hist_length].counts.end();
    for (pos = 0; iter != end; ++iter, pos++) {
      int32 word = iter->first,
          index = word * orderm1 + hist_length;
      assert(word > 0 && word <= num_words);
      map_data[index] = pos;
    }
  }


  // This function writes out the LM-state in lm_states_[hist_length].
  void OutputLmState(int32 hist_length) {
    CheckBackoffStatesExist(hist_length);
    int32 order = hist_length + 1;
    assert(order < 100 && "N-gram order cannot exceed 99.");
    const FloatLmState &lm_state = lm_states_[hist_length];
    const std::vector<int32> &history = lm_state.history;
    // 'prefix' will be something like:
    // ' 3 1842 46 ', consisting of the n-gram order and then
    // the history.
    std::ostringstream prefix;
    prefix << std::setfill(' ') << std::setw(2) << order << std::setw(0) << ' ';
    for (int32 j = hist_length - 1; j >= 0; j--) {
      // we need to go in reverse to get the history-state words into their
      // natural order.
      prefix << history[j] << ' ';
    }
    std::string prefix_str = prefix.str();

    std::vector<int32> reversed_history(lm_state.history);
    std::reverse(reversed_history.begin(), reversed_history.end());
    std::vector<std::pair<int32, float> >::const_iterator
        iter = lm_state.counts.begin(),
        end = lm_state.counts.end();
    float total_count = lm_state.total,
        discount_prob = lm_state.discount / total_count;
    for (; iter != end; ++iter) {
      int32 word = iter->first;
      float prob = iter->second / total_count;
      if (hist_length > 0)
        prob += discount_prob * GetProbability(hist_length - 1, word);
      float log10_prob = log10f(prob);
      assert(log10_prob - log10_prob == 0.0);  // check for NaN/inf.
      std::cout << prefix_str << word << ' ' << log10_prob << '\n';
    }
    num_ngrams_[hist_length] += static_cast<int64>(lm_state.counts.size());
    if (hist_length > 0) {
      // print the backoff prob for this history state.
      std::cout << std::setw(2) << hist_length << std::setw(0);
      for (int32 j = hist_length - 1; j >= 0; j--)
        std::cout << ' ' << history[j];
      float log10_backoff_prob = log10f(discount_prob);
      // we use tab instead of space just before the backoff prob...
      // this ensures that the backoff prob precedes the same-named
      // n-gram prob
      std::cout << '\t' << log10_backoff_prob << "\n";
    }
  }

  // this function returns the count of word 'word' in the currently cached
  // LM-state whose history has length 'hist_length'.
  inline float GetCountForWord(int32 hist_length, int32 word) const {
    assert(word > 0 && word <= num_words_);
    size_t pos = word_to_position_map_[word * (order_ - 1) + hist_length];
    if (pos < lm_states_[hist_length].counts.size() &&
        lm_states_[hist_length].counts[pos].first == word) {
      return lm_states_[hist_length].counts[pos].second;
    } else {
      if (hist_length == 0) {
        std::cerr << "word " << word
                  << "has zero count in unigram counts.";
        exit(1);
      }
      // we allow the count to be zero for orders >0, because
      // it might be possible that we'd prune away a lower-order
      // count while keeping a higher-order one.
      return 0.0;
    }
  }

  // This function gets the probability (not log-prob) of word indexed 'word'
  // given the lm-state of history-length 'hist_length', including backoff.
  // It will crash if the word is not in that history state.
  // We only call this function when handling backoff.  This function
  // does not work for the highest-order history state (order_ - 1).
  inline float GetProbability(int32 hist_length, int32 word) const {
    assert(hist_length < order_ - 1);

    float numerator = GetCountForWord(hist_length, word);
    if (hist_length > 0)
      numerator += lm_states_[hist_length].discount *
          GetProbability(hist_length - 1, word);
    return numerator / lm_states_[hist_length].total;
  }

  // this function checks that the states in lm_states_[i] for
  // i < hist_length are for histories that are the backoff histories
  // of the state in lm_states_[hist_length].
  void CheckBackoffStatesExist(int32 hist_length) const {
    for (int32 i = 1; i < hist_length; i++) {
      assert(static_cast<int32>(lm_states_[i].history.size()) == i);
      assert(std::equal(lm_states_[i].history.begin(),
                        lm_states_[i].history.end(),
                        lm_states_[hist_length].history.begin()));
    }
  }

  void OutputNumNgrams() const {
    assert(static_cast<int32>(num_ngrams_.size()) == order_);
    std::cerr << "float-counts-to-pre-arpa: output [ ";
    for (int32 order = (print_unigrams_ ? 1 : 2); order <= order_; order++) {
      // output will be something like: " 0  3 43142".
      // the "0" will be interpreted by pre-arpa-to-arpa- it tells it
      // that the rest of the line says the number of n-grams for some order.
      // The padding with space is there to ensure that string order and
      // numeric order coincide, up to n-gram order = 99
      std::cout << std::setw(2) << 0 << ' ' << std::setw(2) << order << ' '
                << num_ngrams_[order-1] << '\n';
      std::cerr << num_ngrams_[order-1] << ' ';
    }
    std::cerr << "] n-grams\n";
  }

  // the n-gram order of the LM we're writing.
  int32 order_;

  // the size of the vocabulary, excluding epsilon (equals highest-numbered word).
  int32 num_words_;

  // this vector is indexed by n-gram order - 1 (i.e. by history length).
  std::vector<int64> num_ngrams_;

  // lm-states, indexed by history length.
  std::vector<FloatLmState> lm_states_;

  // This maps from word-index to the position in the 'counts' vectors of the LM
  // states.  It exists to help us do rapid lookup of counts in lower-order
  // states when we are doing the backoff computation.  For history-length 0 <=
  // hist_len < order - 1 and word 0 < word <= num_words_,
  // word_to_position_map_[(order - 1) * word + hist_len] contains the position
  // in lm_states_[hist_len].counts that this word exists.  Note: for words that
  // are not in that counts vector, the value is undefined.
  std::vector<int32> word_to_position_map_;

  // will be false if --no-unigram option was used.
  bool print_unigrams_;
};

}  // namespace pocolm


int main (int argc, const char **argv) {
  if (!(argc == 4 || (argc == 5 && !strcmp(argv[1], "--no-unigram")))) {
    std::cerr << "Usage: float-counts-to-pre-arpa [--no-unigram] <ngram-order> <num-words> <float-counts>  > <pre-arpa-out>\n"
              << "E.g. float-counts-to-pre-arpa 3 40000 float.all | LC_ALL=C sort | pre-arpa-to-pre-arpa words.txt > arpa"
              << "The output is in text form, with lines of the following types:\n"
              << "N-gram probability lines: <n-gram-order> <word1> ... <wordN> <log10-prob>, e.g.:\n"
              << " 3 162 82 978 -1.72432\n"
              << "Backoff probability lines: <n-gram-order> <word1> ... <wordN> b<log10-prob>, e.g:\n"
              << " 3 162 82 978 b-1.72432\n"
              << "Lines (beginning with 0) that announce the counts of n-grams for a\n"
              << " particular n-gram order, e.g.:\n"
              << " 0 3 894121\n"
              << "announces that there are 894121 3-grams.  (we print leading spaces\n"
              << "so that string order will coincide with numeric order).  These will be processed\n"
              << "into the arpa header.\n"
              << "The output of this program will be sorted and then piped into\n"
              << "pre-arpa-to-arpa.\n";
    exit(1);
  }

  // everything gets called from the constructor.
  pocolm::PreArpaGenerator generator(argc, argv);

  return 0;
}

// see discount-counts.cc for a command-line example that was used to test this.

