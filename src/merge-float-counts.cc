// merge-float-counts.cc

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
#include <numeric>
#include <stdlib.h>
#include "pocolm-types.h"
#include "lm-state.h"


/**
  This program reads multiple streams of float counts, merge-sorts them, and
  writes them out as float counts.  It's like merge-counts except it both reads
  and writes float-counts (which are typically the final product of
  discounting).
  Currently it assumes that all the counts being merged are from distinct
  n-gram histories (e.g. counts of different orders).  Later we can extend this
  program if needed to handle the general case.
*/

namespace pocolm {

class FloatCountMerger {
 public:
  FloatCountMerger(int num_sources,
                   const char **source_names) {
    assert(num_sources > 0);
    Init(num_sources, source_names);
    while (!hist_to_sources_.empty())
      OutputState();

    std::cerr << "merge-float-counts: read ";
    int32 size = num_lm_states_read_.size();
    for (int32 i = 0; i < size; i++) {
      std::cerr << num_lm_states_read_[i];
      if (i < size - 1)
        std::cerr << " + ";
    }
    if (size != 1)
      std::cerr << " = " << std::accumulate(num_lm_states_read_.begin(),
                                            num_lm_states_read_.end(), 0);
    std::cerr << " LM states.";
    std::cerr << " Write ";
    size = num_ngrams_write_.size();
    for (int32 i = 0; i < size; i++) {
      std::cerr << num_ngrams_write_[i];
      if (i < size - 1)
        std::cerr << " + ";
    }
    if (size != 1)
      std::cerr << " = " << std::accumulate(num_ngrams_write_.begin(),
                                            num_ngrams_write_.end(), 0);
    std::cerr << " individual n-grams.\n";
  }
  ~FloatCountMerger() {
    delete [] inputs_;
  }
 private:
  void Init(int32 num_sources,
            const char **source_names) {
    inputs_ = new std::ifstream[num_sources];
    float_lm_states_.resize(num_sources);
    num_lm_states_read_.resize(num_sources, 0);
    for (int32 i = 0; i < num_sources; i++) {
      inputs_[i].open(source_names[i], std::ios_base::binary|std::ios_base::in);
      if (inputs_[i].fail()) {
        std::cerr << "merge-float-counts: failed to open file '"
                  << source_names[i] << "' for reading\n";
        exit(1);
      }
      ReadStream(i);
    }
  }

  // Calling this function will attempt to read a new float-lm-state from source
  // stream i >= 0, and will update hist_to_sources_ as appropriate.
  void ReadStream(int32 i) {
    assert(static_cast<size_t>(i) < float_lm_states_.size());
    inputs_[i].peek();
    if (inputs_[i].eof())
      return;
    float_lm_states_[i].Read(inputs_[i]);
    num_lm_states_read_[i]++;
    hist_to_sources_[float_lm_states_[i].history].push_back(i);
  }

  // This function, which expects hist_to_sources_ to be nonempty, takes the
  // (lexicographically) first history state in hist_to_sources_, checks that it
  // exists for only one input (since we don't yet support combining these
  // history states), and writes it to the standard output.
  void OutputState() {
    assert(!hist_to_sources_.empty());
    std::vector<int32> hist = hist_to_sources_.begin()->first,
        sources = hist_to_sources_.begin()->second;
    hist_to_sources_.erase(hist_to_sources_.begin());
    if (hist.size() + 1> num_ngrams_write_.size()) {
        num_ngrams_write_.resize(hist.size() + 1, 0);
    }
    if (sources.size() == 1) {
      FloatLmState &input = float_lm_states_[sources[0]];
      num_ngrams_write_[hist.size()] += input.counts.size();
      input.Write(std::cout);
    } else {
      // If there are multiple states with the same history, we currently
      // assume that they are duplicates of the same state, and that only
      // one of the duplicates should be written out.  (this situation
      // arises when this program is called from get_objf_and_derivs_split.py,
      // where the order-1 LM state is duplicated across splits).
      int32 size = sources.size();
      assert(size > 1);
      FloatLmState &input = float_lm_states_[sources[0]];
      for (int32 n = 1; n < size; n++) {
        FloatLmState &other_input = float_lm_states_[sources[n]];
        assert(other_input.counts == input.counts && "merge-float-counts: "
               "multiple inputs have the same history state but the counts are "
               "not identical.");
      }
      num_ngrams_write_[hist.size() - 1] += input.counts.size();
      input.Write(std::cout);
    }
    for (std::vector<int32>::const_iterator iter = sources.begin();
         iter != sources.end(); ++iter) {
      int32 s = *iter;
      ReadStream(s);
    }
  }

  std::ifstream *inputs_;

  // float_lm_states_, indexed by source, gives contains the LM-state
  // most recently read from each source.
  std::vector<FloatLmState> float_lm_states_;

  std::vector<int64> num_lm_states_read_;

  // num_ngrams written for each order
  std::vector<int64> num_ngrams_write_;

  // This is a map from the history vector to the list of source indexes that
  // currently have an LM-state with that history-vector, that needs to be
  // processed.  Currently we assume that the list of source indexes (i.e. the
  // value) always has exactly one element, since we assume the history-states
  // being merged are always distinct, but we could later extend the code.
  std::map<std::vector<int32>, std::vector<int32> > hist_to_sources_;

};

}  // namespace pocolm

int main (int argc, const char **argv) {
  if (argc <= 1) {
    std::cerr << "merge-float-counts: expected usage: <float-counts-file1> <float-counts-file2> .. \n"
              << " (it writes the merged float-counts to stdout).  For example:\n"
              << " merge-float-counts dir/discounted/1.ngram dir/discounted/2.ngram | ...\n"
              << "This program currently assumes that the LM-states to be merged always\n"
              << "either have distinct histories (in which case no real merging is done\n"
              << "at the LM-state level), or have the same histories but are identical,\n"
              << "in which only one of the identical LM-states are written out.\n";
    exit(1);
  }

  // everything happens in the constructor of class FloatCountMerger.
  pocolm::FloatCountMerger merger(argc - 1, argv + 1);

  return 0;
}


/*
  we use this in the testing of compute-probs.cc, see test example there.

 */
