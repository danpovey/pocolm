// merge-counts.cc

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
#include "lm-state.h"


/**
  This program reads both int-counts and regular counts, merges them as regular
  counts, and writes the merged counts. */

namespace pocolm {

class CountMerger {
 public:
  CountMerger(int num_sources,
              const char **source_names): num_lm_states_written_(0) {
    assert(num_sources > 0);
    Init(num_sources, source_names);
    while (!hist_to_sources_.empty())
      OutputState();
    std::cerr << "merge-counts: wrote " << num_lm_states_written_
              << " LM states.\n";
  }
  ~CountMerger() {
    delete [] inputs_;
  }
 private:
  void Init(int32 num_sources,
            const char **source_names) {
    inputs_ = new std::ifstream[num_sources];
    scales_.resize(num_sources);
    int_lm_states_.resize(num_sources);
    general_lm_states_.resize(num_sources);
    for (int32 i = 0; i < num_sources; i++) {
      std::string name (source_names[i]);
      float scale = -1;
      ssize_t pos = name.find_first_of(',');
      if (pos >= 0) {
        char *endptr = NULL;
        scale = strtod(name.c_str() + pos + 1, &endptr);
        if (!(scale >= 0.0) && *endptr == '\0') {
          std::cerr << "merge-counts: bad command line argument '"
                    << source_names[i] << "'\n";
          exit(1);
        }
        name.resize(pos);
      }
      scales_[i] = scale;
      inputs_[i].open(name.c_str(), std::ios_base::binary|std::ios_base::in);
      if (inputs_[i].fail()) {
        std::cerr << "merge-counts: failed to open file '"
                  << name << "' for reading\n";
        exit(1);
      }
      ReadStream(i);
    }
  }

  // Calling this function will attempt to read a new lm-state from source
  // stream i, and will update hist_to_sources_ as appropriate.
  void ReadStream(int32 i) {
    assert(static_cast<size_t>(i) < scales_.size());
    inputs_[i].peek();
    if (inputs_[i].eof())
      return;
    const std::vector<int32> *this_hist;
    if (scales_[i] == -1) {
      general_lm_states_[i].Read(inputs_[i]);
      this_hist = &(general_lm_states_[i].history);
    } else {
      int_lm_states_[i].Read(inputs_[i]);
      this_hist = &(int_lm_states_[i].history);
    }
    hist_to_sources_[*this_hist].push_back(i);
  }

  // This function, which expects hist_to_sources_ to be nonempty, takes the
  // (lexicographically) first history state in hist_to_sources_, combines the
  // counts across all the inputs, and writes it to the standard output.
  void OutputState() {
    assert(!hist_to_sources_.empty());
    std::vector<int32> hist = hist_to_sources_.begin()->first,
        sources = hist_to_sources_.begin()->second;
    hist_to_sources_.erase(hist_to_sources_.begin());
    num_lm_states_written_++;
    if (sources.size() == 1 &&
        scales_[sources[0]] == -1) {
      GeneralLmState &input = general_lm_states_[sources[0]];
      input.Write(std::cout);
    } else {
      GeneralLmStateBuilder &builder = builder_;
      builder.Clear();
      for (std::vector<int32>::const_iterator iter = sources.begin();
           iter != sources.end(); ++iter) {
        int32 s = *iter;
        if (scales_[s] == -1)
          builder.AddCounts(general_lm_states_[s]);
        else
          builder.AddCounts(int_lm_states_[s], scales_[s]);
      }
      GeneralLmState output_lm_state;
      builder.Output(&(output_lm_state.counts));
      output_lm_state.history = hist;
      output_lm_state.Write(std::cout);
    }
    for (std::vector<int32>::const_iterator iter = sources.begin();
         iter != sources.end(); ++iter) {
      int32 s = *iter;
      ReadStream(s);
    }
  }


  // This vector contains the scale for each source, or -1 if the
  // source has no scale.
  std::vector<float> scales_;

  std::ifstream *inputs_;

  // int_lm_states_, indexed by source, is only active for
  // i such that scales_[i] != -1.
  std::vector<IntLmState> int_lm_states_;
  // general_lm_states_, indexed by source, is only active for
  // i such that scales_[i] == -1.
  std::vector<GeneralLmState> general_lm_states_;

  // this a temporary variable used in a function, but we
  // declare it here to avoid reallocation.
  GeneralLmStateBuilder builder_;



  // This is a map from the history vector to the list of source indexes that
  // currently have an LM-state with that history-vector, that needs to be
  // processed.
  std::map<std::vector<int32>, std::vector<int32> > hist_to_sources_;

  int64 num_lm_states_written_;
};

}  // namespace pocolm

int main (int argc, const char **argv) {
  if (argc <= 1) {
    std::cerr << "merge-counts: expected usage: <counts-file1>[,scale1] <counts-file2>[,scale1] ...\n"
              << " (it writes the merged counts to stdout).  For example:\n"
              << " merge-counts dir/src1/3.ngram,1.0  dir/src2/3.ngram,1.0 dir/discounts/3.ngram | ... \n"
              << "Filename arguments that have a scale attached to them are expected to be\n"
              << "int-counts (as written by get-int-counts); filename arguments without such\n"
              << "a scale are expected to be general counts.\n";
    exit(1);
  }

  // everything happens in the constructor of class merger.
  pocolm::CountMerger merger(argc - 1, argv + 1);

  return 0;
}


/*

  some testing:
( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/stdout /dev/null | print-int-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
 [ 1 ]: 11->2
 print-int-counts: printed 1 LM states, with 1 individual n-grams.

 ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout | print-int-counts
get-int-counts: processed 5 LM states, with 6 individual n-grams.
 [ 11 1 ]: 12->2
 [ 12 11 ]: 13->2
 [ 13 12 ]: 2->1 14->1
 [ 14 13 ]: 2->1
print-int-counts: printed 4 LM states, with 5 individual n-grams.

 */
