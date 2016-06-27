// merge-int-counts.cc

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
  This program reads multiple streams of int-counts, merge-sorts them, and
  writes them out as float counts.  It's like merge-counts except it both reads
  and writes int-counts (which are typically generated directly from text).
*/

namespace pocolm {

class IntCountMerger {
 public:
  IntCountMerger(int num_sources,
                 const char **source_names) {
    assert(num_sources > 0);
    Init(num_sources, source_names);
    while (!hist_to_sources_.empty())
      OutputState();

    std::cerr << "merge-int-counts: read ";
    int32 size = num_lm_states_read_.size();
    for (int32 i = 0; i < size; i++) {
      std::cerr << num_lm_states_read_[i];
      if (i < size - 1)
        std::cerr << " + ";
    }
    if (size != 1)
      std::cerr << " = " << std::accumulate(num_lm_states_read_.begin(),
                                            num_lm_states_read_.end(), 0.0);
    std::cerr << " LM states.\n";
  }
  ~IntCountMerger() {
    delete [] inputs_;
  }
 private:
  void Init(int32 num_sources,
            const char **source_names) {
    inputs_ = new std::ifstream[num_sources];
    int_lm_states_.resize(num_sources);
    num_lm_states_read_.resize(num_sources, 0);
    for (int32 i = 0; i < num_sources; i++) {
      inputs_[i].open(source_names[i], std::ios_base::binary|std::ios_base::in);
      if (inputs_[i].fail()) {
        std::cerr << "merge-int-counts: failed to open file '"
                  << source_names[i] << "' for reading\n";
        exit(1);
      }
      ReadStream(i);
    }
  }

  // Calling this function will attempt to read a new int-lm-state from source
  // stream i >= 0, and will update hist_to_sources_ as appropriate.
  void ReadStream(int32 i) {
    assert(static_cast<size_t>(i) < int_lm_states_.size());
    inputs_[i].peek();
    if (inputs_[i].eof())
      return;
    int_lm_states_[i].Read(inputs_[i]);
    num_lm_states_read_[i]++;
    hist_to_sources_[int_lm_states_[i].history].push_back(i);
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
    if (sources.size() == 1) {
      IntLmState &input = int_lm_states_[sources[0]];
      input.Write(std::cout);
    } else {
      // If there are multiple states with the same history, we add up
      // the counts concerned.
      std::vector<const IntLmState*> source_pointers;
      source_pointers.reserve(sources.size());
      for (std::vector<int32>::iterator iter = sources.begin();
           iter != sources.end(); ++iter) {
        int32 source = *iter;
        source_pointers.push_back(&(int_lm_states_[source]));
      }
      IntLmState merged_state;
      MergeIntLmStates(source_pointers, &merged_state);
      merged_state.Write(std::cout);
    }
    for (std::vector<int32>::const_iterator iter = sources.begin();
         iter != sources.end(); ++iter) {
      int32 s = *iter;
      ReadStream(s);
    }
  }

  std::ifstream *inputs_;

  // int_lm_states_, indexed by source, gives contains the LM-state
  // most recently read from each source.
  std::vector<IntLmState> int_lm_states_;

  std::vector<int64> num_lm_states_read_;

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
    std::cerr << "merge-int-counts: expected usage: <int-counts-file1> <int-counts-file2> .. \n"
              << " (it writes the merged int-counts to stdout).  For example:\n"
              << " merge-int-counts counts/1.int dir/counts/2.int | ...\n";
    exit(1);
  }

  // everything happens in the constructor of class IntCountMerger.
  pocolm::IntCountMerger merger(argc - 1, argv + 1);

  return 0;
}


/*

  ( echo 11 12 13; echo 11 12 13 14 ) | get-text-counts 3 | sort | uniq -c | get-int-counts /dev/null  /dev/null /dev/stdout  > foo
  print-int-counts <foo
#   [ 11 1 ]: 12->2
# [ 12 11 ]: 13->2
# [ 13 12 ]: 2->1 14->1
#[ 14 13 ]: 2->1
  merge-int-counts foo foo | merge-int-counts /dev/stdin foo | print-int-counts
# [ 11 1 ]: 12->6
# [ 12 11 ]: 13->6
# [ 13 12 ]: 2->3 14->3
# [ 14 13 ]: 2->3
 */
