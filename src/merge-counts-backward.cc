// merge-counts-backward.cc

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
#include "lm-state-derivs.h"


/**
  This program reads both int-counts and regular counts, merges them as regular
  counts, and writes the merged counts. */

namespace pocolm {

class CountMergerBackward {
 public:
  CountMergerBackward(int argc,
                      const char **argv): num_lm_states_processed_(0) {
    Init(argc, argv);
    while (!hist_to_sources_.empty())
      ProcessState();
    std::cerr << "merge-counts: processed " << num_lm_states_processed_
              << " LM states.\n";
    FinalizeOutput();
  }

 private:
  void Init(int argc,
            const char **argv) {
    assert(argc % 2 == 1);
    int num_sources = (argc - 3) / 2;
    scales_.resize(num_sources);
    count_inputs_ = new std::ifstream[num_sources];
    deriv_outputs_ = new std::ofstream[num_sources];
    scale_derivs_.resize(num_sources, 0.0);
    int_lm_states_.resize(num_sources);
    general_lm_states_.resize(num_sources);

    OpenStream(argv[1], &merged_count_input_);
    OpenStream(argv[2], &merged_deriv_input_);

    for (int32 i = 0; i < num_sources; i++) {
      const char *count_filename = argv[3 + i*2],
          *scale_or_deriv_filename = argv[4 + i*2];
      OpenStream(count_filename, count_inputs_ + i);
      char *endptr = NULL;
      float scale = strtod(scale_or_deriv_filename, &endptr);
      if (*endptr != '\0') {
        // not a valid floating-point value: assume it was the filename of a
        // derivative.
        scales_[i] = -1;
        OpenStream(scale_or_deriv_filename, deriv_outputs_ + i);
      } else {
        scales_[i] = scale;
        if (!(scale > 0.0)) {
          std::cerr << "merge-counts: bad scale " << scale << "\n";
          exit(1);
        }
      }
      // this buffers the reading for the i'th stream.
      ReadStream(i);
    }
  }

  void FinalizeOutput() {
    delete [] count_inputs_;
    for (size_t i = 0; i < scales_.size(); i++) {
      if (scales_[i] == -1) {
        deriv_outputs_[i].close();
        if (deriv_outputs_[i].fail()) {
          std::cerr << "merge-counts-backward: error closing stream (disk full?)\n";
          exit(1);
        }
      } else {
        std::cout << scale_derivs_[i] << " ";
      }
    }
    std::cout << "\n";
    delete [] deriv_outputs_;
  }

  void OpenStream(const char *filename,
                  std::ifstream *stream) {
    stream->open(filename, std::ios_base::binary|std::ios_base::in);
    if (!stream->is_open()) {
      std::cerr << "discount-counts-backward: failed to open '"
                << filename << "' for reading\n";
      exit(1);
    }
  }

  void OpenStream(const char *filename,
                  std::ofstream *stream) {
    stream->open(filename, std::ios_base::binary|std::ios_base::out);
    if (!stream->is_open()) {
      std::cerr << "discount-counts-backward: failed to open '"
                << filename << "' for writing\n";
      exit(1);
    }
  }

  // Calling this function will attempt to read a new lm-state from source
  // stream i, and will update hist_to_sources_ as appropriate.  [if it can't
  // read due to EOF, it will clear the lm-state].
  void ReadStream(int32 i) {
    assert(static_cast<size_t>(i) < scales_.size());
    if (scales_[i] == -1 && !general_lm_states_[i].counts.empty()) {
      general_lm_states_[i].WriteDerivs(deriv_outputs_[i]);
      general_lm_states_[i].counts.clear();
    }
    count_inputs_[i].peek();
    if (count_inputs_[i].eof())
      return;
    const std::vector<int32> *this_hist;
    if (scales_[i] == -1) {
      general_lm_states_[i].Read(count_inputs_[i]);
      this_hist = &(general_lm_states_[i].history);
    } else {
      int_lm_states_[i].Read(count_inputs_[i]);
      this_hist = &(int_lm_states_[i].history);
    }
    hist_to_sources_[*this_hist].push_back(i);
  }

  // This function, which expects hist_to_sources_ to be nonempty, processes the
  // (lexicographically) first history state in hist_to_sources_.  It expects
  // to read in a state from <merged-counts-file> that corresponds to the
  // output of previously doing this; it creates derivatives for that
  // state and writes them out.
  void ProcessState() {
    assert(!hist_to_sources_.empty());
    num_lm_states_processed_++;
    std::vector<int32> hist = hist_to_sources_.begin()->first,
        sources = hist_to_sources_.begin()->second;
    hist_to_sources_.erase(hist_to_sources_.begin());

    if (sources.size() == 1 &&
        scales_[sources[0]] == -1) {
      GeneralLmStateDerivs &input = general_lm_states_[sources[0]];
      // the next line is just to read the data from the stream so we skip
      // over it.  We could do the same more efficiently with fseek().
      merged_state_.Read(merged_count_input_);
      input.ReadDerivs(merged_deriv_input_);
      assert(merged_state_.counts.size() == input.counts.size() &&
             merged_state_.history == hist);
    } else {
      merged_state_.Read(merged_count_input_);
      merged_state_.ReadDerivs(merged_deriv_input_);
      assert(merged_state_.history == hist && "mismatched data?");
      PopulateWordMap();

      for (std::vector<int32>::const_iterator iter = sources.begin();
           iter != sources.end(); ++iter) {
        int32 s = *iter;
        if (scales_[s] == -1)
          ProcessSourceGeneral(s);
        else
          ProcessSourceInt(s);
      }
    }
    for (std::vector<int32>::const_iterator iter = sources.begin();
         iter != sources.end(); ++iter) {
      int32 s = *iter;
      ReadStream(s);
    }
  }

  // This propagates the derivative back from the merged-counts to the i'th
  // source; this version is called if the i'th source is a GeneralLmState.
  void ProcessSourceGeneral(int32 i) {
    GeneralLmStateDerivs &source_state = general_lm_states_[i];
    std::vector<std::pair<int32, Count> >::const_iterator iter =
        source_state.counts.begin(), end = source_state.counts.end();
    std::vector<Count>::iterator deriv_iter = source_state.count_derivs.begin();
    for (; iter != end; ++iter, ++deriv_iter) {
      int32 word = iter->first;
      const Count &count = iter->second;
      Count &deriv = *deriv_iter;
      assert(static_cast<size_t>(word) < word_map_.size());
      int32 pos = word_map_[word];
      assert(merged_state_.counts[pos].first == word);
      const Count &merged_count = merged_state_.counts[pos].second;
      Count &merged_deriv = merged_state_.count_derivs[pos];
      // the following will backprop the derivative from 'merged_deriv'
      // to 'deriv'.
      merged_count.AddBackward(count, &merged_deriv, &deriv);
    }
  }

  // This propagates the derivative back from the merged-counts to the i'th
  // source; this version is called if the i'th source is an IntLmState.
  void ProcessSourceInt(int32 i) {
    float scale = scales_[i];
    double scale_deriv = 0.0;

    IntLmState &source_state = int_lm_states_[i];
    std::vector<std::pair<int32, int32> >::const_iterator iter =
        source_state.counts.begin(), end = source_state.counts.end();
    for (; iter != end; ++iter) {
      int32 word = iter->first;
      int32 num_words = iter->second;
      assert(static_cast<size_t>(word) < word_map_.size());
      int32 pos = word_map_[word];
      assert(merged_state_.counts[pos].first == word);
      const Count &merged_count = merged_state_.counts[pos].second;
      Count &merged_deriv = merged_state_.count_derivs[pos];
      // the following will backprop the derivative from 'merged_deriv'
      // to 'scale_deriv'.
      merged_count.AddBackward(scale, num_words, &merged_deriv, &scale_deriv);
    }
    scale_derivs_[i] += scale_deriv;
  }

  // sets up word_map_ so that we can look up for any word in merged_state_.counts,
  // the position in that vector.
  void PopulateWordMap() {
    int32 word_map_size = word_map_.size(),
        pos = 0;
    for (std::vector<std::pair<int32, Count> >::iterator iter =
             merged_state_.counts.begin(); iter !=
             merged_state_.counts.end(); ++iter,++pos) {
      int32 word = iter->first;
      if (word >= word_map_size) {
        // there is no need to populate it with any particular value; the values
        // of word_map_[i] for words i not in the current merged_state_ are
        // undefined.
        word_map_size = word + 1;
        word_map_.resize(word_map_size);
        word_map_[word] = pos;
      }
    }
  }


  // This vector contains the scale for each source (for int-count inputs), or
  // -1 if the source has no scale (for sources of general-count type).
  std::vector<float> scales_;

  // an input stream for each of the inputs (same dim as scales_ etc.),
  // to read counts.
  std::ifstream *count_inputs_;
  // an output stream for the derivs w.r.t. each of the inputs of 'general'
  // count type (only valid for indexes i with scales_[i] == -1).
  std::ofstream *deriv_outputs_;

  // derivatives w.r.t. the scales are accumulated here; only valid for indexes
  // i with scales_[i] != -1.
  std::vector<double> scale_derivs_;

  // input stream for the merged language-model states that we
  // previously wrote out in the forward pass.
  std::ifstream merged_count_input_;
  // input stream for the derivatives w.r.t. the merged LM states
  std::ifstream merged_deriv_input_;

  // int_lm_states_, indexed by source, is only active for
  // i such that scales_[i] != -1.
  std::vector<IntLmState> int_lm_states_;
  // general_lm_states_, indexed by source, is only active for
  // i such that scales_[i] == -1.  We create the derivatives
  // here and write them to disk.
  std::vector<GeneralLmStateDerivs> general_lm_states_;

  // The merge LM-state with its derivatives (both read from disk); we declare
  // it at the class level to avoid repeated allocation/deallocation of the
  // vector, but it could just as well be a local variable.
  GeneralLmStateDerivs merged_state_;

  // word_map_ is a map from word-index to the index into the
  // count vectors in 'merged_lm_state_'.  It's only valid for words
  // that are present in the merged LM state (other entries
  // are undefined).
  std::vector<int32> word_map_;

  // This is a map from the history vector to the list of source indexes that
  // currently have an LM-state with that history-vector, that needs to be
  // processed.
  std::map<std::vector<int32>, std::vector<int32> > hist_to_sources_;

  int64 num_lm_states_processed_;
};

}  // namespace pocolm

int main (int argc, const char **argv) {
  if (argc < 5 || (argc % 2 != 1)) {
    std::cerr << "merge-counts-backward: expected usage:\n"
              << "merge-counts-backward <merged-counts-file> <merged-derivs-file>\\\n"
              << "   <counts-file1> (<scale1>|<deriv-file1>) \\\n"
              << "   <counts-file2> (<scale2>|<deriv-file2>) ...\n"
              << " For inputs <counts-fileX> corresponding to general counts, the\n"
              << " outputs are written to the specified files; for those corresponding\n"
              << " to int counts, the derivatives w.r.t. the scaling factors are\n"
              << " computed and they are all written to a single line of the standard\n"
              << " output.\n";
    exit(1);
  }

  // everything happens in the constructor of class merger.
  pocolm::CountMergerBackward merger(argc, argv);

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
