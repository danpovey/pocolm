// lm-state.h

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

#ifndef POCOLM_LM_STATE_H_
#define POCOLM_LM_STATE_H_

#include <iostream>
#include <vector>
#include <utility>
#include "pocolm-types.h"
#include "count.h"

namespace pocolm {



/**
   This class is used to store the count information we have in a language-model
   state-- for a single data-source, prior to any smoothing, weighting, or interpolation.
*/

class IntLmState {
 public:
  // Reversed history, e.g. count of "a b c" would have c as 'next-word', and [
  // b a ] as 'history'.
  std::vector<int32> history;

  // These counts are pairs of (next-word, count).  These are assumed to always
  // be sorted on the next-word.  We don't have to explicitly do the sorting,
  // because in the pipeline we use, 'sort' does it for us.
  std::vector<std::pair<int32, int32> > counts;

  void Init(const std::vector<int32> &h) {
    history = h;
    counts.clear();
  }

  // Adds count for this word.  This is assumed to be called in order
  // from lowest to highest word, without repeats... in the pipelihe
  // we use, this is done for us by 'sort -c'.  We later check
  // this (occasionally) inside Write().
  void AddCount(int32 word, int32 count) {
    counts.push_back(std::pair<int32,int32>(word, count));
  }

  // writes to the ostream.  Throws on error.
  void Write(std::ostream &os) const;

  // prints in text form to the ostream (for debug- the output is not computer readable).
  void Print(std::ostream &os) const;

  // reads from the istream, which is assumed to not be at EOF.
  // Throws on error.
  void Read(std::istream &is);

  // checks the data for validity, dies if that fails.
  void Check() const;
};


/**
   This class is used for storing stats that have been discounted, e.g
   3-gram stats from which the discount amount has been removed.  We store
   the total-counts for each word as floats, and we also store the total
   count for the state and the discounted count (i.e. the amount that
   was removed during discounting, and which will determine the
   backoff weight).
 */
class FloatLmState {
 public:
  // Reversed history, e.g. count of "a b c" would have c as 'next-word', and [
  // b a ] as 'history'.
  std::vector<int32> history;

  // The total count for this state, it's equal to discounted_amount plus the
  // sum of the .second elements of 'counts'.  We treat this as a derived
  // variable...  note, we have to decide what to do about derivatives; most
  // likely we will make sure that the 'total' is zero in the derivatives when
  // we write them to disk.
  float total;
  // The total discount amount for this state-- it equals the amount that was
  // removed via discounting (but it's zero for the unigram state).
  float discount;
  // A vector of pairs (next-word, discounted-count-for-that-word), sorted
  // on word.
  std::vector<std::pair<int32, float> > counts;

  // writes to the ostream.  Throws on error.
  void Write(std::ostream &os) const;

  // prints in text form to the ostream (for debug- the output is not computer readable).
  void Print(std::ostream &os) const;

  // reads from the istream, which is assumed to not be at EOF.
  // Throws on error.
  void Read(std::istream &is);

  // checks the data for validity, dies if that fails.
  void Check() const;

  void Swap(FloatLmState *other) {
    history.swap(other->history);
    std::swap(total, other->total);
    std::swap(discount, other->discount);
    counts.swap(other->counts);
  }
  // this sets 'total' to the sum of all the counts plus 'discount'; it's
  // currently only called from 'perturb-float-counts'.
  void ComputeTotal();
};


/**
   This class is the general case of storing counts for an LM state, in which we
   might have done weighting, smoothing and interpolation.  Unlike IntLmState,
   we assume that the individual counts might be of different size (due to
   weighting and discounting), and we use class Count to keep track of the total
   and of the 3 largest individual counts in the set.
 */
class GeneralLmState {
 public:
  // Reversed history, e.g. count of "a b c" would have c as 'next-word', and [
  // b a ] as 'history'.
  std::vector<int32> history;

  // These counts are pairs of (next-word, count).  These are assumed to always
  // be sorted on the next-word.  We don't have to explicitly do the sorting,
  // because in the pipeline we use, 'sort' does it for us.
  std::vector<std::pair<int32, Count> > counts;

  // writes to the ostream.  Throws on error.
  void Write(std::ostream &os) const;

  // prints in text form to the ostream (for debug- the output is not computer readable).
  void Print(std::ostream &os) const;

  // reads from the istream, which is assumed to not be at EOF.
  // Throws on error.
  void Read(std::istream &is);

  // checks the data for validity, dies if that fails.
  void Check() const;

  void Swap(GeneralLmState *other);
};

/* This class is used in building a GeneralLmState; it allows you to efficiently
   accumulate the counts without requiring things to be added in the correct
   order.
 */
class GeneralLmStateBuilder {
 public:

  unordered_map<int32, int32> word_to_pos;
  std::vector<Count> counts;

  void Clear();

  // Add an individual count of type float
  void AddCount(int32 word, float count);
  // Add an individual count of the form (scale, number-of-pieces)
  void AddCount(int32 word, float scale, int32 num_pieces);
  // Add counts from an IntLmState.
  void AddCounts(const IntLmState &lm_state, float scale);
  // Add an individual count of type Count.
  void AddCount(int32 word, const Count &count);
  // Add counts from a GeneralLmState.
  void AddCounts(const GeneralLmState &lm_state);
  // Output its contents to a sorted vector.
  void Output(std::vector<std::pair<int32, Count> > *output) const;
};

}  // namespace pocolm

#endif  // POCOLM_LM_STATE_H_
