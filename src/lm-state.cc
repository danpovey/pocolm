// lm-state.cc

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
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "lm-state.h"

namespace pocolm {


void IntLmState::Print(std::ostream &os) const {
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  if (discount != 0)
    os << "discount=" << discount << " ";
  for (size_t i = 0; i < counts.size(); i++)
    os << counts[i].first << "->" << counts[i].second << " ";
  os << "\n";
  Check();
}


void IntLmState::Write(std::ostream &os) const {
  if (rand() % 2 == 0)
    Check();
  if (discount != 0) {
    assert(discount > 0);
    // We write the negative of the discount, if it's
    // nonzero, and nothing otherwise... this gives back compatibility
    // in the on-disk format to when there was no 'discount' class member,
    // and also saves a little I/O.
    int32 neg_discount = -discount;
    os.write(reinterpret_cast<const char*>(&neg_discount),
             sizeof(int32));
  }
  int32 history_size = history.size(),
      num_counts = counts.size(),
      buffer_size = (2 + history_size + 2 * num_counts);
  assert(num_counts > 0);
  int32 *buffer = new int32[buffer_size];
  buffer[0] = history_size;
  buffer[1] = num_counts;
  for (int32 i = 0; i < history_size; i++)
    buffer[2 + i] = history[i];
  for (int32 i = 0; i < num_counts; i++) {
    buffer[2 + history_size + 2 * i] = counts[i].first;
    buffer[2 + history_size + 2 * i + 1] = counts[i].second;
  }
  os.write(reinterpret_cast<const char*>(buffer),
           sizeof(int32) * size_t(buffer_size));
  if (!os.good()) {
    std::cerr << "Failure writing IntLmState to stream\n";
    exit(1);
  }
  delete[] buffer;
}


inline static void ReadInt(std::istream &is,
                            int32 *i) {
  size_t bytes_read = is.read(reinterpret_cast<char*>(i),
                              sizeof(int32)).gcount();
  if (bytes_read != sizeof(int32)) {
    std::cerr << "Failure reading IntLmState, expected 4 bytes, got "
              << bytes_read;
    exit(1);
  }
}

void IntLmState::Read(std::istream &is) {
  ReadInt(is, &discount);
  int32 history_size, num_counts;
  if (discount < 0) {
    discount *= -1;  // We just read the negative of the discount.
    ReadInt(is, &history_size);
  } else {
    history_size = discount;  // We just read the history-size, the discount was
                              // zero.
    discount = 0;
  }
  ReadInt(is, &num_counts);

  assert(history_size >= 0 && num_counts > 0);
  history.resize(history_size);
  size_t bytes_read;
  if (history_size > 0) {
    size_t expected_bytes = sizeof(int32) * history_size;
    bytes_read = is.read(reinterpret_cast<char*>(&(history[0])),
                         expected_bytes).gcount();
    if (bytes_read != expected_bytes) {
      std::cerr << "Failure reading IntLmState history, expected "
                << expected_bytes << " bytes, got "
                << bytes_read;
      exit(1);
    }
  }
  counts.resize(num_counts);
  assert(sizeof(std::pair<int32,int32>) == 8);

  size_t expected_bytes = sizeof(int32) * 2 * num_counts;
  bytes_read = is.read(reinterpret_cast<char*>(&(counts[0])),
                       expected_bytes).gcount();
  if (bytes_read != expected_bytes) {
    std::cerr << "Failure reading IntLmState counts, expected "
              << expected_bytes << " bytes, got "
              << bytes_read;
    exit(1);
  }
  if (rand() % 10 == 0)
    Check();
}

void IntLmState::Check() const {
  assert(discount >= 0);
  for (size_t i = 0; i < history.size(); i++)
    assert(history[i] > 0 && history[i] != static_cast<int32>(kEosSymbol));
  assert(counts.size() > 0);
  for (size_t i = 0; i < counts.size(); i++) {
    assert(counts[i].first > 0 && counts[i].first != kBosSymbol);
    assert(counts[i].second > 0);
    if (i + 1 < counts.size())
      assert(counts[i].first < counts[i+1].first);
  }
}

void NullLmState::Write(std::ostream &os) const {
  int32 history_size = history.size(), num_predicted = predicted.size();
  assert(num_predicted > 0);
  os.write(reinterpret_cast<const char*>(&history_size), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&num_predicted), sizeof(int32));
  if (history_size > 0) {
    os.write(reinterpret_cast<const char*>(&(history[0])),
             sizeof(int32) * history_size);
  }
  os.write(reinterpret_cast<const char*>(&(predicted[0])),
           sizeof(int32) * num_predicted);
  if (!os.good()) {
    std::cerr << "Failure writing NullLmState to stream\n";
    exit(1);
  }
}

void NullLmState::Read(std::istream &is) {
  int32 history_size, num_predicted;
  is.read(reinterpret_cast<char*>(&history_size), sizeof(int32));
  is.read(reinterpret_cast<char*>(&num_predicted), sizeof(int32));
  if (!is.good() || is.eof()) {
    std::cerr << "Failure reading FloatLmState from stream\n";
    exit(1);
  }
  if (history_size < 0 || history_size > 10000 || num_predicted <= 0) {
    std::cerr << "Failure reading NullLmState from stream: "
        "got implausible data (wrong input?)\n";
    exit(1);
  }
  history.resize(history_size);
  predicted.resize(num_predicted);
  if (history_size > 0) {
    is.read(reinterpret_cast<char*>(&(history[0])),
            sizeof(int32) * history_size);
  }
  is.read(reinterpret_cast<char*>(&(predicted[0])),
          sizeof(int32) * num_predicted);
  if (!is.good()) {
    std::cerr << "Failure reading NullLmState from stream\n";
    exit(1);
  }
  if (rand() % 10 == 0)
    Check();
}

void NullLmState::Check() const {
  for (size_t i = 0; i < history.size(); i++)
    assert(history[i] > 0 && history[i] != static_cast<int32>(kEosSymbol));
  assert(predicted.size() > 0);
  for (size_t i = 0; i + 1 < predicted.size(); i++) {
    assert(predicted[i] < predicted[i+1]);
  }
}

void NullLmState::Print(std::ostream &os) const {
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  int32 predicted_size = predicted.size();
  for (int32 i = 0; i < predicted_size; i++)
    os << predicted[i] << " ";
  os << "\n";
}


void FloatLmState::Write(std::ostream &os) const {
  int32 history_size = history.size(), num_counts = counts.size();
  assert(num_counts > 0);
  os.write(reinterpret_cast<const char*>(&history_size), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&num_counts), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&total), sizeof(float));
  os.write(reinterpret_cast<const char*>(&discount), sizeof(float));
  if (history_size > 0) {
    os.write(reinterpret_cast<const char*>(&(history[0])),
             sizeof(int32) * history_size);
  }
  os.write(reinterpret_cast<const char*>(&(counts[0])),
           sizeof(std::pair<int32, float>) * num_counts);
  if (!os.good()) {
    std::cerr << "Failure writing FloatLmState to stream\n";
    exit(1);
  }
}

void FloatLmState::Read(std::istream &is) {
  int32 history_size, num_counts;
  is.read(reinterpret_cast<char*>(&history_size), sizeof(int32));
  is.read(reinterpret_cast<char*>(&num_counts), sizeof(int32));
  if (!is.good() || is.eof()) {
    std::cerr << "Failure reading FloatLmState from stream\n";
    exit(1);
  }
  if (history_size < 0 || history_size > 10000 || num_counts <= 0) {
    std::cerr << "Failure reading FloatLmState from stream: "
        "got implausible data (wrong input?)\n";
    exit(1);
  }
  is.read(reinterpret_cast<char*>(&total), sizeof(float));
  is.read(reinterpret_cast<char*>(&discount), sizeof(float));
  history.resize(history_size);
  counts.resize(num_counts);
  if (history_size > 0) {
    is.read(reinterpret_cast<char*>(&(history[0])),
            sizeof(int32) * history_size);
  }
  is.read(reinterpret_cast<char*>(&(counts[0])),
          sizeof(std::pair<int32, float>) * num_counts);
  if (!is.good()) {
    std::cerr << "Failure reading FloatLmState from stream\n";
    exit(1);
  }
  if (rand() % 10 == 0)
    Check();
}


// this fixes small errors in the total-count of LM-states, that
// are caused by the accumulation of numerical roundoff.
void FloatLmState::FixTotalCount() {
  double total_count = discount;
  std::vector<std::pair<int32, float> >::const_iterator
      iter = counts.begin(),
      end = counts.end();
  for (; iter != end; ++iter)
    total_count += iter->second;
  if (fabs(total - total_count) > 0.0001 * total_count) {
    std::cerr << "Fixing lm-state total " << total << " -> "
              << total_count << "\n";
  }
  total = total_count;
}

void FloatLmState::Check() const {
  for (size_t i = 0; i < history.size(); i++)
    assert(history[i] > 0 && history[i] != static_cast<int32>(kEosSymbol));
  assert(counts.size() > 0);
  for (size_t i = 0; i < counts.size(); i++) {
    assert(counts[i].first > 0 && counts[i].first != kBosSymbol);
    if (i + 1 < counts.size())
      assert(counts[i].first < counts[i+1].first);
  }
  assert(discount >= 0.0);
  double my_total = discount;
  for (std::vector<std::pair<int32,float> >::const_iterator iter =
           counts.begin(); iter != counts.end(); ++iter)
    my_total += iter->second;
  if (fabs(total - my_total) > 0.0001 * fabs(my_total)) {
    std::cerr << "warning: in float-counts," << total << " != "
              << my_total << "\n";
  }
}

void FloatLmState::Print(std::ostream &os) const {
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  os << "total=" << total << " discount=" << discount << " ";
  int32 counts_size = counts.size();
  for (int32 i = 0; i < counts_size; i++)
    os << counts[i].first << "->" << counts[i].second << " ";
  os << "\n";
}

void FloatLmState::ComputeTotal() {
  double my_total = discount;
  for (std::vector<std::pair<int32,float> >::iterator iter =
           counts.begin(); iter != counts.end(); ++iter)
    my_total += iter->second;
  total = my_total;
}

void GeneralLmState::Print(std::ostream &os) const {
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  if (discount != 0.0)
    os << "discount=" << discount << " ";
  int32 counts_size = counts.size();
  for (int32 i = 0; i < counts_size; i++)
    os << counts[i].first << "->" << counts[i].second << " ";
  os << "\n";
}

void GeneralLmState::Write(std::ostream &os) const {
  if (rand() % 10 == 0)
    Check();
  int32 history_size = history.size(),
      num_counts = counts.size();
  assert(num_counts > 0);
  // declare a variable so this code won't compile if discount is changed to
  // double, since we use sizeof(float).
  const float *discount_ptr = &discount;
  os.write(reinterpret_cast<const char*>(discount_ptr), sizeof(float));
  os.write(reinterpret_cast<const char*>(&history_size), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&num_counts), sizeof(int32));
  if (history_size > 0) {
    os.write(reinterpret_cast<const char*>(&(history[0])),
             sizeof(int32) * history_size);
  }
  size_t pair_size = sizeof(std::pair<int32, Count>);
  // We don't check that this size equals sizeof(int32) + 4 * sizeof(float).
  // Thus, in principle there could be some kind of padding, and we'd be
  // wasting some space on disk (and also run the risk of this not being
  // readable on a different architecture), but these are only
  // intermediate files used on a single machine-- the final output
  // of this toolkit is going to be a text ARPA file.

  os.write(reinterpret_cast<const char*>(&(counts[0])),
           pair_size * num_counts);

  if (!os.good()) {
    std::cerr << "Failure writing GeneralLmState to stream\n";
    exit(1);
  }
}

void GeneralLmState::Read(std::istream &is) {
  int32 history_size, num_counts;

  size_t bytes_read = is.read(reinterpret_cast<char*>(&discount),
                              sizeof(float)).gcount();
  if (bytes_read != sizeof(float)) {
    std::cerr << "Failure reading GeneralLmState, expected 4 bytes, got "
              << bytes_read;
    exit(1);
  }
  assert(discount >= 0.0 && "Reading GeneralLmState, got bad data");
  bytes_read = is.read(reinterpret_cast<char*>(&history_size),
                       sizeof(int32)).gcount();
  if (bytes_read != sizeof(int32)) {
    std::cerr << "Failure reading GeneralLmState, expected 4 bytes, got "
              << bytes_read;
    exit(1);
  }
  if (history_size > 10000 || history_size < 0) {
    std::cerr << "Reading GeneralLmState, expected history size, got "
              << history_size << " (attempting to read wrong file type?)\n";
    exit(1);
  }
  bytes_read = is.read(reinterpret_cast<char*>(&num_counts),
                       sizeof(int32)).gcount();
  if (bytes_read != sizeof(int32)) {
    std::cerr << "Failure reading GeneralLmState, expected 4 bytes, got "
              << bytes_read;
    exit(1);
  }
  if (num_counts <= 0) {
    std::cerr << "Reading GeneralLmState, expected num-counts, got "
              << num_counts << " (attempting to read wrong file type?)\n";
    exit(1);
  }
  history.resize(history_size);
  if (history_size > 0) {
    size_t expected_bytes = sizeof(int32) * history_size;
    bytes_read = is.read(reinterpret_cast<char*>(&(history[0])),
                         expected_bytes).gcount();
    if (bytes_read != expected_bytes) {
      std::cerr << "Failure reading GeneralLmState history, expected "
                << expected_bytes << " bytes, got "
                << bytes_read;
      exit(1);
    }
  }
  counts.resize(num_counts);
  size_t pair_size = sizeof(std::pair<int32, Count>);

  size_t expected_bytes = pair_size * num_counts;
  bytes_read = is.read(reinterpret_cast<char*>(&(counts[0])),
                       expected_bytes).gcount();
  if (bytes_read != expected_bytes) {
    std::cerr << "Failure reading GeneralLmState counts, expected "
              << expected_bytes << " bytes, got "
              << bytes_read;
    exit(1);
  }
  if (rand() % 10 == 0)
    Check();
}


void GeneralLmState::Check() const {
  assert(discount >= 0.0);
  for (size_t i = 0; i < history.size(); i++)
    assert(history[i] > 0 && history[i] != static_cast<int32>(kEosSymbol));
  assert(counts.size() > 0);
  for (size_t i = 0; i < counts.size(); i++) {
    assert(counts[i].first > 0 && counts[i].first != kBosSymbol);
    if (i + 1 < counts.size())
      assert(counts[i].first < counts[i+1].first);
    // don't do any further checking on the counts, as they could be derivatives
    // and wouldn't pass the normal checks such as for positivity.
  }
}

void GeneralLmStateBuilder::Clear() {
  discount = 0.0;
  word_to_pos.clear();
  counts.clear();
}

void GeneralLmStateBuilder::AddCount(int32 word, float count) {
  int32 cur_counts_size = counts.size();
  std::pair<unordered_map<int32, int32>::iterator, bool> pr =
      word_to_pos.insert(std::pair<const int32, int32>(word,
                                                       cur_counts_size));
  if (pr.second) { // inserted an element.
    counts.push_back(Count(count));
  } else {
    int32 pos = pr.first->second;
    assert(pos < cur_counts_size);
    counts[pos].Add(count);
  }
}


void GeneralLmStateBuilder::AddCount(int32 word, float scale, int32 num_pieces) {
  int32 cur_counts_size = counts.size();
  std::pair<unordered_map<int32, int32>::iterator, bool> pr =
      word_to_pos.insert(std::pair<const int32, int32>(word,
                                                       cur_counts_size));
  if (pr.second) { // inserted an element.
    counts.push_back(Count(scale, num_pieces));
  } else {
    int32 pos = pr.first->second;
    assert(pos < cur_counts_size);
    counts[pos].Add(scale, num_pieces);
  }
}


void GeneralLmStateBuilder::AddCounts(const IntLmState &lm_state, float scale) {
  discount += scale * lm_state.discount;
  for (std::vector<std::pair<int32, int32> >::const_iterator iter =
           lm_state.counts.begin(); iter != lm_state.counts.end();
       ++iter)
    AddCount(iter->first, scale, iter->second);
}
void GeneralLmStateBuilder::AddCount(int32 word, const Count &count) {
  int32 cur_counts_size = counts.size();
  std::pair<unordered_map<int32, int32>::iterator, bool> pr =
      word_to_pos.insert(std::pair<const int32, int32>(word, cur_counts_size));
  if (pr.second) {
    // we inserted a new element into the word->position hash.
    counts.push_back(count);
  } else {
    int32 pos = pr.first->second;
    assert(pos < cur_counts_size);
    counts[pos].Add(count);
  }
}
void GeneralLmStateBuilder::AddCounts(const GeneralLmState &lm_state) {
  discount += lm_state.discount;
  for (std::vector<std::pair<int32, Count> >::const_iterator iter =
           lm_state.counts.begin(); iter != lm_state.counts.end();
       ++iter)
    AddCount(iter->first, iter->second);
}

void GeneralLmStateBuilder::Output(const std::vector<int32> &history,
                                   GeneralLmState *output_state) const {
  output_state->history = history;
  size_t size = counts.size();
  assert(counts.size() == word_to_pos.size());
  output_state->discount = discount;
  std::vector<std::pair<int32, int32> > pairs;
  pairs.reserve(size);
  for (unordered_map<int32, int32>::const_iterator iter =
           word_to_pos.begin(); iter != word_to_pos.end(); ++iter)
    pairs.push_back(std::pair<int32,int32>(iter->first, iter->second));
  std::sort(pairs.begin(), pairs.end());
  output_state->counts.clear();
  output_state->counts.resize(size);
  for (size_t i = 0; i < size; i++) {
    output_state->counts[i].first = pairs[i].first;
    output_state->counts[i].second = counts[pairs[i].second];
  }
}

void GeneralLmState::Swap(GeneralLmState *other) {
  history.swap(other->history);
  counts.swap(other->counts);
  std::swap(discount, other->discount);
}

// This is an inefficient implementation.  I hope Ke and Zhouyang will be able
// to improve it.
void MergeIntLmStates(const std::vector<const IntLmState*> &source_pointers,
                      IntLmState *merged_state) {
  assert(source_pointers.size() > 1);
  std::vector<std::pair<int32, int32> > temp_counts;
  merged_state->history = source_pointers[0]->history;
  size_t total_size = 0;
  for (size_t i = 0; i < source_pointers.size(); i++)
    total_size += source_pointers[i]->counts.size();

  temp_counts.reserve(total_size);
  for (size_t i = 0; i < source_pointers.size(); i++) {
    const IntLmState &src = *(source_pointers[i]);
    temp_counts.insert(temp_counts.end(),
                       src.counts.begin(), src.counts.end());
  }
  std::sort(temp_counts.begin(), temp_counts.end());
  // now merge any identical counts.
  std::vector<std::pair<int32, int32> >::const_iterator
      src = temp_counts.begin(), end = temp_counts.end();
  std::vector<std::pair<int32, int32> >::iterator
      dest = temp_counts.begin();
  while (src != end) {
    int32 cur_word = src->first;
    *dest = *src;
    src++;
    while (src != end && src->first == cur_word) {
      dest->second += src->second;
      src++;
    }
    dest++;
  }
  temp_counts.resize(dest - temp_counts.begin());
  merged_state->counts.swap(temp_counts);
}


}

