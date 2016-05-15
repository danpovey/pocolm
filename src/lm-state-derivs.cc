// lm-state-derivs.cc

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
#include "lm-state-derivs.h"

namespace pocolm {


// note, this doesn't read the derivatives (ReadDerivs does that), it
// just reads the base-class members and correctly sizes and zeroes the
// derivatives.
void FloatLmStateDerivs::Read(std::istream &is) {
  FloatLmState::Read(is);
  total_deriv = 0.0;
  discount_deriv = 0.0;
  count_derivs.clear();
  count_derivs.resize(counts.size(), 0.0);
}


void FloatLmStateDerivs::ReadDerivs(std::istream &is) {
  total_deriv = 0.0;
  is.read(reinterpret_cast<char*>(&discount_deriv), sizeof(double));
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size;
  is.read(reinterpret_cast<char*>(&count_size), sizeof(int32));
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for float-counts "
        "(empty or truncated input?)\n";
        exit(1);
  }
  if (count_size != static_cast<int32>(counts.size())) {
    std::cerr << "Count size mismatch: expected " << counts.size()
              << ", got " << count_size
              << " when reading float-count derivs (wrong file?)\n";
    exit(1);
  }
  // make sure they have the correct size.
  count_derivs.resize(count_size);
  assert(count_size > 0);
  is.read(reinterpret_cast<char*>(&(count_derivs[0])),
          sizeof(double) * count_size);
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for float-counts "
        "(empty or truncated input?)\n";
    exit(1);
  }
}

void FloatLmStateDerivs::ReadDerivsAdding(std::istream &is) {
  total_deriv += 0.0;  // This statement is just for clarification that the
                       // on-disk format assumes the total_deriv is zero; we
                       // know that it does nothing.
  double discount_deriv_part;
  is.read(reinterpret_cast<char*>(&discount_deriv_part), sizeof(double));
  discount_deriv += discount_deriv_part;
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size;
  is.read(reinterpret_cast<char*>(&count_size), sizeof(int32));
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
        exit(1);
  }
  if (count_size != static_cast<int32>(counts.size())) {
    std::cerr << "Count size mismatch: expected " << counts.size()
              << ", got " << count_size
              << " when reading count derivs (wrong file?)\n";
    exit(1);
  }
  assert(count_derivs.size() == size_t(count_size) && count_size > 0);
  std::vector<double> temp_derivs(count_size);
  is.read(reinterpret_cast<char*>(&(temp_derivs[0])),
          sizeof(double) * count_size);
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
    exit(1);
  }
  std::vector<double>::const_iterator in_iter = temp_derivs.begin(),
      in_end = temp_derivs.end();
  std::vector<double>::iterator out_iter = count_derivs.begin();
  for (; in_iter != in_end; ++in_iter, ++out_iter)
    *out_iter += *in_iter;
}


void FloatLmStateDerivs::BackpropFromTotalDeriv() {
  if (total_deriv == 0.0) return;
  discount_deriv += total_deriv;
  for (std::vector<double>::iterator iter = count_derivs.begin();
       iter != count_derivs.end(); ++iter)
    *iter += total_deriv;
  total_deriv = 0.0;
}

void FloatLmStateDerivs::WriteDerivs(std::ostream &os) {
  BackpropFromTotalDeriv();
  os.write(reinterpret_cast<const char*>(&discount_deriv), sizeof(double));
  assert(count_derivs.size() == counts.size());
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size = count_derivs.size();
  os.write(reinterpret_cast<const char*>(&count_size), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&(count_derivs[0])),
           sizeof(double) * count_size);
  if (os.fail()) {
    std::cerr << "Error writing derivatives for float-counts\n";
    exit(1);
  }
}

void FloatLmStateDerivs::Print(std::ostream &os) const {
  assert(counts.size() == count_derivs.size());
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  os << "total=" << total << ",d=" << total_deriv
     << " discount=" << discount << ",d=" << discount_deriv << " ";
  int32 counts_size = counts.size();
  for (int32 i = 0; i < counts_size; i++)
    os << counts[i].first << "->" << counts[i].second
       << ",d=" << count_derivs[i] << " ";
  os << "\n";
}



// note, this doesn't read the derivatives (ReadDerivs does that), it
// just reads the base-class members and correctly sizes and zeroes the
// derivatives.
void GeneralLmStateDerivs::Read(std::istream &is) {
  GeneralLmState::Read(is);
  count_derivs.clear();
  count_derivs.resize(counts.size(), 0.0);
}


void GeneralLmStateDerivs::ReadDerivs(std::istream &is) {
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size;
  is.read(reinterpret_cast<char*>(&count_size), sizeof(int32));
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
        exit(1);
  }
  if (count_size != static_cast<int32>(counts.size())) {
    std::cerr << "Count size mismatch: expected " << counts.size()
              << ", got " << count_size
              << " when reading count derivs (wrong file?)\n";
    exit(1);
  }
  // make sure they have the correct size.
  count_derivs.resize(count_size);
  assert(count_size > 0);
  is.read(reinterpret_cast<char*>(&(count_derivs[0])),
          sizeof(Count) * count_size);
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
    exit(1);
  }
}


void GeneralLmStateDerivs::ReadDerivsAdding(std::istream &is) {
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size;
  is.read(reinterpret_cast<char*>(&count_size), sizeof(int32));
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
        exit(1);
  }
  if (count_size != static_cast<int32>(counts.size())) {
    std::cerr << "Count size mismatch: expected " << counts.size()
              << ", got " << count_size
              << " when reading count derivs (wrong file?)\n";
    exit(1);
  }
  assert(count_derivs.size() == size_t(count_size) && count_size > 0);
  std::vector<Count> temp_derivs(count_size);
  is.read(reinterpret_cast<char*>(&(temp_derivs[0])),
          sizeof(Count) * count_size);
  if (is.fail() || is.eof()) {
    std::cerr << "Error reading derivatives for counts "
        "(empty or truncated input?)\n";
    exit(1);
  }
  std::vector<Count>::const_iterator in_iter = temp_derivs.begin(),
      in_end = temp_derivs.end();
  std::vector<Count>::iterator out_iter = count_derivs.begin();
  for (; in_iter != in_end; ++in_iter, ++out_iter) {
    // we want to add up the individual components of the derivative.
    // We can't use the Add() function of class Count, because that doesn't
    // do componentwise addition.
    out_iter->total += in_iter->total;
    out_iter->top1 += in_iter->top1;
    out_iter->top2 += in_iter->top2;
    out_iter->top3 += in_iter->top3;
  }
}


void GeneralLmStateDerivs::WriteDerivs(std::ostream &os) const {
  assert(count_derivs.size() == counts.size());
  // we only write and read in the size of the counts as a way to double-check
  // that there is no mismatch.
  int32 count_size = count_derivs.size();
  os.write(reinterpret_cast<const char*>(&count_size), sizeof(int32));
  os.write(reinterpret_cast<const char*>(&(count_derivs[0])),
           sizeof(Count) * count_size);
  if (os.fail()) {
    std::cerr << "Error writing derivatives for counts\n";
    exit(1);
  }
}


void GeneralLmStateDerivs::Print(std::ostream &os) const {
  os << " [ ";
  int32 hist_size = history.size();
  for (int32 i = 0; i < hist_size; i++)
    os << history[i] << " ";
  os << "]: ";
  int32 counts_size = counts.size();
  for (int32 i = 0; i < counts_size; i++)
    os << counts[i].first << "->" << counts[i].second
       << ",d=" << count_derivs[i] << " ";
  os << "\n";
}


}

