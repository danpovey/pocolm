// lm-state-derivs.h

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

#ifndef POCOLM_LM_STATE_DERIVS_H_
#define POCOLM_LM_STATE_DERIVS_H_

#include <iostream>
#include <vector>
#include <utility>
#include "pocolm-types.h"
#include "lm-state.h"
#include "count.h"

namespace pocolm {


/**
   This class extends class FloatLmState by adding derivatives w.r.t. the
   'discount' and the float elements of 'counts'.  It's used during
   backpropagation of derivatives through the discounting process.

   You need to be a bit cautious with the Read and Write functions.
   The Read and Write functions are just inherited from the base-class
   FloatLmState, and we add ReadDerivs() and WriteDerivs() functions
   that read and write the derivative info (in practice, this will
   be from/to separate streams).
 */
class FloatLmStateDerivs: public FloatLmState {
 public:
  FloatLmStateDerivs(): total_deriv(0.0), discount_deriv(0.0) { }

  // derivative of the objective function w.r.t. 'total'.  Note, this is not
  // written to or read from disk; since 'total' is a derived variable that just
  // equals, the sum of 'discount' plus the individual counts, we add this
  // quantitity to all the individual derivatives before writing (and on
  // reading, we set it to zero).
  double total_deriv;

  // derivative of the objective function w.r.t. 'discount'
  double discount_deriv;

  // vector of the same length as 'counts', containing the derivative of the
  // objective function w.r.t. the floats in 'counts'.
  std::vector<double> count_derivs;

  // This function reads the contents of base-class FloatLmState and sizes the
  // derivatives appropriately, setting them to zero.
  void Read(std::istream &is);

  // this is just to clarify that the Write(std::ostream &os) const function is
  // inherited from the base-class.
  using FloatLmState::Write;

  // This function reads the derivatives from the stream.  It assumes
  // that the base-class has already been read or computed, so the 'counts'
  // vector is correctly sized.
  void ReadDerivs(std::istream &is);

  // this function reads derivatives and adds them to the existing derivatives..
  void ReadDerivsAdding(std::istream &is);

  // Writes the derivatives.  Note: it's not const because prior to writing,
  // it 'normalizes' the derivatives by adding total_deriv to all the other
  // derivative quantities and then zeroing total_deriv.
  void WriteDerivs(std::ostream &os);

  // Used for debug, this function prints both the base-class parameters and
  // derivatives in a human-readable way.
  void Print(std::ostream &os) const;

  void Swap(FloatLmStateDerivs *other) {
    FloatLmState::Swap(other);
    std::swap(total_deriv, other->total_deriv);
    std::swap(discount_deriv, other->discount_deriv);
    count_derivs.swap(other->count_derivs);
  }

 private:
  // called from Write, this adds total_deriv to discount_deriv
  // and each member of count_derivs, then zeroes it.
  void BackpropFromTotalDeriv();

};


/**
   This class extends class GeneralLmState by storing derivatives of the total
   data-log-likelihood w.r.t. the counts; these derivatives can be read and
   written independently of the underlying parameters.
 */
class GeneralLmStateDerivs: public GeneralLmState {
 public:
  GeneralLmStateDerivs(): discount_deriv(0.0) { }

  // This is the derivative of the objective function w.r.t.  the 'discount'
  // value of this state (note: the 'discount' value would only be nonzero if we
  // had applied a min-count).
  float discount_deriv;

  // This vector stores derivatives of the objective function
  // (the total data likelihood) w.r.t. the individual counts.
  std::vector<Count> count_derivs;


  // This function reads the contents of base-class FloatLmState and sizes the
  // derivatives appropriately, setting them to zero.
  void Read(std::istream &is);

  // This just clarifies that we are using the function
  // 'void Write(std::ostream &os); const' from the base-class,
  // to write the base-class (not the derivatives).
  using GeneralLmState::Write;

  // writes the derivatives (not the parameters stored in the base-class) to the
  // ostream.  Throws on error.
  void WriteDerivs(std::ostream &os) const;

 // Used for debug, this function prints both the base-class parameters and
  // derivatives in a human-readable way.
  void Print(std::ostream &os) const;

  // reads the derivatives from the istream, which is assumed to not be at EOF.
  // Throws on error.  Requires that the parameters already be set up (either by
  // being computed or by being read) and that the number of counts matches that
  // number of counts in the derivative.
  void ReadDerivs(std::istream &is);

  // this function reads derivatives and adds them to the existing derivatives..
  void ReadDerivsAdding(std::istream &is);

  void Swap(GeneralLmStateDerivs *other) {
    GeneralLmState::Swap(other);
    count_derivs.swap(other->count_derivs);
  }
};


}  // namespace pocolm

#endif  // POCOLM_LM_STATE_DERIVS_H_
