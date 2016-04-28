// count-inl.h

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

#ifndef POCOLM_COUNT_INL_H_
#define POCOLM_COUNT_INL_H_

// Do not include this file directly, it is included by count.h.

namespace pocolm {


// inline
void Count::Add(const Count &other) {
  float f, g;
  total += other.total;
  if (other.top1 > top1) {
    f = top1;
    g = top2;  // we'll need this in either branch so get it now.
    top1 = other.top1;
    // we need to place f somewhere.
    if (f > other.top2) {
      top2 = f;
      // we need to place g somewhere.
      if (g > other.top2) {
        // g is old top2, so must be >= top3.
        top3 = g;
      } else {
        // other.top2 > g > old top2 > top3
        top3 = other.top2;
      }
    } else {
      top2 = other.top2;
      if (f > other.top3)
        top3 = f;
      else
        top3 = other.top3;
    }
  } else {
    if (other.top1 > top2) {
      f = top2;
      top2 = other.top1;
      if (other.top2 > f) {
        top3 = other.top2;
      } else {
        top3 = f;
      }
    } else if (other.top1 > top3) {
      top3 = other.top1;
    }
  }
}

// inline
void Count::Add(float f) {
  assert(f >= 0.0f);
  total += f;
  if (f > top1) {
    float g;
    g = top1;
    top1 = f;
    f = g;
  }
  if (f > top2) {
    float g;
    g = top2;
    top2 = f;
    f = g;
  }
  if (f > top3)
    top3 = f;
}


void Count::Add(float scale, int32 num_pieces) {
  if (num_pieces == 1) {
    Add(scale);
  } else {
    assert(num_pieces > 0 && "Adding zero count");
    total += num_pieces * scale;
    if (scale >= top1) {
      if (num_pieces > 2) top3 = scale;
      else top3 = top1;
      top1 = scale;
      top2 = scale;
    } else if (scale >= top2) {
      top2 = scale;
      top3 = scale;
    } else if (scale >= top3) {
      top3 = scale;
    }
  }
}

// inline
void Count::AddBackward(const Count &other,
                        Count *this_deriv,
                        Count *other_deriv) const {
  // TODO: remove these checks.
  Check();
  other.Check();

  other_deriv->total += this_deriv->total;

  // We tried to have more optimized code for this, that took advantage of the
  // sorting to avoid doing all 9 comparisons, but it was very complex and hard
  // to verify.  This is easier.  Anyway this code isn't accessed enough times
  // for us to worry too much about efficiency.
  AddBackwardInternal(other.top1, this_deriv, &other_deriv->top1);
  AddBackwardInternal(other.top2, this_deriv, &other_deriv->top2);
  AddBackwardInternal(other.top3, this_deriv, &other_deriv->top3);
}

// inline
void Count::AddBackward(float f,
                        Count *this_deriv,
                        float *f_deriv) const {
  (*f_deriv) += this_deriv->total;
  AddBackwardInternal(f, this_deriv, f_deriv);
}

void Count::AddBackwardInternal(float f,
                                Count *this_deriv,
                                float *f_deriv) const {
  if (f == top1 && this_deriv->top1 != 0.0) {
    (*f_deriv) += this_deriv->top1;
    this_deriv->top1 = 0.0;
  } else if (f == top2 && this_deriv->top2 != 0.0) {
    (*f_deriv) += this_deriv->top2;
    this_deriv->top2 = 0.0;
  } else if (f == top3 && this_deriv->top3 != 0.0) {
    (*f_deriv) += this_deriv->top3;
    this_deriv->top3 = 0.0;
  }
}

// inline
float Count::DotProduct(const Count &other) const {
  return total * other.total +
      top1 * other.top1 +
      top2 * other.top2 +
      top3 * other.top3;
}

inline void Count::Check() const {
  assert(total >= 0.99 * (top1 + top2 + top3));
  assert(top1 >= top2);
  assert(top2 >= top3);
}

}

#endif  // POCOLM_COUNT_INL_H_
