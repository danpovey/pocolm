// count-test.cc

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

#include <vector>
#include <set>
#include <algorithm>
#include <cassert>
#include <math.h>
#include <numeric>
#include "count.h"
#include "pocolm-types.h"





namespace pocolm {

float RandUniform() {
  return (rand() % 1000) / 1000.0f;
}
bool ApproxEqual(float a, float b, float delta = 0.0001) {
  return (fabs(a - b) <= delta * std::max(fabs(a), fabs(b)));
}

void TestCountAdd() {
  int32 num_elements = rand() % 10;

  Count sum1(0), sum2(0);
  std::vector<float> vec;
  for (int32 i = 0; i < num_elements; i++) {
    float f = RandUniform();
    vec.push_back(f);
    if (rand() % 2 == 0) {
      if (rand() % 2 == 0) {
        sum1.Add(Count(f));
      } else {
        sum1.Add(f);
      }
    } else {
      if (rand() % 2 == 0) {
        sum2.Add(Count(f));
      } else {
        sum2.Add(f);
      }
    }
    if (rand() % 3 == 0) {
      sum1.Add(sum2);
      sum2 = 0;
    }
  }
  sum1.Add(sum2);
  sum2 = 0;

  float total = std::accumulate(vec.begin(), vec.end(), 0.0);
  assert(ApproxEqual(total, sum1.total));
  std::sort(vec.begin(), vec.end(), std::greater<float>());
  vec.push_back(0.0);
  vec.push_back(0.0);
  vec.push_back(0.0);
  assert(sum1.top1 == vec[0]);
  assert(sum1.top2 == vec[1]);
  assert(sum1.top3 == vec[2]);
}

// This function tests AddBackward.
// It accumulates a sum, and a sum of slightly perturbed values, and
// checks that the derivatives are the same.

void TestCountAddBackward() {
  std::set<float> seen_before;

  int32 num_counts = rand() % 6;

  std::vector<Count> counts(num_counts, 0.0f),
      counts_perturbed(num_counts, 0.0f),
      derivs(num_counts, 0.0f);

  Count total_count(0.0),
      total_count_perturbed(0.0);

  float delta = 1.0e-03;

  for (int32 i = 0; i < num_counts; i++) {
    Count &c = counts[i], &c_perturbed = counts_perturbed[i];
    int32 num_elements = rand() % 4;
    for (int32 j = 0; j < num_elements; j++) {
      float f;
      do { f = RandUniform(); }
      while (seen_before.count(f) != 0);
      seen_before.insert(f);
      float f_perturbed = f + ((RandUniform() - 0.5) * delta);
      c.Add(f);
      c_perturbed.Add(f_perturbed);
    }
    total_count.Add(c);
    total_count_perturbed.Add(c_perturbed);
  }

  // total_deriv is the derivative of our randomly chosen
  // objective function w.r.t. the 'total_count' sum.
  // actually the objective function is just linear.
  Count total_deriv(0.0f);
  total_deriv.total = RandUniform() - 0.5;
  total_deriv.top1 = RandUniform() - 0.5;
  total_deriv.top2 = RandUniform() - 0.5;
  total_deriv.top3 = RandUniform() - 0.5;

  float objf = total_count.DotProduct(total_deriv),
      objf_perturbed = total_count_perturbed.DotProduct(total_deriv);

  Count total_deriv_orig(total_deriv);

  float objf_delta = objf_perturbed - objf,
      objf_delta_check = 0.0;

  // Compute the derivatives.
  for (int32 i = 0; i < num_counts; i++) {
    Count &c = counts[i], &c_perturbed = counts_perturbed[i],
        &c_deriv = derivs[i];
    total_count.AddBackward(c, &total_deriv,
                            &c_deriv);
    objf_delta_check += c_deriv.DotProduct(c_perturbed) -
        c_deriv.DotProduct(c);
  }
  std::cerr << "objf_delta = " << objf_delta
            << ", objf_delta_check = " << objf_delta_check
            << (ApproxEqual(objf_delta, objf_delta_check, 0.01) ? " SUCCESS" : " FAIL")
            << "\n";
}

}

int main() {
  using namespace pocolm;
  for (int32 i = 0; i < 20; i++) {
    TestCountAdd();
    TestCountAddBackward();
  }
  return 0;
}
