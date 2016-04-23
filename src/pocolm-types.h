#include <stdint.h>

#ifndef POCOLM_POCOLM_TYPES_H_
#define POCOLM_POCOLM_TYPES_H_ 1


// putting these types in global namespace for now- we can revisit this later if
// it causes conflicts.

typedef int8_t   int8;
typedef int16_t  int16;
typedef int32_t  int32;
typedef int64_t  int64;

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;
typedef float    float32;
typedef double   double64;

// don't change these, there is code that relies on the fact that kBosSymbol is
// first and kEosSymbol is second, e.g. search for kEosSymbol in
// discount-counts-1gram.cc.
enum SpecialSymbols {
  kBosSymbol = 1,
  kEosSymbol = 2,
  kUnkSymbol = 3
};

#ifdef _MSC_VER
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#elif __cplusplus > 199711L || defined(__GXX_EXPERIMENTAL_CXX0X__)
#include <unordered_map>
#include <unordered_set>
using std::unordered_map;
using std::unordered_set;
#else
#include <tr1/unordered_map>
#include <tr1/unordered_set>
using std::tr1::unordered_map;
using std::tr1::unordered_set;
#endif

#endif  // POCOLM_POCOLM_TYPES_H_
