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

// some hardcoded constants; we use #defines because dealing with constants in
// C/C++ is such a headache.
// please see discount-counts-1gram.cc for how these are used.
#define POCOLM_UNIGRAM_D1 0.75
#define POCOLM_UNIGRAM_D2 0.25
#define POCOLM_UNIGRAM_D3 0.1
// In unigram discounting, 'POCOLM_UNK_PROPORTION' is the proportion of the
// discounted amount that we assign to the unknown-word '<unk>'... the remaining
// discounted amount is divided equally between all words except <s> and <unk>.
#define POCOLM_UNK_PROPORTION 0.5
// 1 if when discounting, we want keep the parts of the counts separate as we
// add them to the backoff state.
#define POCOLM_SEPARATE_COUNTS 1

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
