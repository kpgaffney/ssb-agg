#include "q1_common.hpp"

#include <oneapi/tbb.h>

uint32_t q1_agg(const WideTable &t, const uint32_t *b) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), 0,
      [&](const tbb::blocked_range<size_t> &r, uint32_t acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            acc += t.lo_extendedprice[j + k] * t.lo_discount[j + k];
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      std::plus<>());
}
