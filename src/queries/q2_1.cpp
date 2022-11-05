#include "q2_1.hpp"
#include "encode.hpp"
#include "helpers.hpp"
#include "q2_common.hpp"

#include <oneapi/tbb.h>
#include <unordered_map>
#include <x86intrin.h>

void q2_1_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, int64_t> &slot =
      acc[((t.d_year[i] - 1992) << 6) | ((t.p_brand1[i] - 40) & 0b111111)];
  slot.first = true;
  slot.second += t.lo_revenue[i];
}

void q2_1_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint8_t k_mfgr_12 = encode_p_category("MFGR#12");
  static uint8_t k_america = encode_region("AMERICA");
  for (size_t i = begin; i < end; ++i) {
    if (t.p_category[i] == k_mfgr_12 && t.s_region[i] == k_america) {
      q2_1_agg_step(t, i, acc);
    }
  }
}

uint16_t q2_1_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_mfgr_12_16u8 = _mm_set1_epi8((char)encode_p_category("MFGR#12"));
  static __m128i k_america_16u8 = _mm_set1_epi8((char)encode_region("AMERICA"));

  // Compute p_category = 'MFGR#12'.
  __m128i mask_16u8 = eq_16u8(&t.p_category[i], k_mfgr_12_16u8);

  // Compute s_region = 'AMERICA'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.s_region[i], k_america_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q2_1::result_type Q2_1::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(512),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q2_1_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}

Q2_1::result_type Q2_1::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(512),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        acc.reserve(acc.size() + (r.end() - r.begin()));
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q2_1_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_1_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q2_1_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q2_agg_order(acc);
}

void Q2_1::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q2_1_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint8_t k_mfgr_12 = encode_p_category("MFGR#12");
  static uint8_t k_america = encode_region("AMERICA");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.p_category[i] == k_mfgr_12 && t.s_region[i] == k_america) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q2_1::result_type Q2_1::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(512),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_1_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}
