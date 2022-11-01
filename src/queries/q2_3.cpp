#include "q2_3.hpp"
#include "encode.hpp"
#include "helpers.hpp"
#include "q2_common.hpp"

#include <oneapi/tbb.h>
#include <unordered_map>
#include <x86intrin.h>

void q2_3_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, uint64_t> &slot = acc[t.d_year[i] - 1992];
  slot.first = true;
  slot.second += t.lo_revenue[i];
}

void q2_3_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint16_t k_mfgr_2221 = encode_p_brand1("MFGR#2221");
  static uint8_t k_europe = encode_region("EUROPE");
  for (size_t i = begin; i < end; ++i) {
    if (t.p_brand1[i] == k_mfgr_2221 && t.s_region[i] == k_europe) {
      q2_3_agg_step(t, i, acc);
    }
  }
}

uint16_t q2_3_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_mfgr_2221_8u16 = _mm_set1_epi16((short)encode_p_brand1("MFGR#2221"));
  static __m128i k_europe_16u8 = _mm_set1_epi8((char)encode_region("EUROPE"));

  // Compute p_brand1 = 'MFGR#2221'.
  __m128i mask_16u8 = eq_16u16(&t.p_brand1[i], k_mfgr_2221_8u16);

  // Compute s_region = 'EUROPE'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.s_region[i], k_europe_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q2_3::result_type Q2_3::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(8),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q2_3_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}

Q2_3::result_type Q2_3::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(8),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        acc.reserve(acc.size() + (r.end() - r.begin()));
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q2_3_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_3_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q2_3_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q2_agg_order(acc);
}

void Q2_3::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q2_3_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint16_t k_q2_3_mfgr_2221 = encode_p_brand1("MFGR#2221");
  static uint8_t k_europe = encode_region("EUROPE");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.p_brand1[i] == k_q2_3_mfgr_2221 && t.s_region[i] == k_europe) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q2_3::result_type Q2_3::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(8),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_3_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}
