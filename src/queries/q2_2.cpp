#include "q2_2.hpp"
#include "encode.hpp"
#include "helpers.hpp"
#include "q2_common.hpp"

#include <oneapi/tbb.h>
#include <unordered_map>
#include <x86intrin.h>

void q2_2_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, int64_t> &slot =
      acc[((t.d_year[i] - 1992) << 3) | ((t.p_brand1[i] - 260) & 0b111)];
  slot.first = true;
  slot.second += t.lo_revenue[i];
}

void q2_2_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint16_t k_mfgr_2221 = encode_p_brand1("MFGR#2221");
  static uint16_t k_mfgr_2228 = encode_p_brand1("MFGR#2228");
  static uint8_t k_asia = encode_region("ASIA");
  for (size_t i = begin; i < end; ++i) {
    if (t.p_brand1[i] >= k_mfgr_2221 && t.p_brand1[i] <= k_mfgr_2228 && t.s_region[i] == k_asia) {
      q2_2_agg_step(t, i, acc);
    }
  }
}

uint16_t q2_2_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_mfgr_2221_8u16 = _mm_set1_epi16((short)encode_p_brand1("MFGR#2221"));
  static __m128i k_mfgr_2228_8u16 = _mm_set1_epi16((short)encode_p_brand1("MFGR#2228"));
  static __m128i k_asia_16u8 = _mm_set1_epi8((char)encode_region("ASIA"));

  // Compute p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'.
  __m128i mask_16u8 = ge_16u16(&t.p_brand1[i], k_mfgr_2221_8u16);
  mask_16u8 = _mm_and_si128(mask_16u8, le_16u16(&t.p_brand1[i], k_mfgr_2228_8u16));

  // Compute s_region = 'ASIA'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.s_region[i], k_asia_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q2_2::result_type Q2_2::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(64),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q2_2_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}

Q2_2::result_type Q2_2::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(64),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        acc.reserve(acc.size() + (r.end() - r.begin()));
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q2_2_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_2_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q2_2_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q2_agg_order(acc);
}

void Q2_2::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q2_2_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint16_t k_mfgr_2221 = encode_p_brand1("MFGR#2221");
  static uint16_t k_mfgr_2228 = encode_p_brand1("MFGR#2228");
  static uint8_t k_asia = encode_region("ASIA");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.p_brand1[i] >= k_mfgr_2221 && t.p_brand1[i] <= k_mfgr_2228 && t.s_region[i] == k_asia) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q2_2::result_type Q2_2::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(64),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q2_2_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}
