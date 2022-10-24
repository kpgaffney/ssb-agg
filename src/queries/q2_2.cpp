#include "q2_2.hpp"
#include "encode.hpp"
#include "q2_common.hpp"

#include <oneapi/tbb.h>
#include <unordered_map>
#include <x86intrin.h>

uint16_t k_mfgr_2221() {
  static uint16_t k = encode_p_brand1("MFGR#2221");
  return k;
}

uint16_t k_mfgr_2228() {
  static uint16_t k = encode_p_brand1("MFGR#2228");
  return k;
}

uint8_t k_asia() {
  static uint8_t k = encode_region("ASIA");
  return k;
}

void q2_2_agg_chunk(const WideTable &t, size_t begin, size_t end, q2_acc_type &acc) {
  acc.reserve(acc.size() + end - begin);
  for (size_t i = begin; i < end; ++i) {
    if (t.p_brand1[i] >= k_mfgr_2221() && t.p_brand1[i] <= k_mfgr_2228() &&
        t.s_region[i] == k_asia()) {
      acc[((uint32_t)t.d_year[i] << 16) | t.p_brand1[i]] += t.lo_revenue[i];
    }
  }
}

uint16_t q2_2_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_mfgr_2221_8u16 = _mm_set1_epi16((short)k_mfgr_2221());
  static __m128i k_mfgr_2228_8u16 = _mm_set1_epi16((short)k_mfgr_2228());
  static __m128i k_asia_16u8 = _mm_set1_epi8((char)k_asia());

  // Compute p_brand1 BETWEEN 'MFGR#2221' AND 'MFGR#2228'.
  __m128i p_brand1_0_8u16 = _mm_lddqu_si128((__m128i *)&t.p_brand1[i]);
  __m128i p_brand1_1_8u16 = _mm_lddqu_si128((__m128i *)&t.p_brand1[i + 8]);
  __m128i p_brand1_mask_0_lb_8u16 =
      _mm_or_si128(_mm_cmpgt_epi16(p_brand1_0_8u16, k_mfgr_2221_8u16),
                   _mm_cmpeq_epi16(p_brand1_0_8u16, k_mfgr_2221_8u16));
  __m128i p_brand1_mask_0_ub_8u16 =
      _mm_or_si128(_mm_cmplt_epi16(p_brand1_0_8u16, k_mfgr_2228_8u16),
                   _mm_cmpeq_epi16(p_brand1_0_8u16, k_mfgr_2228_8u16));
  __m128i p_brand1_mask_0_8u16 = _mm_and_si128(p_brand1_mask_0_lb_8u16, p_brand1_mask_0_ub_8u16);
  __m128i p_brand1_mask_1_lb_8u16 =
      _mm_or_si128(_mm_cmpgt_epi16(p_brand1_1_8u16, k_mfgr_2221_8u16),
                   _mm_cmpeq_epi16(p_brand1_1_8u16, k_mfgr_2221_8u16));
  __m128i p_brand1_mask_1_ub_8u16 =
      _mm_or_si128(_mm_cmplt_epi16(p_brand1_1_8u16, k_mfgr_2228_8u16),
                   _mm_cmpeq_epi16(p_brand1_1_8u16, k_mfgr_2228_8u16));
  __m128i p_brand1_mask_1_8u16 = _mm_and_si128(p_brand1_mask_1_lb_8u16, p_brand1_mask_1_ub_8u16);
  __m128i mask_16u8 = _mm_packs_epi16(p_brand1_mask_0_8u16, p_brand1_mask_1_8u16);

  // Compute s_region = 'ASIA'.
  __m128i s_region_16u8 = _mm_lddqu_si128((__m128i *)&t.s_region[i]);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmpeq_epi8(s_region_16u8, k_asia_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q2_2::result_type Q2_2::scalar(const WideTable &t) {
  q2_acc_type acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), q2_acc_type(),
      [&](const tbb::blocked_range<size_t> &r, q2_acc_type acc) {
        q2_2_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}

Q2_2::result_type Q2_2::sse(const WideTable &t) {
  q2_acc_type acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), q2_acc_type(),
      [&](const tbb::blocked_range<size_t> &r, q2_acc_type acc) {
        acc.reserve(acc.size() + (r.end() - r.begin()));
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q2_2_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            acc[((uint32_t)t.d_year[j + k] << 16) | t.p_brand1[j + k]] += t.lo_revenue[j + k];
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
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.p_brand1[i] >= k_mfgr_2221() && t.p_brand1[i] <= k_mfgr_2228() &&
        t.s_region[i] == k_asia()) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q2_2::result_type Q2_2::agg(const WideTable &t, const uint32_t *b) {
  q2_acc_type acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), q2_acc_type(),
      [&](const tbb::blocked_range<size_t> &r, q2_acc_type acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            acc[((uint32_t)t.d_year[j + k] << 16) | t.p_brand1[j + k]] += t.lo_revenue[j + k];
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q2_agg_order(acc);
}
