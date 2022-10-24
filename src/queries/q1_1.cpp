#include "q1_1.hpp"
#include "q1_common.hpp"

#include <oneapi/tbb.h>
#include <x86intrin.h>

void q1_1_agg_chunk(const WideTable &t, size_t begin, size_t end, uint32_t &acc) {
  for (size_t i = begin; i < end; ++i) {
    if (t.d_year[i] == 1993 && t.lo_discount[i] >= 1 && t.lo_discount[i] <= 3 &&
        t.lo_quantity[i] < 25) {
      acc += t.lo_extendedprice[i] * t.lo_discount[i];
    }
  }
}

uint16_t q1_1_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_1993_8u16 = _mm_set1_epi16(1993);
  static __m128i k_0_16u8 = _mm_set1_epi8(0);
  static __m128i k_4_16u8 = _mm_set1_epi8(4);
  static __m128i k_25_16u8 = _mm_set1_epi8(25);

  // Compute d_year = 1993.
  __m128i d_year_0_8u16 = _mm_lddqu_si128((__m128i *)&t.d_year[i]);
  __m128i d_year_1_8u16 = _mm_lddqu_si128((__m128i *)&t.d_year[i + 8]);
  __m128i d_year_mask_0_8u16 = _mm_cmpeq_epi16(d_year_0_8u16, k_1993_8u16);
  __m128i d_year_mask_1_8u16 = _mm_cmpeq_epi16(d_year_1_8u16, k_1993_8u16);
  __m128i mask_16u8 = _mm_packs_epi16(d_year_mask_0_8u16, d_year_mask_1_8u16);

  // Compute lo_discount BETWEEN 1 AND 3.
  __m128i lo_discount_16u8 = _mm_lddqu_si128((__m128i *)&t.lo_discount[i]);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmpgt_epi8(lo_discount_16u8, k_0_16u8));
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmplt_epi8(lo_discount_16u8, k_4_16u8));

  // Compute lo_quantity < 25.
  __m128i lo_quantity_16u8 = _mm_lddqu_si128((__m128i *)&t.lo_quantity[i]);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmplt_epi8(lo_quantity_16u8, k_25_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

uint32_t Q1_1::scalar(const WideTable &t) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), 0,
      [&](const tbb::blocked_range<size_t> &r, uint32_t acc) {
        q1_1_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      std::plus<>());
}

uint32_t Q1_1::sse(const WideTable &t) {
  uint32_t acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), 0,
      [&](const tbb::blocked_range<size_t> &r, uint32_t acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q1_1_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            acc += t.lo_extendedprice[j + k] * t.lo_discount[j + k];
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      std::plus<>());

  // Process the remaining records.
  q1_1_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return acc;
}

void Q1_1::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q1_1_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.d_year[i] == 1993 && t.lo_discount[i] >= 1 && t.lo_discount[i] <= 3 &&
        t.lo_quantity[i] < 25) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

uint32_t Q1_1::agg(const WideTable &t, const uint32_t *b) { return q1_agg(t, b); }
