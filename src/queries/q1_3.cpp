#include "q1_3.hpp"

#include <oneapi/tbb.h>
#include <x86intrin.h>

void q1_3_agg_chunk(const WideTable &t, size_t begin, size_t end, uint32_t &acc) {
  for (size_t i = begin; i < end; ++i) {
    if (t.d_weeknuminyear[i] == 6 && t.d_year[i] == 1994 && t.lo_discount[i] >= 5 &&
        t.lo_discount[i] <= 7 && t.lo_quantity[i] >= 36 && t.lo_quantity[i] <= 40) {
      acc += t.lo_extendedprice[i] * t.lo_discount[i];
    }
  }
}

uint16_t q1_3_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_6_16u8 = _mm_set1_epi8(6);
  static __m128i k_1994_8u16 = _mm_set1_epi16(1994);
  static __m128i k_4_16u8 = _mm_set1_epi8(4);
  static __m128i k_8_16u8 = _mm_set1_epi8(8);
  static __m128i k_35_16u8 = _mm_set1_epi8(35);
  static __m128i k_41_16u8 = _mm_set1_epi8(41);

  // Compute d_weeknuminyear = 6.
  __m128i d_weeknuminyear_16u8 = _mm_lddqu_si128((__m128i *)&t.d_weeknuminyear[i]);
  __m128i mask_16u8 = _mm_cmpeq_epi8(d_weeknuminyear_16u8, k_6_16u8);

  // Compute d_year = 1994.
  __m128i d_year_0_8u16 = _mm_lddqu_si128((__m128i *)&t.d_year[i]);
  __m128i d_year_1_8u16 = _mm_lddqu_si128((__m128i *)&t.d_year[i + 8]);
  __m128i d_year_mask_0_8u16 = _mm_cmpeq_epi16(d_year_0_8u16, k_1994_8u16);
  __m128i d_year_mask_1_8u16 = _mm_cmpeq_epi16(d_year_1_8u16, k_1994_8u16);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_packs_epi16(d_year_mask_0_8u16, d_year_mask_1_8u16));

  // Compute lo_discount BETWEEN 5 AND 7.
  __m128i lo_discount_16u8 = _mm_lddqu_si128((__m128i *)&t.lo_discount[i]);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmpgt_epi8(lo_discount_16u8, k_4_16u8));
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmplt_epi8(lo_discount_16u8, k_8_16u8));

  // Compute lo_quantity BETWEEN 36 AND 40.
  __m128i lo_quantity_16u8 = _mm_lddqu_si128((__m128i *)&t.lo_quantity[i]);
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmpgt_epi8(lo_quantity_16u8, k_35_16u8));
  mask_16u8 = _mm_and_si128(mask_16u8, _mm_cmplt_epi8(lo_quantity_16u8, k_41_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

uint32_t q1_3_scalar(const WideTable &t) {
  return tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.lo_quantity.size()), 0,
      [&](const tbb::blocked_range<size_t> &r, uint32_t acc) {
        q1_3_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      std::plus<>());
}

uint32_t q1_3_sse(const WideTable &t) {
  uint32_t acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), 0,
      [&](const tbb::blocked_range<size_t> &r, uint32_t acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q1_3_sse_filter_chunk(t, j);

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
  q1_3_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return acc;
}

void q1_3_filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q1_3_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.d_yearmonthnum[i] == 199401 && t.lo_discount[i] >= 4 && t.lo_discount[i] <= 6 &&
        t.lo_quantity[i] >= 26 && t.lo_quantity[i] <= 35) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

uint32_t q1_3_agg(const WideTable &t, const uint32_t *b) {
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
