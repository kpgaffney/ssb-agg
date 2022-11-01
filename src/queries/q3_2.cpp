#include "q3_2.hpp"
#include "encode.hpp"
#include "helpers.hpp"

#include <oneapi/tbb.h>
#include <x86intrin.h>

void q3_2_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, uint64_t> &slot =
      acc[((t.c_city[i] - 240) << 7) | ((t.s_city[i] - 240) << 3) | (t.d_year[i] - 1992)];
  slot.first = true;
  slot.second += t.lo_revenue[i];
}

std::vector<Q3_2Row> q3_2_agg_order(const Accumulator &acc) {
  std::vector<Q3_2Row> result;

  for (size_t i = 0; i < acc.size(); ++i) {
    if (acc[i].first) {
      uint8_t c_city = (i >> 7) + 240;
      uint8_t s_city = ((i >> 3) & 0b1111) + 240;
      uint16_t d_year = (i & 0b111) + 1992;
      result.emplace_back(c_city, s_city, d_year, acc[i].second);
    }
  }

  std::sort(result.begin(), result.end(), [](const Q3_2Row &a, const Q3_2Row &b) {
    return a.d_year < b.d_year || (a.d_year == b.d_year && a.sum_lo_revenue > b.sum_lo_revenue);
  });

  return result;
}

void q3_2_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint8_t k_united_states = encode_nation("UNITED STATES");
  for (size_t i = begin; i < end; ++i) {
    if (t.c_nation[i] == k_united_states && t.s_nation[i] == k_united_states &&
        t.d_year[i] >= 1992 && t.d_year[i] <= 1997) {
      q3_2_agg_step(t, i, acc);
    }
  }
}

uint16_t q3_2_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_united_states = _mm_set1_epi8((char)encode_nation("UNITED STATES"));
  static __m128i k_1992_8u16 = _mm_set1_epi16(1992);
  static __m128i k_1997_8u16 = _mm_set1_epi16(1997);

  // Compute c_nation = 'UNITED STATES'.
  __m128i mask_16u8 = eq_16u8(&t.c_nation[i], k_united_states);

  // Compute s_nation = 'UNITED STATES'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.s_nation[i], k_united_states));

  // Compute d_year >= 1992.
  mask_16u8 = _mm_and_si128(mask_16u8, ge_16u16(&t.d_year[i], k_1992_8u16));

  // Compute d_year <= 1997.
  mask_16u8 = _mm_and_si128(mask_16u8, le_16u16(&t.d_year[i], k_1997_8u16));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q3_2::result_type Q3_2::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(2048),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q3_2_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q3_2_agg_order(acc);
}

Q3_2::result_type Q3_2::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(2048),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q3_2_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q3_2_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q3_2_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q3_2_agg_order(acc);
}

void Q3_2::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q3_2_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint8_t k_united_states = encode_nation("UNITED STATES");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.c_nation[i] == k_united_states && t.s_nation[i] == k_united_states &&
        t.d_year[i] >= 1992 && t.d_year[i] <= 1997) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q3_2::result_type Q3_2::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(2048),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];

          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q3_2_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q3_2_agg_order(acc);
}
