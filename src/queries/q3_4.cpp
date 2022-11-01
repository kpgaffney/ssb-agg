#include "q3_4.hpp"
#include "encode.hpp"
#include "helpers.hpp"

#include <oneapi/tbb.h>
#include <x86intrin.h>

void q3_4_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, uint64_t> &slot = acc[((t.c_city[i] - 230) << 3) | (t.s_city[i] - 230)];
  slot.first = true;
  slot.second += t.lo_revenue[i];
}

std::vector<Q3_2Row> q3_4_agg_order(const Accumulator &acc) {
  std::vector<Q3_2Row> result;

  for (size_t i = 0; i < acc.size(); ++i) {
    if (acc[i].first) {
      uint8_t c_city = (i >> 3) + 230;
      uint8_t s_city = (i & 0b111) + 230;
      result.emplace_back(c_city, s_city, 1997, acc[i].second);
    }
  }

  std::sort(result.begin(), result.end(), [](const Q3_2Row &a, const Q3_2Row &b) {
    return a.d_year < b.d_year || (a.d_year == b.d_year && a.sum_lo_revenue > b.sum_lo_revenue);
  });

  return result;
}

void q3_4_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint8_t k_united_ki1 = encode_city("UNITED KI1");
  static uint8_t k_united_ki5 = encode_city("UNITED KI5");
  static uint8_t k_dec1997 = encode_d_yearmonth("Dec1997");
  for (size_t i = begin; i < end; ++i) {
    if ((t.c_city[i] == k_united_ki1 || t.c_city[i] == k_united_ki5) &&
        (t.s_city[i] == k_united_ki1 || t.s_city[i] == k_united_ki5) &&
        t.d_yearmonth[i] == k_dec1997) {
      q3_4_agg_step(t, i, acc);
    }
  }
}

uint16_t q3_4_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_united_ki1_16u8 = _mm_set1_epi8((char)encode_city("UNITED KI1"));
  static __m128i k_united_ki5_16u8 = _mm_set1_epi8((char)encode_city("UNITED KI5"));
  static __m128i k_dec1997_16u8 = _mm_set1_epi8((char)encode_d_yearmonth("Dec1997"));

  // Compute c_city = 'UNITED KI1' OR c_city = 'UNITED KI5'.
  __m128i c_city_16u8 = _mm_lddqu_si128((__m128i *)&t.c_city[i]);
  __m128i mask_16u8 = _mm_or_si128(_mm_cmpeq_epi8(c_city_16u8, k_united_ki1_16u8),
                                   _mm_cmpeq_epi8(c_city_16u8, k_united_ki5_16u8));

  // Compute s_city = 'UNITED KI1' OR s_city = 'UNITED KI5'.
  __m128i s_city_16u8 = _mm_lddqu_si128((__m128i *)&t.s_city[i]);
  mask_16u8 =
      _mm_and_si128(mask_16u8, _mm_or_si128(_mm_cmpeq_epi8(s_city_16u8, k_united_ki1_16u8),
                                            _mm_cmpeq_epi8(s_city_16u8, k_united_ki5_16u8)));

  // Compute d_yearmonth = 'Dec1997'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.d_yearmonth[i], k_dec1997_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q3_4::result_type Q3_4::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(64),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q3_4_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q3_4_agg_order(acc);
}

Q3_4::result_type Q3_4::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(64),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q3_4_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q3_4_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q3_4_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q3_4_agg_order(acc);
}

void Q3_4::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q3_4_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint8_t k_united_ki1 = encode_city("UNITED KI1");
  static uint8_t k_united_ki5 = encode_city("UNITED KI5");
  static uint8_t k_dec1997 = encode_d_yearmonth("Dec1997");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if ((t.c_city[i] == k_united_ki1 || t.c_city[i] == k_united_ki5) &&
        (t.s_city[i] == k_united_ki1 || t.s_city[i] == k_united_ki5) &&
        t.d_yearmonth[i] == k_dec1997) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q3_4::result_type Q3_4::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(512),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];

          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q3_4_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q3_4_agg_order(acc);
}
