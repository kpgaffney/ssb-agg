#include "q4_1.hpp"
#include "encode.hpp"
#include "helpers.hpp"

#include <iomanip>
#include <oneapi/tbb.h>
#include <x86intrin.h>

Q4_1Row::Q4_1Row(uint16_t d_year, uint8_t c_nation, uint64_t sum_profit)
    : d_year(d_year), c_nation(c_nation), sum_profit(sum_profit) {}

bool operator==(const Q4_1Row &a, const Q4_1Row &b) {
  return a.d_year == b.d_year && a.c_nation == b.c_nation && a.sum_profit == b.sum_profit;
}

std::ostream &operator<<(std::ostream &os, const Q4_1Row &row) {
  os << row.d_year << '|' << (int)row.c_nation << '|' << row.sum_profit;
  return os;
}

void q4_1_agg_step(const WideTable &t, size_t i, Accumulator &acc) {
  std::pair<bool, uint64_t> &slot = acc[((t.d_year[i] - 1992) << 5) | t.c_nation[i]];
  slot.first = true;
  slot.second += t.lo_revenue[i] - t.lo_supplycost[i];
}

std::vector<Q4_1Row> q4_1_agg_order(const Accumulator &acc) {
  std::vector<Q4_1Row> result;
  result.reserve(acc.size());

  for (size_t i = 0; i < acc.size(); ++i) {
    if (acc[i].first) {
      uint16_t d_year = (i >> 5) + 1992;
      uint8_t c_nation = i & 0b11111;
      result.emplace_back(d_year, c_nation, acc[i].second);
    }
  }

  std::sort(result.begin(), result.end(), [](const Q4_1Row &a, const Q4_1Row &b) {
    return a.d_year < b.d_year || (a.d_year == b.d_year && a.c_nation < b.c_nation);
  });

  return result;
}

void q4_1_agg_chunk(const WideTable &t, size_t begin, size_t end, Accumulator &acc) {
  static uint8_t k_america = encode_region("AMERICA");
  static uint8_t k_mfgr_1 = encode_p_mfgr("MFGR#1");
  static uint8_t k_mfgr_2 = encode_p_mfgr("MFGR#2");
  for (size_t i = begin; i < end; ++i) {
    if (t.c_region[i] == k_america && t.s_region[i] == k_america &&
        (t.p_mfgr[i] == k_mfgr_1 || t.p_mfgr[i] == k_mfgr_2)) {
      q4_1_agg_step(t, i, acc);
    }
  }
}

uint16_t q4_1_sse_filter_chunk(const WideTable &t, size_t i) {
  // Initialize constants.
  static __m128i k_america_16u8 = _mm_set1_epi8((char)encode_region("AMERICA"));
  static __m128i k_mfgr_1_16u8 = _mm_set1_epi8((char)encode_p_mfgr("MFGR#1"));
  static __m128i k_mfgr_2_16u8 = _mm_set1_epi8((char)encode_p_mfgr("MFGR#2"));

  // Compute c_region = 'AMERICA'.
  __m128i mask_16u8 = eq_16u8(&t.c_region[i], k_america_16u8);

  // Compute s_region = 'AMERICA'.
  mask_16u8 = _mm_and_si128(mask_16u8, eq_16u8(&t.s_region[i], k_america_16u8));

  // Compute p_mfgr = 'MFGR#1' OR p_mfgr = 'MFGR#2'
  mask_16u8 = _mm_and_si128(mask_16u8, eq_or_16u8(&t.p_mfgr[i], k_mfgr_1_16u8, k_mfgr_2_16u8));

  // Compact the mask.
  return _mm_movemask_epi8(mask_16u8);
}

Q4_1::result_type Q4_1::scalar(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n()), Accumulator(256),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        q4_1_agg_chunk(t, r.begin(), r.end(), acc);
        return acc;
      },
      agg_merge);

  return q4_1_agg_order(acc);
}

Q4_1::result_type Q4_1::sse(const WideTable &t) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 16), Accumulator(256),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 16;
          uint32_t mask = q4_1_sse_filter_chunk(t, j);

          // Perform the aggregation.
          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q4_1_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }

        return acc;
      },
      agg_merge);

  // Process the remaining records.
  q4_1_agg_chunk(t, t.n() / 16 * 16, t.n(), acc);

  return q4_1_agg_order(acc);
}

void Q4_1::filter(const WideTable &t, uint32_t *b) {
  tbb::parallel_for(tbb::blocked_range<size_t>(0, t.n() / 16),
                    [&](const tbb::blocked_range<size_t> &r) {
                      for (size_t i = r.begin(); i < r.end(); ++i) {
                        ((uint16_t *)b)[i] = q4_1_sse_filter_chunk(t, i * 16);
                      }
                    });

  // Process the remaining records.
  static uint8_t k_america = encode_region("AMERICA");
  static uint8_t k_mfgr_1 = encode_p_mfgr("MFGR#1");
  static uint8_t k_mfgr_2 = encode_p_mfgr("MFGR#2");
  for (size_t i = t.n() / 16 * 16; i < t.n(); ++i) {
    if (t.c_region[i] == k_america && t.s_region[i] == k_america &&
        (t.p_mfgr[i] == k_mfgr_1 || t.p_mfgr[i] == k_mfgr_2)) {
      b[i / 32] |= (1 << (i % 32));
    }
  }
}

Q4_1::result_type Q4_1::agg(const WideTable &t, const uint32_t *b) {
  Accumulator acc = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, t.n() / 32 + (t.n() % 32 != 0)), Accumulator(256),
      [&](const tbb::blocked_range<size_t> &r, Accumulator acc) {
        for (size_t i = r.begin(); i < r.end(); ++i) {
          size_t j = i * 32;
          uint32_t mask = b[i];

          while (mask != 0) {
            size_t k = __builtin_ctz(mask);
            q4_1_agg_step(t, j + k, acc);
            mask ^= (1 << k);
          }
        }
        return acc;
      },
      agg_merge);

  return q4_1_agg_order(acc);
}
