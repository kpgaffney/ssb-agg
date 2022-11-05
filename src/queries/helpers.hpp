#ifndef SSB_AGG_HELPERS_HPP
#define SSB_AGG_HELPERS_HPP

#include <vector>
#include <x86intrin.h>

using Accumulator = std::vector<std::pair<bool, int64_t>>;

inline __m128i eq_16u8(const uint8_t *a, __m128i b) {
  __m128i a_8u16 = _mm_lddqu_si128((__m128i *)a);
  return _mm_cmpeq_epi8(a_8u16, b);
}

inline __m128i eq_or_16u8(const uint8_t *a, __m128i b1, __m128i b2) {
  __m128i a_8u16 = _mm_lddqu_si128((__m128i *)a);
  return _mm_or_si128(_mm_cmpeq_epi8(a_8u16, b1), _mm_cmpeq_epi8(a_8u16, b2));
}

inline __m128i eq_16u16(const uint16_t *a, __m128i b) {
  __m128i a_0_8u16 = _mm_lddqu_si128((__m128i *)a);
  __m128i a_1_8u16 = _mm_lddqu_si128((__m128i *)(a + 8));
  __m128i a_mask_eq_0_8u16 = _mm_cmpeq_epi16(a_0_8u16, b);
  __m128i a_mask_eq_1_8u16 = _mm_cmpeq_epi16(a_1_8u16, b);
  return _mm_packs_epi16(a_mask_eq_0_8u16, a_mask_eq_1_8u16);
}

inline __m128i ge_16u16(const uint16_t *a, __m128i b) {
  __m128i a_0_8u16 = _mm_lddqu_si128((__m128i *)a);
  __m128i a_1_8u16 = _mm_lddqu_si128((__m128i *)(a + 8));
  __m128i a_mask_ge_0_8u16 =
      _mm_or_si128(_mm_cmpgt_epi16(a_0_8u16, b), _mm_cmpeq_epi16(a_0_8u16, b));
  __m128i a_mask_ge_1_8u16 =
      _mm_or_si128(_mm_cmpgt_epi16(a_1_8u16, b), _mm_cmpeq_epi16(a_1_8u16, b));
  return _mm_packs_epi16(a_mask_ge_0_8u16, a_mask_ge_1_8u16);
}

inline __m128i le_16u16(const uint16_t *a, __m128i b) {
  __m128i a_0_8u16 = _mm_lddqu_si128((__m128i *)a);
  __m128i a_1_8u16 = _mm_lddqu_si128((__m128i *)(a + 8));
  __m128i a_mask_le_0_8u16 =
      _mm_or_si128(_mm_cmplt_epi16(a_0_8u16, b), _mm_cmpeq_epi16(a_0_8u16, b));
  __m128i a_mask_le_1_8u16 =
      _mm_or_si128(_mm_cmplt_epi16(a_1_8u16, b), _mm_cmpeq_epi16(a_1_8u16, b));
  return _mm_packs_epi16(a_mask_le_0_8u16, a_mask_le_1_8u16);
}

inline __m128i eq_or_16u16(const uint16_t *a, __m128i b1, __m128i b2) {
  __m128i a_0_8u16 = _mm_lddqu_si128((__m128i *)a);
  __m128i a_1_8u16 = _mm_lddqu_si128((__m128i *)(a + 8));
  __m128i a_mask_eq_0_8u16 =
      _mm_or_si128(_mm_cmpeq_epi16(a_0_8u16, b1), _mm_cmpeq_epi16(a_0_8u16, b2));
  __m128i a_mask_eq_1_8u16 =
      _mm_or_si128(_mm_cmpeq_epi16(a_1_8u16, b1), _mm_cmpeq_epi16(a_1_8u16, b2));
  return _mm_packs_epi16(a_mask_eq_0_8u16, a_mask_eq_1_8u16);
}

inline Accumulator agg_merge(Accumulator a, const Accumulator &b) {
  for (size_t i = 0; i < a.size(); ++i) {
    a[i].first = a[i].first || b[i].first;
    a[i].second += b[i].second;
  }
  return a;
}

#endif // SSB_AGG_HELPERS_HPP
