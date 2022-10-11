#ifndef SSB_AGG_Q1_3_HPP
#define SSB_AGG_Q1_3_HPP

#include "table.hpp"

#include <cstdlib>

uint32_t q1_3_scalar(const WideTable &t);
uint32_t q1_3_sse(const WideTable &t);
void q1_3_filter(const WideTable &t, uint32_t *b);
uint32_t q1_3_agg(const WideTable &t, const uint32_t *b);

#endif // SSB_AGG_Q1_3_HPP
