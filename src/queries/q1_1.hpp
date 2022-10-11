#ifndef SSB_AGG_Q1_1_HPP
#define SSB_AGG_Q1_1_HPP

#include "table.hpp"

#include <cstdlib>

uint32_t q1_1_scalar(const WideTable &t);
uint32_t q1_1_sse(const WideTable &t);
void q1_1_filter(const WideTable &t, uint32_t *b);
uint32_t q1_1_agg(const WideTable &t, const uint32_t *b);

#endif // SSB_AGG_Q1_1_HPP
