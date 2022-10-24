#ifndef SSB_AGG_Q1_3_HPP
#define SSB_AGG_Q1_3_HPP

#include "table.hpp"

#include <cstdlib>

class Q1_3 {
public:
  using result_type = uint32_t;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q1_3_HPP
