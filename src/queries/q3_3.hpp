#ifndef SSB_AGG_Q3_3_HPP
#define SSB_AGG_Q3_3_HPP

#include "q3_common.hpp"
#include "table.hpp"

class Q3_3 {
public:
  using result_type = std::vector<Q3_2Row>;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q3_3_HPP
