#ifndef SSB_AGG_Q3_1_HPP
#define SSB_AGG_Q3_1_HPP

#include "q3_common.hpp"
#include "table.hpp"

struct Q3_1Row {
  uint8_t c_nation;
  uint8_t s_nation;
  uint16_t d_year;
  uint16_t sum_lo_revenue;
  Q3_1Row(uint8_t c_nation, uint8_t s_nation, uint16_t d_year, uint16_t sum_lo_revenue);
  friend bool operator==(const Q3_1Row &a, const Q3_1Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q3_1Row &row);
};

class Q3_1 {
public:
  using result_type = std::vector<Q3_1Row>;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q3_1_HPP
