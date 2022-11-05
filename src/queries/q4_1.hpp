#ifndef SSB_AGG_Q4_1_HPP
#define SSB_AGG_Q4_1_HPP

#include "table.hpp"

struct Q4_1Row {
  uint16_t d_year;
  uint8_t c_nation;
  int64_t sum_profit;
  Q4_1Row(uint16_t d_year, uint8_t c_nation, int64_t sum_profit);
  friend bool operator==(const Q4_1Row &a, const Q4_1Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q4_1Row &row);
};

class Q4_1 {
public:
  using result_type = std::vector<Q4_1Row>;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q4_1_HPP
