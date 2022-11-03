#ifndef SSB_AGG_Q4_2_HPP
#define SSB_AGG_Q4_2_HPP

#include "table.hpp"

struct Q4_2Row {
  uint16_t d_year;
  uint8_t s_nation;
  uint8_t p_category;
  size_t sum_profit;
  Q4_2Row(uint16_t d_year, uint8_t s_nation, uint8_t p_category, uint64_t sum_profit);
  friend bool operator==(const Q4_2Row &a, const Q4_2Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q4_2Row &row);
};

class Q4_2 {
public:
  using result_type = std::vector<Q4_2Row>;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q4_2_HPP
