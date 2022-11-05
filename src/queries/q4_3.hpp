#ifndef SSB_AGG_Q4_3_HPP
#define SSB_AGG_Q4_3_HPP

#include "table.hpp"

struct Q4_3Row {
  uint16_t d_year;
  uint8_t s_city;
  uint16_t p_brand1;
  int64_t sum_profit;
  Q4_3Row(uint16_t d_year, uint8_t s_city, uint16_t p_brand1, int64_t sum_profit);
  friend bool operator==(const Q4_3Row &a, const Q4_3Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q4_3Row &row);
};

class Q4_3 {
public:
  using result_type = std::vector<Q4_3Row>;
  static result_type scalar(const WideTable &t);
  static result_type sse(const WideTable &t);
  static void filter(const WideTable &t, uint32_t *b);
  static result_type agg(const WideTable &t, const uint32_t *b);
};

#endif // SSB_AGG_Q4_3_HPP
