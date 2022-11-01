#ifndef SSB_AGG_Q3_COMMON_HPP
#define SSB_AGG_Q3_COMMON_HPP

#include <cstdint>
#include <ostream>

struct Q3_2Row {
  uint8_t c_city;
  uint8_t s_city;
  uint16_t d_year;
  uint16_t sum_lo_revenue;
  Q3_2Row(uint8_t c_city, uint8_t s_city, uint16_t d_year, uint16_t sum_lo_revenue);
  friend bool operator==(const Q3_2Row &a, const Q3_2Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q3_2Row &row);
};

#endif // SSB_AGG_Q3_COMMON_HPP
