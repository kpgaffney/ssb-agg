#include "q3_common.hpp"

Q3_2Row::Q3_2Row(uint8_t c_city, uint8_t s_city, uint16_t d_year, uint16_t sum_lo_revenue)
    : c_city(c_city), s_city(s_city), d_year(d_year), sum_lo_revenue(sum_lo_revenue) {}

bool operator==(const Q3_2Row &a, const Q3_2Row &b) {
  return a.c_city == b.c_city && a.s_city == b.s_city && a.d_year == b.d_year &&
         a.sum_lo_revenue == b.sum_lo_revenue;
}

std::ostream &operator<<(std::ostream &os, const Q3_2Row &row) {
  os << (int)row.c_city << '|' << (int)row.s_city << '|' << row.d_year << '|' << row.sum_lo_revenue;
  return os;
}
