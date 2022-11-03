#include "q2_common.hpp"

#include <algorithm>
#include <iomanip>

Q2Row::Q2Row(uint16_t d_year, uint16_t p_brand1, uint32_t sum_lo_revenue)
    : d_year(d_year), p_brand1(p_brand1), sum_lo_revenue(sum_lo_revenue) {}

bool operator==(const Q2Row &a, const Q2Row &b) {
  return a.d_year == b.d_year && a.p_brand1 == b.p_brand1 && a.sum_lo_revenue == b.sum_lo_revenue;
}

std::ostream &operator<<(std::ostream &os, const Q2Row &row) {
  os << row.d_year << '|' << std::setw(4) << row.p_brand1 << '|' << row.sum_lo_revenue;
  return os;
}

std::vector<Q2Row> q2_agg_order(const Accumulator &acc) {
  std::vector<Q2Row> result;

  for (size_t i = 0; i < acc.size(); ++i) {
    if (acc[i].first) {
      result.emplace_back((i >> 6) + 1992, (i & 0b111111) + 40, acc[i].second);
    }
  }

  std::sort(result.begin(), result.end(), [](const Q2Row &a, const Q2Row &b) {
    return a.d_year < b.d_year || (a.d_year == b.d_year && a.p_brand1 < b.p_brand1);
  });

  return result;
}
