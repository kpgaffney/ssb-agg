#ifndef SSB_AGG_Q2_COMMON_HPP
#define SSB_AGG_Q2_COMMON_HPP

#include <cstdlib>
#include <ostream>
#include <unordered_map>
#include <vector>

struct Q2Row {
  uint16_t d_year;
  uint16_t p_brand1;
  uint32_t sum_lo_revenue;
  Q2Row(uint16_t d_year, uint16_t p_brand1, uint32_t sum_lo_revenue);
  friend bool operator==(const Q2Row &a, const Q2Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q2Row &row);
};

std::vector<Q2Row> q2_agg_order(const std::vector<uint16_t> &acc);

std::vector<uint16_t> agg_merge(std::vector<uint16_t> a, const std::vector<uint16_t> &b);

#endif // SSB_AGG_Q2_COMMON_HPP
