#ifndef SSB_AGG_Q2_COMMON_HPP
#define SSB_AGG_Q2_COMMON_HPP

#include <cstdlib>
#include <ostream>
#include <unordered_map>
#include <vector>

using q2_acc_type = std::unordered_map<uint32_t, uint32_t>;

struct Q2Row {
  uint16_t d_year;
  uint16_t p_brand1;
  uint32_t sum_lo_revenue;
  Q2Row(uint16_t d_year, uint16_t p_brand1, uint32_t sum_lo_revenue);
  friend bool operator==(const Q2Row &a, const Q2Row &b);
  friend std::ostream &operator<<(std::ostream &os, const Q2Row &row);
};

std::vector<Q2Row> q2_agg_order(const q2_acc_type &acc);

q2_acc_type agg_merge(q2_acc_type a, const q2_acc_type &b);

#endif // SSB_AGG_Q2_COMMON_HPP
