#ifndef SSB_AGG_TABLE_HPP
#define SSB_AGG_TABLE_HPP

#include <cstdlib>
#include <vector>

struct WideTable {
  [[nodiscard]] size_t n() const { return lo_quantity.size(); }

  std::vector<uint8_t> lo_quantity;
  std::vector<uint16_t> lo_extendedprice;
  std::vector<uint8_t> lo_discount;
  std::vector<uint16_t> lo_revenue;
  std::vector<uint16_t> lo_supplycost;
  std::vector<uint8_t> p_mfgr;
  std::vector<uint8_t> p_category;
  std::vector<uint16_t> p_brand1;
  std::vector<uint8_t> s_city;
  std::vector<uint8_t> s_nation;
  std::vector<uint8_t> s_region;
  std::vector<uint8_t> c_city;
  std::vector<uint8_t> c_nation;
  std::vector<uint8_t> c_region;
  std::vector<uint16_t> d_year;
  std::vector<uint32_t> d_yearmonthnum;
  std::vector<uint8_t> d_yearmonth;
  std::vector<uint8_t> d_month;
  std::vector<uint8_t> d_weeknuminyear;
};

#endif // SSB_AGG_TABLE_HPP
