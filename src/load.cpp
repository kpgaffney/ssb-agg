#include "load.hpp"
#include "encode.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>

struct PartData {
  uint8_t p_mfgr;
  uint8_t p_category;
  uint16_t p_brand1;
};

struct SupplierData {
  uint8_t s_city;
  uint8_t s_nation;
  uint8_t s_region;
};

struct CustomerData {
  uint8_t c_city;
  uint8_t c_nation;
  uint8_t c_region;
};

struct DateData {
  uint16_t d_year;
  uint32_t d_yearmonthnum;
  uint8_t d_yearmonth;
  uint8_t d_month;
  uint8_t d_week;
};

template <typename F> void read_lines(const std::string &path, F &&f) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error(strerror(errno));
  }

  std::string line;
  std::vector<std::string> line_split;
  while (std::getline(file, line)) {
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, '|')) {
      line_split.push_back(std::move(item));
    }
    f(line_split);
    line_split.clear();
  }
}

WideTable load(const std::string &data_dir) {
  // Load part.
  std::unordered_map<uint32_t, PartData> part_map;
  read_lines(data_dir + "/part.tbl", [&](const std::vector<std::string> &line) {
    uint32_t key = std::stoul(line[0]);
    part_map.emplace(key, PartData{.p_mfgr = encode_p_mfgr(line[2]),
                                   .p_category = encode_p_category(line[3]),
                                   .p_brand1 = encode_p_brand1(line[4])});
  });

  // Load supplier.
  std::unordered_map<uint32_t, SupplierData> supplier_map;
  read_lines(data_dir + "/supplier.tbl", [&](const std::vector<std::string> &line) {
    uint32_t key = std::stoul(line[0]);
    supplier_map.emplace(key, SupplierData{
                                  .s_city = encode_city(line[3]),
                                  .s_nation = encode_nation(line[4]),
                                  .s_region = encode_region(line[5]),
                              });
  });

  // Load customer.
  std::unordered_map<uint32_t, CustomerData> customer_map;
  read_lines(data_dir + "/customer.tbl", [&](const std::vector<std::string> &line) {
    uint32_t key = std::stoul(line[0]);
    customer_map.emplace(key, CustomerData{
                                  .c_city = encode_city(line[3]),
                                  .c_nation = encode_nation(line[4]),
                                  .c_region = encode_region(line[5]),
                              });
  });

  // Load date.
  std::unordered_map<uint32_t, DateData> date_map;
  read_lines(data_dir + "/date.tbl", [&](const std::vector<std::string> &line) {
    uint32_t key = std::stoul(line[0]);
    date_map.emplace(key, DateData{
                              .d_year = (uint16_t)std::stoul(line[4]),
                              .d_yearmonthnum = (uint32_t)std::stoul(line[5]),
                              .d_yearmonth = encode_d_yearmonth(line[6]),
                              .d_month = (uint8_t)std::stoul(line[10]),
                              .d_week = (uint8_t)std::stoul(line[11]),
                          });
  });

  WideTable table;
  read_lines(data_dir + "/lineorder.tbl", [&](const std::vector<std::string> &line) {
    table.lo_quantity.push_back(std::stoul(line[8]));
    table.lo_extendedprice.push_back(std::stoul(line[9]));
    table.lo_discount.push_back(std::stoul(line[11]));
    table.lo_revenue.push_back(std::stoul(line[12]));
    table.lo_supplycost.push_back(std::stoul(line[13]));

    PartData part_data = part_map.at(std::stoul(line[3]));
    table.p_mfgr.push_back(part_data.p_mfgr);
    table.p_category.push_back(part_data.p_category);
    table.p_brand1.push_back(part_data.p_brand1);

    SupplierData supplier_data = supplier_map.at(std::stoul(line[4]));
    table.s_city.push_back(supplier_data.s_city);
    table.s_nation.push_back(supplier_data.s_nation);
    table.s_region.push_back(supplier_data.s_region);

    CustomerData customer_data = customer_map.at(std::stoul(line[2]));
    table.c_city.push_back(customer_data.c_city);
    table.c_nation.push_back(customer_data.c_nation);
    table.c_region.push_back(customer_data.c_region);

    DateData date_data = date_map.at(std::stoul(line[5]));
    table.d_year.push_back(date_data.d_year);
    table.d_yearmonthnum.push_back(date_data.d_yearmonthnum);
    table.d_yearmonth.push_back(date_data.d_yearmonth);
    table.d_month.push_back(date_data.d_month);
    table.d_weeknuminyear.push_back(date_data.d_week);
  });

  return table;
}
