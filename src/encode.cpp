#include "encode.hpp"

#include <regex>
#include <unordered_map>

uint8_t encode_d_yearmonth(const std::string &s) {
  static std::unordered_map<std::string, uint8_t> month_dictionary = {
      {"Jan", 0}, {"Feb", 1}, {"Mar", 2}, {"Apr", 3}, {"May", 4},  {"Jun", 5},
      {"Jul", 6}, {"Aug", 7}, {"Sep", 8}, {"Oct", 9}, {"Nov", 10}, {"Dec", 11}};
  auto it = month_dictionary.find(s.substr(0, 3));
  if (it == month_dictionary.end()) {
    throw std::logic_error("could not encode yearmonth from " + s);
  }
  return (it->second << 3) | (std::stoul(s.substr(3, 4)) - 1992);
}

uint8_t encode_p_mfgr(const std::string &s) {
  std::regex r(R"(MFGR#(\d))");
  std::smatch m;
  if (std::regex_match(s, m, r)) {
    return std::stoul(m[1]) - 1;
  } else {
    throw std::logic_error("could not encode p_mfgr from " + s);
  }
}

uint8_t encode_p_category(const std::string &s) {
  std::regex r(R"(MFGR#(\d)(\d))");
  std::smatch m;
  if (std::regex_match(s, m, r)) {
    return (std::stoul(m[1]) - 1) * 5 + (std::stoul(m[2]) - 1);
  } else {
    throw std::logic_error("could not encode p_category from " + s);
  }
}

uint16_t encode_p_brand1(const std::string &s) {
  std::regex r(R"(MFGR#(\d)(\d)(\d{1,2}))");
  std::smatch m;
  if (std::regex_match(s, m, r)) {
    return ((std::stoul(m[1]) - 1) * 5 + (std::stoul(m[2]) - 1)) * 40 + (std::stoul(m[3]) - 1);
  } else {
    throw std::logic_error("could not encode p_brand1 from " + s);
  }
}

uint8_t encode_city(const std::string &s) {
  static std::unordered_map<std::string, uint8_t> nation_dictionary = {
      {"ALGERIA  ", 0},   {"ARGENTINA", 10},  {"BRAZIL   ", 20},  {"CANADA   ", 30},
      {"EGYPT    ", 40},  {"ETHIOPIA ", 50},  {"FRANCE   ", 60},  {"GERMANY  ", 70},
      {"INDIA    ", 80},  {"INDONESIA", 90},  {"IRAN     ", 100}, {"IRAQ     ", 110},
      {"JAPAN    ", 120}, {"JORDAN   ", 130}, {"KENYA    ", 140}, {"MOROCCO  ", 150},
      {"MOZAMBIQU", 160}, {"PERU     ", 170}, {"CHINA    ", 180}, {"ROMANIA  ", 190},
      {"SAUDI ARA", 200}, {"VIETNAM  ", 210}, {"RUSSIA   ", 220}, {"UNITED KI", 230},
      {"UNITED ST", 240}};
  std::regex r("(.{9})(\\d)");
  std::smatch m;
  if (std::regex_match(s, m, r)) {
    auto it = nation_dictionary.find(m[1]);
    if (it == nation_dictionary.end()) {
      throw std::logic_error("could not encode city from " + s);
    }
    return it->second + std::stoul(m[2]);
  } else {
    throw std::logic_error("could not encode city from " + s);
  }
}

uint8_t encode_nation(const std::string &s) {
  static std::unordered_map<std::string, uint8_t> nation_dictionary = {
      {"ALGERIA", 0},       {"ARGENTINA", 1}, {"BRAZIL", 2},  {"CANADA", 3},
      {"EGYPT", 4},         {"ETHIOPIA", 5},  {"FRANCE", 6},  {"GERMANY", 7},
      {"INDIA", 8},         {"INDONESIA", 9}, {"IRAN", 10},   {"IRAQ", 11},
      {"JAPAN", 12},        {"JORDAN", 13},   {"KENYA", 14},  {"MOROCCO", 15},
      {"MOZAMBIQUE", 16},   {"PERU", 17},     {"CHINA", 18},  {"ROMANIA", 19},
      {"SAUDI ARABIA", 20}, {"VIETNAM", 21},  {"RUSSIA", 22}, {"UNITED KINGDOM", 23},
      {"UNITED STATES", 24}};
  auto it = nation_dictionary.find(s);
  if (it == nation_dictionary.end()) {
    throw std::logic_error("could not encode nation from " + s);
  }
  return it->second;
}

uint8_t encode_region(const std::string &s) {
  static std::unordered_map<std::string, uint8_t> region_dictionary = {
      {"AFRICA", 0}, {"AMERICA", 1}, {"ASIA", 2}, {"EUROPE", 3}, {"MIDDLE EAST", 4}};
  auto it = region_dictionary.find(s);
  if (it == region_dictionary.end()) {
    throw std::logic_error("could not encode nation from " + s);
  }
  return it->second;
}
