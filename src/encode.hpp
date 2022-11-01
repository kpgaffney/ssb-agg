#ifndef SSB_AGG_ENCODE_HPP
#define SSB_AGG_ENCODE_HPP

#include <cstdlib>
#include <string>

uint8_t encode_d_yearmonth(const std::string &s);
uint8_t encode_p_mfgr(const std::string &s);
uint8_t encode_p_category(const std::string &s);
uint16_t encode_p_brand1(const std::string &s);
uint8_t encode_city(const std::string &s);
uint8_t encode_nation(const std::string &s);
uint8_t encode_region(const std::string &s);

#endif // SSB_AGG_ENCODE_HPP
