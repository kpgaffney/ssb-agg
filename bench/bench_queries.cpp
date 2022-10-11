#include "load.hpp"
#include "queries.hpp"

#include <chrono>
#include <iostream>

template <typename F> double time(F &&f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  f();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char **argv) {
  if (argc != 2) {
    throw std::logic_error("expected 1 argument (data directory)");
  }

  std::string data_dir = argv[1];

  WideTable table = load(data_dir);
  std::vector<uint32_t> b(table.n() / 32 + (table.n() % 32 != 0));

  std::cout << q1_1_scalar(table) << std::endl;
  std::cout << q1_1_sse(table) << std::endl;
  q1_1_filter(table, b.data());
  std::cout << q1_1_agg(table, b.data()) << std::endl;

  std::cout << time([&] { q1_1_scalar(table); }) << std::endl;
  std::cout << time([&] { q1_1_sse(table); }) << std::endl;
  std::cout << time([&] { q1_1_filter(table, b.data()); }) << std::endl;
  std::cout << time([&] { q1_1_agg(table, b.data()); }) << std::endl;

  std::fill(b.begin(), b.end(), 0);

  std::cout << q1_2_scalar(table) << std::endl;
  std::cout << q1_2_sse(table) << std::endl;
  q1_2_filter(table, b.data());
  std::cout << q1_2_agg(table, b.data()) << std::endl;

  std::cout << time([&] { q1_2_scalar(table); }) << std::endl;
  std::cout << time([&] { q1_2_sse(table); }) << std::endl;
  std::cout << time([&] { q1_2_filter(table, b.data()); }) << std::endl;
  std::cout << time([&] { q1_2_agg(table, b.data()); }) << std::endl;

  std::fill(b.begin(), b.end(), 0);

  std::cout << q1_3_scalar(table) << std::endl;
  std::cout << q1_3_sse(table) << std::endl;
  q1_3_filter(table, b.data());
  std::cout << q1_3_agg(table, b.data()) << std::endl;

  std::cout << time([&] { q1_3_scalar(table); }) << std::endl;
  std::cout << time([&] { q1_3_sse(table); }) << std::endl;
  std::cout << time([&] { q1_3_filter(table, b.data()); }) << std::endl;
  std::cout << time([&] { q1_2_agg(table, b.data()); }) << std::endl;

  return 0;
}
