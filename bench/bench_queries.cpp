#include "load.hpp"
#include "queries.hpp"

#include <chrono>
#include <iostream>

constexpr size_t trials = 10;

struct BenchResult {
  std::vector<double> t_scalar;
  std::vector<double> t_sse;
  std::vector<double> t_filter;
  std::vector<double> t_agg;
};

template <typename F> double time(F &&f) {
  auto t0 = std::chrono::high_resolution_clock::now();
  f();
  auto t1 = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(t1 - t0).count();
}

void print(uint32_t result) { std::cout << result << std::endl; }

template <typename T> void print(std::vector<T> result) {
  if (result.empty()) {
    return;
  }

  std::cout << result.front() << std::endl;

  if (result.size() > 2) {
    std::cout << "..." << std::endl;
  }

  std::cout << result.back() << std::endl;
}

void print_err(uint32_t result) { std::cerr << result << std::endl; }

template <typename T> void print_err(std::vector<T> result) {
  for (const T &v : result) {
    std::cerr << v << std::endl;
  }
}

template <typename Q> BenchResult bench(const WideTable &table) {
  std::vector<uint32_t> b(table.n() / 32 + (table.n() % 32 != 0));

  typename Q::result_type scalar_result = Q::scalar(table);
  typename Q::result_type sse_result = Q::sse(table);
  Q::filter(table, b.data());
  typename Q::result_type agg_result = Q::agg(table, b.data());

  if (scalar_result == sse_result && sse_result == agg_result) {
    std::cout << "Methods agree." << std::endl;
    print(scalar_result);
  } else {
    std::cerr << "Methods disagree." << std::endl;
    std::cerr << "Scalar result:" << std::endl;
    print_err(scalar_result);
    std::cerr << "SSE result:" << std::endl;
    print_err(sse_result);
    std::cerr << "Agg result:" << std::endl;
    print_err(agg_result);
  }

  BenchResult result;

  for (size_t t = 0; t < trials; ++t) {
    result.t_scalar.push_back(time([&] { Q::scalar(table); }));
  }

  for (size_t t = 0; t < trials; ++t) {
    result.t_sse.push_back(time([&] { Q::sse(table); }));
  }

  for (size_t t = 0; t < trials; ++t) {
    result.t_filter.push_back(time([&] { Q::filter(table, b.data()); }));
  }

  for (size_t t = 0; t < trials; ++t) {
    result.t_agg.push_back(time([&] { Q::agg(table, b.data()); }));
  }

  return result;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    throw std::logic_error("expected 1 argument (data directory)");
  }

  std::string data_dir = argv[1];

  WideTable table = load(data_dir);

  BenchResult q1_1_result = bench<Q1_1>(table);
  BenchResult q1_2_result = bench<Q1_2>(table);
  BenchResult q1_3_result = bench<Q1_3>(table);
  BenchResult q2_1_result = bench<Q2_1>(table);
  BenchResult q2_2_result = bench<Q2_2>(table);
  BenchResult q2_3_result = bench<Q2_3>(table);

  std::cout << q1_1_result.t_scalar.back() << std::endl;
  std::cout << q1_1_result.t_sse.back() << std::endl;
  std::cout << q1_1_result.t_filter.back() << std::endl;
  std::cout << q1_1_result.t_agg.back() << std::endl;

  std::cout << q2_1_result.t_scalar.back() << std::endl;
  std::cout << q2_1_result.t_sse.back() << std::endl;
  std::cout << q2_1_result.t_filter.back() << std::endl;
  std::cout << q2_1_result.t_agg.back() << std::endl;

  std::cout << q2_3_result.t_scalar.back() << std::endl;
  std::cout << q2_3_result.t_sse.back() << std::endl;
  std::cout << q2_3_result.t_filter.back() << std::endl;
  std::cout << q2_3_result.t_agg.back() << std::endl;

  return 0;
}
