#include "load.hpp"
#include "queries.hpp"

#include <chrono>
#include <fstream>
#include <iostream>

constexpr size_t trials = 10;

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

template <typename Q>
void bench(const std::string &name, const WideTable &table, std::ofstream &f) {
  std::cout << name << std::endl;

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

  for (size_t t = 0; t < trials; ++t) {
    double t_scalar = time([&] { Q::scalar(table); });
    double t_sse = time([&] { Q::sse(table); });
    double t_filter = time([&] { Q::filter(table, b.data()); });
    double t_agg = time([&] { Q::agg(table, b.data()); });

    f << name << ',' << t << ',' << t_scalar << ',' << t_sse << ',' << t_filter << ',' << t_agg
      << std::endl;
  }
}

int main(int argc, char **argv) {
  if (argc != 2) {
    throw std::logic_error("expected 1 argument (data directory)");
  }

  std::string data_dir = argv[1];

  WideTable table = load(data_dir);
  std::ofstream f("results.csv");
  f << "Query,Trial,Scalar,SSE,Filter,Agg" << std::endl;

  bench<Q1_1>("Q1.1", table, f);
  bench<Q1_2>("Q1.2", table, f);
  bench<Q1_3>("Q1.3", table, f);
  bench<Q2_1>("Q2.1", table, f);
  bench<Q2_2>("Q2.2", table, f);
  bench<Q2_3>("Q2.3", table, f);
  bench<Q3_1>("Q3.1", table, f);
  bench<Q3_2>("Q3.2", table, f);
  bench<Q3_3>("Q3.3", table, f);
  bench<Q3_4>("Q3.4", table, f);
  bench<Q4_1>("Q4.1", table, f);
  bench<Q4_2>("Q4.2", table, f);
  bench<Q4_3>("Q4.3", table, f);

  f.close();

  return 0;
}
