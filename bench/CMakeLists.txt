add_executable(bench_queries bench_queries.cpp)
target_link_libraries(bench_queries ssb_agg)
target_include_directories(bench_queries PRIVATE ../src)
