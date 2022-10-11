add_library(
        ssb_agg
        table.hpp
        load.hpp load.cpp
        encode.hpp encode.cpp
        queries.hpp
        queries/q1_1.hpp queries/q1_1.cpp
        queries/q1_2.hpp queries/q1_2.cpp
        queries/q1_3.hpp queries/q1_3.cpp
)
target_link_libraries(ssb_agg TBB::tbb)
target_include_directories(ssb_agg PRIVATE .)
