add_library(
        ssb_agg
        table.hpp
        load.hpp load.cpp
        encode.hpp encode.cpp
        queries.hpp
        queries/helpers.hpp
        queries/q1_common.hpp queries/q1_common.cpp
        queries/q1_1.hpp queries/q1_1.cpp
        queries/q1_2.hpp queries/q1_2.cpp
        queries/q1_3.hpp queries/q1_3.cpp
        queries/q2_common.hpp queries/q2_common.cpp
        queries/q2_1.hpp queries/q2_1.cpp
        queries/q2_2.hpp queries/q2_2.cpp
        queries/q2_3.hpp queries/q2_3.cpp
        queries/q3_common.hpp queries/q3_common.cpp
        queries/q3_1.hpp queries/q3_1.cpp
        queries/q3_2.hpp queries/q3_2.cpp
        queries/q3_3.hpp queries/q3_3.cpp
        queries/q3_4.hpp queries/q3_4.cpp
        queries/q4_common.hpp queries/q4_common.cpp
        queries/q4_1.hpp queries/q4_1.cpp
        queries/q4_2.hpp queries/q4_2.cpp
        queries/q4_3.hpp queries/q4_3.cpp
)
target_link_libraries(ssb_agg TBB::tbb)
target_include_directories(ssb_agg PRIVATE .)
