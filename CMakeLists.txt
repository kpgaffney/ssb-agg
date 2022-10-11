cmake_minimum_required(VERSION 3.14)
project(ssb_agg)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)

FetchContent_Declare(
        tbb
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
        GIT_TAG v2021.6.0
)
set(TBB_TEST OFF)
FetchContent_MakeAvailable(tbb)

add_subdirectory(src)
add_subdirectory(bench)
