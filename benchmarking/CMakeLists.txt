cmake_minimum_required(VERSION 3.14...3.22)

project(mallocMCBenchmarking LANGUAGES CXX)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/maxCollisionHashFun
    ${CMAKE_BINARY_DIR}/benchmarking/maxCollisionHashFun
)

add_custom_target(
    mallocMCBenchmarking
    DEPENDS mallocMCBenchmarkingMaxCollisionHashFun
    COMMENT "Shortcut for building all benchmarking."
)