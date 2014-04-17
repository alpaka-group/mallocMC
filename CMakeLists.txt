project(ScatterAlloc)
cmake_minimum_required(VERSION 2.8.5)

# helper for libs and packages
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


###############################################################################
# CUDA
###############################################################################
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "-arch=sm_20;-use_fast_math;")
set(CUDA_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
cuda_include_directories(${CUDA_INCLUDE_DIRS})

OPTION(CUDA_OUTPUT_INTERMEDIATE_CODE "Output ptx code" OFF)
if(CUDA_OUTPUT_INTERMEDIATE_CODE)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};-Xptxas;-v;--keep")
endif(CUDA_OUTPUT_INTERMEDIATE_CODE)

SET(CUDA_OPTIMIZATION_TYPE "unset" CACHE STRING "CUDA Optimization")
set_property(CACHE CUDA_OPTIMIZATION_TYPE PROPERTY STRINGS "unset;-G0;-O0;-O1;-O2;-O3")
if(NOT ${CUDA_OPTIMIZATION_TYPE} STREQUAL  "unset")
  set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS};${CUDA_OPTIMIZATION_TYPE}")
endif()


###############################################################################
# Boost
###############################################################################
find_package(Boost 1.49.0 REQUIRED)
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
set(LIBS ${LIBS} ${Boost_LIBRARIES})

# nvcc + boost 1.55 work around
if(Boost_VERSION EQUAL 105500)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NOINLINE='__attribute__ ((noinline))'")
endif(Boost_VERSION EQUAL 105500)


###############################################################################
# Executables
###############################################################################
cuda_add_executable(PolicyAllocExample
                    examples/policy_example.cu )
cuda_add_executable(PolicyAllocExample01
                    examples/policy_example01.cu )
cuda_add_executable(PolicyAllocExample02
                    examples/policy_example02.cu )
cuda_add_executable(PolicyAllocExample03
                    examples/newdelete_example/newdelete.cu )

target_link_libraries(PolicyAllocExample ${LIBS})
target_link_libraries(PolicyAllocExample01 ${LIBS})
target_link_libraries(PolicyAllocExample02 ${LIBS})
target_link_libraries(PolicyAllocExample03 ${LIBS})

