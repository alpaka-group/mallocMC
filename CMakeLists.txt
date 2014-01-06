project(ScatterAlloc)
cmake_minimum_required(VERSION 2.6)
set(CUDA_NVCC_FLAGS "-arch=sm_20;-use_fast_math;")
find_package(CUDA REQUIRED)
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

cuda_add_executable(ScatterAllocExample 
                    example.cu 
                    ${CMAKE_CURRENT_SOURCE_DIR}/tools/heap.cuh
					${CMAKE_CURRENT_SOURCE_DIR}/tools/heap_impl.cuh 
					${CMAKE_CURRENT_SOURCE_DIR}/tools/utils.h)