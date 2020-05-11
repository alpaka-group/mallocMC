project(mallocMC LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# helper for libs and packages
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/"
    "$ENV{CUDA_ROOT}" "$ENV{BOOST_ROOT}")


################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()


###############################################################################
# CUDA
###############################################################################
if(NOT DEFINED COMPUTE_CAPABILITY)
  set(COMPUTE_CAPABILITY "30")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${COMPUTE_CAPABILITY} -use_fast_math")
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

OPTION(CUDA_OUTPUT_INTERMEDIATE_CODE "Output ptx code" OFF)
if(CUDA_OUTPUT_INTERMEDIATE_CODE)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v --keep")
endif()

SET(CUDA_OPTIMIZATION_TYPE "unset" CACHE STRING "CUDA Optimization")
set_property(CACHE CUDA_OPTIMIZATION_TYPE PROPERTY STRINGS "unset;-G0;-O0;-O1;-O2;-O3")
if(NOT ${CUDA_OPTIMIZATION_TYPE} STREQUAL "unset")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_OPTIMIZATION_TYPE}")
endif()


################################################################################
# Warnings
################################################################################
if(CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow -Wno-unknown-pragmas -Wextra -Wno-unused-parameter -Wno-unused-local-typedefs")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wshadow")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Minform=inform")
endif()


###############################################################################
# Installation
###############################################################################

# copy include folder to install folder
INSTALL(
  DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/include/."
  DESTINATION include
  PATTERN ".git" EXCLUDE
  PATTERN "mallocMC_config.hpp" EXCLUDE
)


###############################################################################
# Executables
###############################################################################
add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example02 mallocMC_Example03 VerifyHeap)

add_executable(mallocMC_Example01 EXCLUDE_FROM_ALL examples/mallocMC_example01.cu)
add_executable(mallocMC_Example02 EXCLUDE_FROM_ALL examples/mallocMC_example02.cu)
add_executable(mallocMC_Example03 EXCLUDE_FROM_ALL examples/mallocMC_example03.cu)
add_executable(VerifyHeap EXCLUDE_FROM_ALL tests/verify_heap.cu)
