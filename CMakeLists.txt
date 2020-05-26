project(mallocMC LANGUAGES CUDA CXX)
cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# helper for libs and packages
set(CMAKE_CUDA_STANDARD 14)
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

# find alpaka
set(mallocMC_ALPAKA_PROVIDER "intern" CACHE STRING "Select which alpaka is used")
set_property(CACHE mallocMC_ALPAKA_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(mallocMC_ALPAKA_PROVIDER)
if(${mallocMC_ALPAKA_PROVIDER} STREQUAL "intern")
    set(alpaka_BUILD_EXAMPLES OFF)
    set(BUILD_TESTING OFF)
    add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/alpaka ${CMAKE_BINARY_DIR}/alpaka)
else()
    find_package(alpaka HINTS $ENV{ALPAKA_ROOT})
endif()

if(NOT TARGET alpaka::alpaka)
    message(FATAL "Required mallocMC dependency alpaka could not be found!")
endif()

# Catch2
set(mallocMC_CATCH2_PROVIDER "intern" CACHE STRING "Select which Catch2 is used")
set_property(CACHE mallocMC_CATCH2_PROVIDER PROPERTY STRINGS "intern;extern")
mark_as_advanced(mallocMC_CATCH2_PROVIDER)
if(${mallocMC_CATCH2_PROVIDER} STREQUAL "intern")
    add_library(Catch2::Catch2 INTERFACE IMPORTED)
    target_include_directories(Catch2::Catch2 INTERFACE ${CMAKE_CURRENT_LIST_DIR}/thirdParty/catch2/include)
else()
    find_package(Catch2 CONFIG REQUIRED)
endif()

###############################################################################
# CUDA
###############################################################################
if(NOT DEFINED COMPUTE_CAPABILITY)
  set(COMPUTE_CAPABILITY "30")
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${COMPUTE_CAPABILITY} -use_fast_math")

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
file(GLOB_RECURSE headers src/include/**)
add_custom_target(mallocMC SOURCES ${headers}) # create a target with the header files for IDE projects
source_group(TREE ${CMAKE_CURRENT_LIST_DIR}/src/include FILES ${headers})

include_directories(${CMAKE_CURRENT_LIST_DIR}/src/include)
add_executable(mallocMC_Example01 EXCLUDE_FROM_ALL examples/mallocMC_example01.cu examples/mallocMC_example01_config.hpp)
add_executable(mallocMC_Example02 EXCLUDE_FROM_ALL examples/mallocMC_example02.cu)
add_executable(mallocMC_Example03 EXCLUDE_FROM_ALL examples/mallocMC_example03.cu)
add_executable(VerifyHeap EXCLUDE_FROM_ALL tests/verify_heap.cu tests/verify_heap_config.hpp)
add_custom_target(examples DEPENDS mallocMC_Example01 mallocMC_Example02 mallocMC_Example03 VerifyHeap)
