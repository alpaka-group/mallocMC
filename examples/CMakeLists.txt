cmake_minimum_required(VERSION 3.14...3.22)

project(Examples LANGUAGES CXX)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/vectorAdd
    ${CMAKE_BINARY_DIR}/examples/vectorAdd
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/getAvailableSlots
    ${CMAKE_BINARY_DIR}/examples/getAvailableSlots
)

add_subdirectory(
    ${CMAKE_CURRENT_LIST_DIR}/native-cuda
    ${CMAKE_BINARY_DIR}/examples/native-cuda
)

add_custom_target(
    mallocMCExamples
    DEPENDS mallocMCExampleVectorAdd mallocMCExampleGetAvailableSlots mallocMCExampleNativeCuda
    COMMENT "Shortcut for building all examples."
)
