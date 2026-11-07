cmake_minimum_required(VERSION 3.22)

project(microtorch_cpp LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_library(microtorch_cpp STATIC
    src/engine.cpp
)

target_include_directories(microtorch_cpp
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(microtorch_cpp PUBLIC Eigen3::Eigen)

enable_testing()

add_executable(microtorch_cpp_tests
    tests/engine_test.cpp
)

target_link_libraries(microtorch_cpp_tests
    PRIVATE
        microtorch_cpp
        GTest::gtest_main
)

add_test(NAME microtorch_cpp_tests COMMAND microtorch_cpp_tests)
