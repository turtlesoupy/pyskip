cmake_minimum_required(VERSION 3.16)

project(
  skimpy
  VERSION 0.0.1
  DESCRIPTION "Space efficient arrays"
  LANGUAGES CXX
)

#== BUILD OPTIONS ==#

# User options controlling what we build.
option(SKIMPY_ENABLE_PYTHON "Whether to build the happy CPython extension" ON)
option(SKIMPY_BUILD_BENCHMARKS "Whether to build performance benchmarks" ON)
option(SKIMPY_BUILD_BENCHMARKS "Whether to build performance benchmarks" ON)

# Compiler settings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#== BUILD TARGETS ==#

# Add third-party dependencies.
add_subdirectory(third_party)

# Add the skimpy library.
add_library(skimpy INTERFACE)
target_include_directories(skimpy INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(skimpy INTERFACE fmt::fmt)
target_compile_features(skimpy INTERFACE cxx_std_17)

# Add the src targets (e.g. the Python CPP extension).
add_subdirectory(src)

# Build with OpenMP support if available.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(skimpy INTERFACE OpenMP::OpenMP_CXX)
endif()

# fmtlib has propensity to cause "Fatal Error C1128":
# https://tinyurl.com/yaflrbx9
if (MSVC)
  target_compile_options(skimpy INTERFACE /bigobj)
endif()

#== BUILD TESTS ==#

# Add our unit tests.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
