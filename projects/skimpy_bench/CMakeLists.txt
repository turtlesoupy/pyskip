cmake_minimum_required(VERSION 3.16)

project(
  skimpy_bench
  VERSION 0.0.1
  DESCRIPTION "Comprehensive benchmarking tests for skimpy arrays"
  LANGUAGES CXX
)

#== BUILD OPTIONS ==#

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
get_filename_component(SKIMPY_DIR ../../ ABSOLUTE)
add_subdirectory(${SKIMPY_DIR} ${SKIMPY_DIR}/build)

# Add the src targets (e.g. the Python CPP extension).
add_subdirectory(src)
