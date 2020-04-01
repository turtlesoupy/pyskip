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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX512")

#== BUILD TARGETS ==#

# Add third-party dependencies.
add_subdirectory(third_party)

# Add the skimpy library.
add_library(skimpy INTERFACE)
target_include_directories(skimpy INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(skimpy INTERFACE fmt::fmt)
target_compile_features(skimpy INTERFACE cxx_std_17)

# Build with OpenMP support if available.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(skimpy INTERFACE OpenMP::OpenMP_CXX)
endif()

# Add python extension targets.
if(SKIMPY_ENABLE_PYTHON)
  add_subdirectory(python)
endif()

#== BUILD TESTS ==#

# Add our unit tests.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
