cmake_minimum_required(VERSION 3.16)

project(
  pyskip
  VERSION 0.0.1
  DESCRIPTION "Space efficient arrays"
  LANGUAGES CXX
)

#== BUILD OPTIONS ==#

# User options controlling what we build.
option(PYSKIP_ENABLE_PYTHON "Whether to build the happy CPython extension" ON)
option(PYSKIP_BUILD_BENCHMARKS "Whether to build performance benchmarks" ON)

# Compiler settings.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Configuring in debug mode")
elseif(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Configuring in release mode")
elseif(CMAKE_BUILD_TYPE MATCHES RelWithDebInfo)
  message(STATUS "Configuring in release mode with debug flags")
elseif(CMAKE_BUILD_TYPE MATCHES MinRelSize)
  message(STATUS "Configuring in release mode with minimized size")
else()
  message(STATUS "Unrecognized build type - will use cmake defaults")
endif()

if (NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "-g")
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

#== BUILD TARGETS ==#

# Add third-party dependencies.
add_subdirectory(third_party)

# Add the pyskip library.
add_library(pyskip INTERFACE)
target_include_directories(pyskip INTERFACE "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(pyskip INTERFACE fmt::fmt)
target_compile_features(pyskip INTERFACE cxx_std_17)

# Add the src targets (e.g. the Python CPP extension).
add_subdirectory(src)

# Build with OpenMP support if available.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(pyskip INTERFACE OpenMP::OpenMP_CXX)
endif()

# fmtlib has propensity to cause "Fatal Error C1128":
# https://tinyurl.com/yaflrbx9
if (MSVC)
  target_compile_options(pyskip INTERFACE /bigobj)
endif()

#== BUILD TESTS ==#

# Add our unit tests.
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME AND BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
