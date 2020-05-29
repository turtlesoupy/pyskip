cmake_minimum_required(VERSION 3.8)

# Set third-party options.
if(CMAKE_BUILD_TYPE MATCHES RELEASE)
  set(ASMJIT_BUILD_RELEASE ON)
else()
  set(ASMJIT_BUILD_DEBUG ON)
endif()
set(ASMJIT_STATIC ON)

# TACO exposes this as a global
set(OPENMP ON)

# Include third-party targets.
add_subdirectory(asmjit)
add_subdirectory(catch2)
add_subdirectory(fmt)
add_subdirectory(pybind11)


if(CMAKE_ENABLE_TACO)
add_subdirectory(taco)
endif()
