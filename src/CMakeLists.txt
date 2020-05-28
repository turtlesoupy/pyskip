cmake_minimum_required(VERSION 3.8)

# Add python extension targets.
if(SKIMPY_ENABLE_PYTHON)
  add_subdirectory(cpp_ext)
  add_subdirectory(skimpy_bench)
endif()
