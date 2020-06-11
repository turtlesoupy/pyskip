cmake_minimum_required(VERSION 3.8)

# Add python extension targets.
if(PYSKIP_ENABLE_PYTHON)
  add_subdirectory(cpp_ext)
endif()
