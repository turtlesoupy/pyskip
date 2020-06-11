cmake_minimum_required(VERSION 3.8)

# Add and configure the pyskip python extension target.
SET(
  ALL_BINDS
  binds/bool_binds.cpp 
  binds/char_binds.cpp 
  binds/float_binds.cpp 
  binds/int_binds.cpp
)

add_library(pyskip_ext MODULE pyskip_ext.cpp ${ALL_BINDS})
target_link_libraries(pyskip_ext PRIVATE pybind11::module pyskip)
set_target_properties(
  pyskip_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_pyskip_cpp_ext"
)
