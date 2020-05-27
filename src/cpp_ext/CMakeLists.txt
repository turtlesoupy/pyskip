cmake_minimum_required(VERSION 3.8)

# Add and configure the skimpy python extension target.
add_library(skimpy_ext MODULE skimpy_ext.cpp binds/bool_binds.cpp binds/char_binds.cpp binds/float_binds.cpp binds/int_binds.cpp)
target_link_libraries(skimpy_ext PRIVATE pybind11::module skimpy)
set_target_properties(
  skimpy_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_skimpy_cpp_ext"
)
