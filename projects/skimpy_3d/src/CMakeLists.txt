cmake_minimum_required(VERSION 3.16)

add_library(skimpy_3d_cpp_ext MODULE cpp_ext.cpp)
target_link_libraries(skimpy_3d_cpp_ext PRIVATE pybind11::module skimpy)

set_target_properties(
  skimpy_3d_cpp_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_skimpy_3d_cpp_ext"
)
