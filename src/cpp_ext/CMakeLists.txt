cmake_minimum_required(VERSION 3.8)

# Add and configure the skimpy python extension target.
add_library(skimpy_ext MODULE skimpy_ext.cpp binds/arrays.cpp binds/builders.cpp binds/helpers.cpp binds/module_methods.cpp binds/tensors.cpp)
target_link_libraries(skimpy_ext PRIVATE pybind11::module skimpy)
set_target_properties(
  skimpy_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_skimpy_cpp_ext"
)
