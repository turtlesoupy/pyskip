cmake_minimum_required(VERSION 3.8)

find_package(OpenMP REQUIRED)

add_library(skimpy_bench_cpp_ext MODULE cpp_ext.cpp)
target_compile_options(skimpy_bench_cpp_ext PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(skimpy_bench_cpp_ext PRIVATE pybind11::module ${OpenMP_CXX_FLAGS})
set_target_properties(
  skimpy_bench_cpp_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_skimpy_bench_cpp_ext"
)
