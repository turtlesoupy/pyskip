cmake_minimum_required(VERSION 3.16)

find_package(OpenMP REQUIRED)

OPTION(PYSKIP_BENCH_ENABLE_TACO "Enable taco benchmarking" ON)

if((CMAKE_CXX_COMPILER_ID MATCHES "GNU") OR (CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
  SET(AVX_COMPILE_OPTION -mavx2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  SET(AVX_COMPILE_OPTION /QxAVX2)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
  SET(AVX_COMPILE_OPTION /arch:AVX2)
  OPTION(PYSKIP_BENCH_ENABLE_TACO OFF)
endif()

if(PYSKIP_BENCH_ENABLE_TACO)
SET(TACO_LINK taco_bench.cpp)
else()
SET(TACO_LINK)
endif()

add_library(pyskip_bench_cpp_ext MODULE cpp_ext.cpp ${TACO_LINK})
target_compile_options(pyskip_bench_cpp_ext PRIVATE ${OpenMP_CXX_FLAGS} ${AVX_COMPILE_OPTION})
target_link_libraries(pyskip_bench_cpp_ext PRIVATE pybind11::module OpenMP::OpenMP_CXX)

if(PYSKIP_BENCH_ENABLE_TACO)
  target_compile_definitions(pyskip_bench_cpp_ext PRIVATE PYSKIP_BENCH_ENABLE_TACO=1)
  target_link_libraries(pyskip_bench_cpp_ext PRIVATE taco)
  target_include_directories(pyskip_bench_cpp_ext PRIVATE ${CMAKE_SOURCE_DIR}/third_party/taco/include)
endif()

set_target_properties(
  pyskip_bench_cpp_ext
  PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    OUTPUT_NAME "_pyskip_bench_cpp_ext"
)
