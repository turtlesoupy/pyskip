#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

void bind_module_methods(py::module &m) {
  // Module routines
  // TODO: Should these conversion routines output a Tensor?
  m.def("from_numpy", [](py::array_t<int>& array) {
    return skimpy::from_buffer(array.size(), array.data());
  });
  m.def("from_numpy", [](py::array_t<float>& array) {
    return skimpy::from_buffer(array.size(), array.data());
  });
  m.def("from_numpy", [](py::array_t<char>& array) {
    return skimpy::from_buffer(array.size(), array.data());
  });
  m.def("from_numpy", [](py::array_t<bool>& array) {
    return skimpy::from_buffer(array.size(), array.data());
  });
}