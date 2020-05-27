#include <pybind11/pybind11.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

void int_binds(py::module &m);
void float_binds(py::module &m);
void char_binds(py::module &m);
void bool_binds(py::module &m);

PYBIND11_MODULE(_skimpy_cpp_ext, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.1.5";

  int_binds(m);
  float_binds(m);
  char_binds(m);
  bool_binds(m);
}
