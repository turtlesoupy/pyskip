#include <pybind11/pybind11.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

void bind_builders(py::module &m);
void bind_arrays(py::module &m);
void bind_tensors(py::module &m);
void bind_module_methods(py::module &m);

PYBIND11_MODULE(_skimpy_cpp_ext, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.1.5";

  bind_arrays(m);
  bind_builders(m);
  bind_tensors(m);
  bind_module_methods(m);
}
