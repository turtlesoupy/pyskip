#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

PYBIND11_MODULE(skimpy, m) {
  m.doc() = "Space-optimized arrays";

  using IntArray = skimpy::Array<int>;
  py::class_<IntArray>(m, "IntArray")
      .def(py::init<int, int>())
      .def("__len__", &IntArray::len)
      .def("__getitem__", [](IntArray& self, int pos) { return self.get(pos); })
      .def("__setitem__", [](IntArray& self, int pos, int val) {
        self.set(pos, val);
      });
}
