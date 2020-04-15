#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

template <typename Val>
auto convert_slice(const skimpy::Array<Val>& array, py::slice slice) {
  skimpy::Pos start = 0;
  skimpy::Pos stop = array.len();
  skimpy::Pos stride = 1;

  // Set non-default slice values.
  if (!slice.attr("start").is_none()) {
    start = slice.attr("start").cast<skimpy::Pos>();
  }
  if (!slice.attr("stop").is_none()) {
    stop = slice.attr("stop").cast<skimpy::Pos>();
  }
  if (!slice.attr("step").is_none()) {
    stride = slice.attr("step").cast<skimpy::Pos>();
  }

  // Handle negative values.
  if (start < 0) {
    start = array.len() - start;
  }
  if (stop < 0) {
    stop = array.len() - stop;
  }
  CHECK_ARGUMENT(stride > 0);

  // Truncate start and stop values before returning.
  stop = std::min(stop, array.len());
  start = std::min(start, stop);
  return skimpy::Slice(start, stop, stride);
}

PYBIND11_MODULE(skimpy, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.0.1";

  using IntLazySetter = skimpy::LazySetter<int>;
  py::class_<IntLazySetter>(m, "IntLazySetter")
      .def(py::init<skimpy::Array<int>&>())
      .def("__enter__", [](IntLazySetter& self) {})
      .def(
          "__exit__",
          [](IntLazySetter& self,
             py::object exc_type,
             py::object exc_value,
             py::object traceback) { self.flush(); })
      .def(
          "__setitem__",
          [](IntLazySetter& self, int pos, int val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](IntLazySetter& self, py::slice slice, int val) {
            self.set(convert_slice(self.destination(), slice), val);
          })
      .def(
          "__setitem__",
          [](IntLazySetter& self,
             py::slice slice,
             const skimpy::Array<int>& other) {
            self.set(convert_slice(self.destination(), slice), other);
          });

  using IntArray = skimpy::Array<int>;
  py::class_<IntArray>(m, "IntArray")
      .def(py::init<int, int>())
      .def("__len__", &IntArray::len)
      .def(
          "__repr__",
          [](IntArray& self) {
            std::string ret = "[";
            ret += std::to_string(self.get(0));
            if (self.len() <= 10) {
              for (int i = 1; i < self.len(); i += 1) {
                ret += ", " + std::to_string(self.get(i));
              }
            } else {
              ret += ", " + std::to_string(self.get(1));
              ret += ", " + std::to_string(self.get(2));
              ret += ", " + std::to_string(self.get(3));
              ret += ", ...";
              ret += ", " + std::to_string(self.get(self.len() - 1));
            }
            return ret + "]";
          })
      .def("__getitem__", [](IntArray& self, int pos) { return self.get(pos); })
      .def(
          "__getitem__",
          [](IntArray& self, py::slice slice) {
            return self.get(convert_slice(self, slice));
          })
      .def(
          "__setitem__",
          [](IntArray& self, int pos, int val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](IntArray& self, py::slice slice, int val) {
            self.set(convert_slice(self, slice), val);
          })
      .def(
          "__setitem__",
          [](IntArray& self, py::slice slice, IntArray other) {
            self.set(convert_slice(self, slice), other);
          })
      .def("__neg__", [](const IntArray& self) { return -self; })
      .def("__pos__", [](const IntArray& self) { return +self; })
      .def("__abs__", [](const IntArray& self) { return self.abs(); })
      .def("__invert__", [](const IntArray& self) { return ~self; })
      .def("__add__", [](const IntArray& self, int val) { return self + val; })
      .def("__radd__", [](const IntArray& self, int val) { return val + self; })
      .def(
          "__add__",
          [](const IntArray& self, const IntArray& other) {
            return self + other;
          })
      .def("__sub__", [](const IntArray& self, int val) { return self - val; })
      .def("__rsub__", [](const IntArray& self, int val) { return val - self; })
      .def(
          "__sub__",
          [](const IntArray& self, const IntArray& other) {
            return self - other;
          })
      .def("__mul__", [](const IntArray& self, int val) { return self * val; })
      .def("__rmul__", [](const IntArray& self, int val) { return val * self; })
      .def(
          "__mul__",
          [](const IntArray& self, const IntArray& other) {
            return self * other;
          })
      .def(
          "__floordiv__",
          [](const IntArray& self, int val) { return self / val; })
      .def(
          "__rfloordiv__",
          [](const IntArray& self, int val) { return val / self; })
      .def(
          "__floordiv__",
          [](const IntArray& self, const IntArray& other) {
            return self / other;
          })
      .def("__mod__", [](const IntArray& self, int val) { return self % val; })
      .def("__rmod__", [](const IntArray& self, int val) { return val % self; })
      .def(
          "__mod__",
          [](const IntArray& self, const IntArray& other) {
            return self % other;
          })
      .def(
          "__pow__",
          [](const IntArray& self, int val) { return skimpy::pow(self, val); })
      .def(
          "__rpow__",
          [](const IntArray& self, int val) { return skimpy::pow(val, self); })
      .def(
          "__pow__",
          [](const IntArray& self, const IntArray& other) {
            return self.pow(other);
          })
      .def("__and__", [](const IntArray& self, int val) { return self & val; })
      .def("__rand__", [](const IntArray& self, int val) { return val & self; })
      .def(
          "__and__",
          [](const IntArray& self, const IntArray& other) {
            return self & other;
          })
      .def("__or__", [](const IntArray& self, int val) { return self | val; })
      .def("__ror__", [](const IntArray& self, int val) { return val | self; })
      .def(
          "__or__",
          [](const IntArray& self, const IntArray& other) {
            return self | other;
          })
      .def("__xor__", [](const IntArray& self, int val) { return self ^ val; })
      .def("__xor__", [](const IntArray& self, int val) { return val ^ self; })
      .def(
          "__xor__",
          [](const IntArray& self, const IntArray& other) {
            return self ^ other;
          })
      .def(
          "__lshift__",
          [](const IntArray& self, int val) { return self << val; })
      .def(
          "__lshift__",
          [](const IntArray& self, int val) { return val << self; })
      .def(
          "__lshift__",
          [](const IntArray& self, const IntArray& other) {
            return self << other;
          })
      .def(
          "__rshift__",
          [](const IntArray& self, int val) { return self >> val; })
      .def(
          "__rshift__",
          [](const IntArray& self, int val) { return val >> self; })
      .def(
          "__rshift__",
          [](const IntArray& self, const IntArray& other) {
            return self >> other;
          })
      .def(
          "min",
          [](const IntArray& self, int val) { return skimpy::min(self, val); })
      .def(
          "min",
          [](const IntArray& self, int val) { return skimpy::min(val, self); })
      .def(
          "min",
          [](const IntArray& self, const IntArray& other) {
            return self.min(other);
          })
      .def(
          "max",
          [](const IntArray& self, int val) { return skimpy::max(self, val); })
      .def(
          "max",
          [](const IntArray& self, int val) { return skimpy::max(val, self); })
      .def(
          "max",
          [](const IntArray& self, const IntArray& other) {
            return self.max(other);
          })
      .def("abs", [](const IntArray& self) { return self.abs(); })
      .def("sqrt", [](const IntArray& self) { return self.abs(); })
      .def("exp", [](const IntArray& self) { return self.abs(); });
}
