#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

auto convert_slice(skimpy::Pos length, py::slice slice) {
  skimpy::Pos start = 0;
  skimpy::Pos stop = length;
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
    start = length + start;
  }
  if (stop < 0) {
    stop = length + stop;
  }
  CHECK_ARGUMENT(stride > 0);

  // Truncate start and stop values before returning.
  stop = std::min(stop, length);
  start = std::min(start, stop);
  return skimpy::Slice(start, stop, stride);
}

PYBIND11_MODULE(skimpy, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.0.1";

  // Module routines
  m.def("from_numpy", [](py::array_t<int>& array) {
    auto store = skimpy::conv::to_store(array.size(), array.data());
    return skimpy::Array<int>(store);
  });

  // ArrayBuilder class for int value types
  using IntArrayBuilder = skimpy::ArrayBuilder<int>;
  py::class_<IntArrayBuilder>(m, "IntArrayBuilder")
      .def(py::init<int, int>())
      .def(py::init<skimpy::Array<int>>())
      .def("__len__", &IntArrayBuilder::len)
      .def(
          "__repr__",
          [](IntArrayBuilder& self) {
            return fmt::format("ArrayBuilder<int>({})", self.build().str());
          })
      .def(
          "__setitem__",
          [](IntArrayBuilder& self, int pos, int val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](IntArrayBuilder& self, py::slice slice, int val) {
            self.set(convert_slice(self.len(), slice), val);
          })
      .def(
          "__setitem__",
          [](IntArrayBuilder& self,
             py::slice slice,
             const skimpy::Array<int>& other) {
            self.set(convert_slice(self.len(), slice), other);
          })
      .def("build", &IntArrayBuilder::build);

  // Array class for int value types
  using IntArray = skimpy::Array<int>;
  py::class_<IntArray>(m, "IntArray")
      .def(py::init<int, int>())
      .def("__len__", &IntArray::len)
      .def(
          "__repr__",
          [](IntArray& self) {
            return fmt::format("Array<int>({})", self.str());
          })
      .def("clone", &IntArray::clone)
      .def("eval", [](IntArray& self) { return IntArray(self.store()); })
      .def(
          "dumps",
          [](IntArray& self) { return skimpy::conv::to_string(*self.store()); })
      .def(
          "to_numpy",
          [](IntArray& self) {
            auto store = self.store();
            int* buffer = new int[store->span()];
            skimpy::conv::to_buffer(*store, buffer);
            return py::array_t<int>(store->span(), buffer);
          })
      .def("__getitem__", [](IntArray& self, int pos) { return self.get(pos); })
      .def(
          "__getitem__",
          [](IntArray& self, py::slice slice) {
            return self.get(convert_slice(self.len(), slice));
          })
      .def(
          "__setitem__",
          [](IntArray& self, int pos, int val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](IntArray& self, py::slice slice, int val) {
            self.set(convert_slice(self.len(), slice), val);
          })
      .def(
          "__setitem__",
          [](IntArray& self, py::slice slice, IntArray other) {
            self.set(convert_slice(self.len(), slice), other);
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
