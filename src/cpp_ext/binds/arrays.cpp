#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/array.hpp>
#include "helpers.hpp"

namespace py = pybind11;

template <typename Val>
void bind_array_class(py::module& m, const char* class_name) {
  using Array = skimpy::Array<Val>;
  auto& cls =
      py::class_<Array>(m, class_name)
          .def(py::init([](skimpy::Pos span, Val fill) {
            return skimpy::make_array<Val>(span, fill);
          }))
          .def("__len__", &Array::len)
          .def("__repr__", &Array::repr)
          .def("clone", &Array::clone)
          .def("eval", &Array::eval)
          .def("dumps", &Array::str)
          .def(
              "tensor",
              [](Array& self) { skimpy::make_tensor<1>({self.len()}, self); })
          .def(
              "to_numpy",
              [](Array& self) {
                int size;
                Val* buffer;
                skimpy::to_buffer(self, &size, &buffer);
                return py::array_t<Val>(size, buffer);
              })
          .def(
              "__getitem__",
              [](Array& self, skimpy::Pos pos) { return self.get(pos); })
          .def(
              "__getitem__",
              [](Array& self, py::slice slice) {
                return self.get(convert_slice(self.len(), slice));
              })
          .def(
              "__setitem__",
              [](Array& self, skimpy::Pos pos, Val val) { self.set(pos, val); })
          .def(
              "__setitem__",
              [](Array& self, py::slice slice, Val val) {
                self.set(convert_slice(self.len(), slice), val);
              })
          .def(
              "__setitem__",
              [](Array& self, py::slice slice, const Array& other) {
                self.set(convert_slice(self.len(), slice), other);
              });

  // Add numerical operations.
  if constexpr (std::is_same_v<Val, float> || std::is_same_v<Val, int>) {
    cls.def("__neg__", [](const Array& self) { return -self; })
        .def("__pos__", [](const Array& self) { return +self; })
        .def("__abs__", [](const Array& self) { return skimpy::abs(self); })
        .def("__add__", [](const Array& self, Val val) { return self + val; })
        .def("__radd__", [](const Array& self, Val val) { return val + self; })
        .def(
            "__add__",
            [](const Array& self, const Array& other) { return self + other; })
        .def("__sub__", [](const Array& self, Val val) { return self - val; })
        .def("__rsub__", [](const Array& self, Val val) { return val - self; })
        .def(
            "__sub__",
            [](const Array& self, const Array& other) { return self - other; })
        .def("__mul__", [](const Array& self, Val val) { return self * val; })
        .def("__rmul__", [](const Array& self, Val val) { return val * self; })
        .def(
            "__mul__",
            [](const Array& self, const Array& other) { return self * other; })
        .def(
            "__floordiv__",
            [](const Array& self, Val val) { return self / val; })
        .def(
            "__rfloordiv__",
            [](const Array& self, Val val) { return val / self; })
        .def(
            "__floordiv__",
            [](const Array& self, const Array& other) { return self / other; })
        .def(
            "__truediv__",
            [](const Array& self, Val val) {
              return skimpy::cast<float>(self) / static_cast<float>(val);
            })
        .def(
            "__rtruediv__",
            [](const Array& self, Val val) {
              return static_cast<float>(val) / skimpy::cast<float>(self);
            })
        .def(
            "__truediv__",
            [](const Array& self, const Array& other) {
              return skimpy::cast<float>(self) / skimpy::cast<float>(other);
            })
        .def("__mod__", [](const Array& self, Val val) { return self % val; })
        .def("__rmod__", [](const Array& self, Val val) { return val % self; })
        .def(
            "__mod__",
            [](const Array& self, const Array& other) { return self % other; })
        .def(
            "__pow__",
            [](const Array& self, Val val) { return skimpy::pow(self, val); })
        .def(
            "__pow__",
            [](const Array& self, Val val) { return skimpy::pow(val, self); })
        .def(
            "__pow__",
            [](const Array& self, const Array& other) {
              return skimpy::pow(self, other);
            })
        .def("abs", [](const Array& self) { return skimpy::abs(self); })
        .def("sqrt", [](const Array& self) { return skimpy::sqrt(self); })
        .def("exp", [](const Array& self) { return skimpy::exp(self); });
  }

  // Add bitwise operations.
  if constexpr (std::is_same_v<Val, int>) {
    cls.def("__invert__", [](const Array& self) { return ~self; })
        .def("__and__", [](const Array& self, Val val) { return self & val; })
        .def("__rand__", [](const Array& self, Val val) { return val & self; })
        .def(
            "__and__",
            [](const Array& self, const Array& other) { return self & other; })
        .def("__or__", [](const Array& self, Val val) { return self | val; })
        .def("__ror__", [](const Array& self, Val val) { return val | self; })
        .def(
            "__or__",
            [](const Array& self, const Array& other) { return self | other; })
        .def("__xor__", [](const Array& self, Val val) { return self ^ val; })
        .def("__xor__", [](const Array& self, Val val) { return val ^ self; })
        .def(
            "__xor__",
            [](const Array& self, const Array& other) { return self ^ other; })
        .def(
            "__lshift__",
            [](const Array& self, Val val) { return self << val; })
        .def(
            "__lshift__",
            [](const Array& self, Val val) { return val << self; })
        .def(
            "__lshift__",
            [](const Array& self, const Array& other) { return self << other; })
        .def(
            "__rshift__",
            [](const Array& self, Val val) { return self >> val; })
        .def(
            "__rshift__",
            [](const Array& self, Val val) { return val >> self; })
        .def("__rshift__", [](const Array& self, const Array& other) {
          return self >> other;
        });
  }

  // Add logical comparison operations.
  // TODO: Add remaining operations.
  cls.def(
         "min",
         [](const Array& self, Val val) { return skimpy::min(self, val); })
      .def(
          "min",
          [](const Array& self, Val val) { return skimpy::min(val, self); })
      .def(
          "min",
          [](const Array& self, const Array& other) {
            return skimpy::min(self, other);
          })
      .def(
          "max",
          [](const Array& self, Val val) { return skimpy::max(self, val); })
      .def(
          "max",
          [](const Array& self, Val val) { return skimpy::max(val, self); })
      .def("max", [](const Array& self, const Array& other) {
        return skimpy::min(self, other);
      });

  // Add conversion operations.
  cls.def("int", [](const Array& self) { return skimpy::cast<int>(self); })
      .def("char", [](const Array& self) { return skimpy::cast<char>(self); })
      .def("float", [](const Array& self) { return skimpy::cast<float>(self); })
      .def("bool", [](const Array& self) { return skimpy::cast<bool>(self); });
}

void bind_arrays(py::module &m) {
  bind_array_class<int>(m, "IntArray");
  bind_array_class<float>(m, "FloatArray");
  bind_array_class<char>(m, "CharArray");
  bind_array_class<bool>(m, "BoolArray");
}