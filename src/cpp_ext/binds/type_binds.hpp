#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

inline auto convert_band(skimpy::Pos length, py::slice slice) {
  CHECK_ARGUMENT(slice.attr("step").is_none());
  skimpy::Pos start = 0;
  skimpy::Pos stop = length;

  // Set non-default slice values.
  if (!slice.attr("start").is_none()) {
    start = slice.attr("start").cast<skimpy::Pos>();
  }
  if (!slice.attr("stop").is_none()) {
    stop = slice.attr("stop").cast<skimpy::Pos>();
  }

  // Handle negative values.
  if (start < 0) {
    start = length + start;
  }
  if (stop < 0) {
    stop = length + stop;
  }

  // Truncate start and stop values before returning.
  stop = std::min(stop, length);
  start = std::min(start, stop);
  return skimpy::Band(start, stop);
}

inline auto convert_slice(skimpy::Pos length, py::slice slice) {
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

template <size_t dim>
inline auto convert_tensor_slice(
    skimpy::TensorShape<dim> shape, std::array<py::slice, dim> slices) {
  std::array<std::array<skimpy::Pos, 3>, dim> components;
  for (int i = 0; i < dim; i += 1) {
    auto slice = convert_slice(shape[i], slices[i]);
    components[i][0] = slice.start;
    components[i][1] = slice.stop;
    components[i][2] = slice.stride;
  }
  return skimpy::TensorSlice<dim>(std::move(components));
}

template <size_t dim>
inline auto get_tensor_shape(const skimpy::TensorShape<dim>& shape) {
  std::array<int, dim> ret;
  for (int i = 0; i < dim; i += 1) {
    ret[i] = shape[i];
  }
  return std::tuple_cat(ret);
}

template <typename Val>
inline void bind_builder_class(py::module& m, const char* class_name) {
  using Builder = skimpy::ArrayBuilder<Val>;
  py::class_<Builder>(m, class_name)
      .def(py::init<skimpy::Pos, Val>())
      .def(py::init<skimpy::Array<Val>>())
      .def("__len__", &Builder::len)
      .def("__repr__", &Builder::repr)
      .def(
          "__setitem__",
          [](Builder& self, skimpy::Pos pos, Val val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](Builder& self, py::slice slice, Val val) {
            self.set(convert_band(self.len(), slice), val);
          })
      .def(
          "__setitem__",
          [](Builder& self, py::slice slice, const skimpy::Array<Val>& other) {
            self.set(convert_band(self.len(), slice), other);
          })
      .def("dumps", &Builder::str)
      .def("build", &Builder::build);
}

template <typename Val>
inline void bind_array_class(py::module& m, const char* class_name) {
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
          .def("rle_length", &Array::rleLength)
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
  cls.def("__eq__", [](const Array& self, Val val) { return self == val; })
      .def("__ne__", [](const Array& self, Val val) { return self != val; })
      .def("__le__", [](const Array& self, Val val) { return self <= val; })
      .def("__lt__", [](const Array& self, Val val) { return self < val; })
      .def("__ge__", [](const Array& self, Val val) { return self >= val; })
      .def("__gt__", [](const Array& self, Val val) { return self > val; })
      .def("coalesce", [](const Array& self, Val val) {
        return skimpy::coalesce(self, val);
      })
      .def("coalesce", [](const Array& self, const Array& other) {
        return skimpy::coalesce(self, other);
      });

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

template <size_t dim, typename Val>
inline void bind_tensor_class(py::module& m, const char* class_name) {
  using Tensor = skimpy::Tensor<dim, Val>;
  py::class_<Tensor>(m, class_name)
      .def(py::init([](const std::array<skimpy::Pos, dim>& s, Val v) {
        return skimpy::make_tensor<dim, Val>(skimpy::make_shape<dim>(s), v);
      }))
      .def(py::init([](const std::array<skimpy::Pos, dim> s,
                       const skimpy::Array<Val>& v) {
        return skimpy::make_tensor<dim, Val>(skimpy::make_shape<dim>(s), v);
      }))
      .def("__len__", &Tensor::len)
      .def("__repr__", &Tensor::repr)
      .def("clone", &Tensor::clone)
      .def("eval", &Tensor::eval)
      .def("dumps", &Tensor::str)
      .def("array", &Tensor::array)
      .def(
          "shape",
          [](const Tensor& self) { return get_tensor_shape(self.shape()); })
      .def(
          "__getitem__",
          [](Tensor& self, std::array<skimpy::Pos, dim> pos) {
            return self.get(pos);
          })
      .def(
          "__getitem__",
          [](Tensor& self, std::array<py::slice, dim> slices) {
            return self.get(convert_tensor_slice(self.shape(), slices));
          })
      .def(
          "__setitem__",
          [](Tensor& self, std::array<skimpy::Pos, dim> pos, Val val) {
            self.set(pos, val);
          })
      .def(
          "__setitem__",
          [](Tensor& self, std::array<py::slice, dim> slices, Val val) {
            self.set(convert_tensor_slice(self.shape(), slices), val);
          })
      .def(
          "__setitem__",
          [](Tensor& self,
             std::array<py::slice, dim> slices,
             const Tensor& other) {
            self.set(convert_tensor_slice(self.shape(), slices), other);
          });
}

template <typename Val>
inline void type_binds(
    py::module& m,
    const std::string& friendlyName,
    const std::string& templatePostfix) {
  bind_builder_class<Val>(m, (friendlyName + "Builder").c_str());
  bind_array_class<Val>(m, (friendlyName + "Array").c_str());
  bind_tensor_class<1, Val>(m, ("Tensor1" + templatePostfix).c_str());
  bind_tensor_class<2, Val>(m, ("Tensor2" + templatePostfix).c_str());
  bind_tensor_class<3, Val>(m, ("Tensor3" + templatePostfix).c_str());
  bind_tensor_class<4, Val>(m, ("Tensor4" + templatePostfix).c_str());

  m.def(
      "from_numpy",
      [](py::array_t<Val, py::array::c_style | py::array::forcecast>& array) {
        return skimpy::from_buffer(array.size(), array.data());
      });
}
