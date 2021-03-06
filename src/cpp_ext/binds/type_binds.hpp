#if defined(__GNUC__) && defined(_WIN32)
#define _hypot hypot
#endif

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pyskip/pyskip.hpp>

namespace py = pybind11;

using pyskip::Pos;

inline auto convert_band(Pos length, py::slice slice) {
  CHECK_ARGUMENT(slice.attr("step").is_none());
  Pos start = 0;
  Pos stop = length;

  // Set non-default slice values.
  if (!slice.attr("start").is_none()) {
    start = slice.attr("start").cast<Pos>();
  }
  if (!slice.attr("stop").is_none()) {
    stop = slice.attr("stop").cast<Pos>();
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
  return pyskip::Band(start, stop);
}

inline auto convert_slice(Pos length, py::slice slice) {
  Pos start = 0;
  Pos stop = length;
  Pos stride = 1;

  // Set non-default slice values.
  if (!slice.attr("start").is_none()) {
    start = slice.attr("start").cast<Pos>();
  }
  if (!slice.attr("stop").is_none()) {
    stop = slice.attr("stop").cast<Pos>();
  }
  if (!slice.attr("step").is_none()) {
    stride = slice.attr("step").cast<Pos>();
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
  return pyskip::Slice(start, stop, stride);
}

template <size_t dim>
inline auto convert_tensor_slice(
    pyskip::TensorShape<dim> shape, std::array<py::slice, dim> slices) {
  std::array<std::array<Pos, 3>, dim> components;
  for (int i = 0; i < dim; i += 1) {
    auto slice = convert_slice(shape[i], slices[i]);
    components[i][0] = slice.start;
    components[i][1] = slice.stop;
    components[i][2] = slice.stride;
  }
  return pyskip::TensorSlice<dim>(std::move(components));
}

template <size_t dim>
inline auto get_tensor_shape(const pyskip::TensorShape<dim>& shape) {
  std::array<int, dim> ret;
  for (int i = 0; i < dim; i += 1) {
    ret[i] = shape[i];
  }
  return std::tuple_cat(ret);
}

template <typename Val>
inline void bind_builder_class(py::module& m, const char* class_name) {
  using Builder = pyskip::ArrayBuilder<Val>;
  py::class_<Builder>(m, class_name)
      .def(py::init<Pos, Val>())
      .def(py::init<pyskip::Array<Val>>())
      .def("__len__", &Builder::len)
      .def("__repr__", &Builder::repr)
      .def(
          "__setitem__",
          [](Builder& self, Pos pos, Val val) { self.set(pos, val); })
      .def(
          "__setitem__",
          [](Builder& self, py::slice slice, Val val) {
            self.set(convert_band(self.len(), slice), val);
          })
      .def(
          "__setitem__",
          [](Builder& self, py::slice slice, const pyskip::Array<Val>& other) {
            self.set(convert_band(self.len(), slice), other);
          })
      .def("dumps", &Builder::str)
      .def("build", &Builder::build);
}

template <typename Val>
inline void bind_array_class(py::module& m, const char* class_name) {
  using Array = pyskip::Array<Val>;
  auto& cls =
      py::class_<Array>(m, class_name)
          .def(py::init([](Pos span, Val fill) {
            return pyskip::make_array<Val>(span, fill);
          }))
          .def("__len__", &Array::len)
          .def("__repr__", &Array::repr)
          .def("clone", &Array::clone)
          .def("eval", &Array::eval)
          .def("dumps", &Array::str)
          .def(
              "tensor",
              [](Array& self) { pyskip::make_tensor<1>({self.len()}, self); })
          .def(
              "to_numpy",
              [](Array& self) {
                int size;
                Val* buffer;
                pyskip::to_buffer(self, &size, &buffer);
                return py::array_t<Val>(size, buffer);
              })
          .def(
              "runs",
              [](Array& self) {
                auto store = self.store();
                return std::make_tuple(
                    py::array_t<Pos>(store->size, store->ends.release()),
                    py::array_t<Val>(store->size, store->vals.release()));
              })
          .def(
              "__getitem__", [](Array& self, Pos pos) { return self.get(pos); })
          .def(
              "__getitem__",
              [](Array& self, py::slice slice) {
                return self.get(convert_slice(self.len(), slice));
              })
          .def(
              "__setitem__",
              [](Array& self, Pos pos, Val val) { self.set(pos, val); })
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
        .def("__abs__", [](const Array& self) { return pyskip::abs(self); })
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
              return pyskip::cast<float>(self) / static_cast<float>(val);
            })
        .def(
            "__rtruediv__",
            [](const Array& self, Val val) {
              return static_cast<float>(val) / pyskip::cast<float>(self);
            })
        .def(
            "__truediv__",
            [](const Array& self, const Array& other) {
              return pyskip::cast<float>(self) / pyskip::cast<float>(other);
            })
        .def("__mod__", [](const Array& self, Val val) { return self % val; })
        .def("__rmod__", [](const Array& self, Val val) { return val % self; })
        .def(
            "__mod__",
            [](const Array& self, const Array& other) { return self % other; })
        .def(
            "__pow__",
            [](const Array& self, Val val) { return pyskip::pow(self, val); })
        .def(
            "__pow__",
            [](const Array& self, Val val) { return pyskip::pow(val, self); })
        .def(
            "__pow__",
            [](const Array& self, const Array& other) {
              return pyskip::pow(self, other);
            })
        .def("abs", [](const Array& self) { return pyskip::abs(self); })
        .def("sqrt", [](const Array& self) { return pyskip::sqrt(self); })
        .def("exp", [](const Array& self) { return pyskip::exp(self); });
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
        .def("__rxor__", [](const Array& self, Val val) { return val ^ self; })
        .def(
            "__xor__",
            [](const Array& self, const Array& other) { return self ^ other; })
        .def(
            "__lshift__",
            [](const Array& self, Val val) { return self << val; })
        .def(
            "__rlshift__",
            [](const Array& self, Val val) { return val << self; })
        .def(
            "__lshift__",
            [](const Array& self, const Array& other) { return self << other; })
        .def(
            "__rshift__",
            [](const Array& self, Val val) { return self >> val; })
        .def(
            "__rrshift__",
            [](const Array& self, Val val) { return val >> self; })
        .def("__rshift__", [](const Array& self, const Array& other) {
          return self >> other;
        });
  } else if constexpr (std::is_same_v<Val, bool>) {
    cls.def("__and__", [](const Array& self, Val val) { return self && val; })
        .def("__rand__", [](const Array& self, Val val) { return val && self; })
        .def(
            "__and__",
            [](const Array& self, const Array& other) { return self && other; })
        .def("__or__", [](const Array& self, Val val) { return self || val; })
        .def("__ror__", [](const Array& self, Val val) { return val || self; })
        .def("__or__", [](const Array& self, const Array& other) {
          return self || other;
        });
  }

  // Add logical operations.
  if constexpr (std::is_same_v<Val, bool>) {
    cls.def("__and__", [](const Array& self, Val val) { return self && val; })
        .def("__rand__", [](const Array& self, Val val) { return val && self; })
        .def(
            "__and__",
            [](const Array& self, const Array& other) { return self && other; })
        .def("__or__", [](const Array& self, Val val) { return self || val; })
        .def("__ror__", [](const Array& self, Val val) { return val || self; })
        .def("__or__", [](const Array& self, const Array& other) {
          return self || other;
        });
  }

  // Add comparison operations.
  cls.def("__eq__", [](const Array& self, Val val) { return self == val; })
      .def(
          "__eq__",
          [](const Array& self, const Array& other) { return self == other; })
      .def("__ne__", [](const Array& self, Val val) { return self != val; })
      .def(
          "__ne__",
          [](const Array& self, const Array& other) { return self != other; })
      .def("__le__", [](const Array& self, Val val) { return self <= val; })
      .def(
          "__le__",
          [](const Array& self, const Array& other) { return self <= other; })
      .def("__lt__", [](const Array& self, Val val) { return self < val; })
      .def(
          "__lt__",
          [](const Array& self, const Array& other) { return self < other; })
      .def("__ge__", [](const Array& self, Val val) { return self >= val; })
      .def(
          "__ge__",
          [](const Array& self, const Array& other) { return self >= other; })
      .def("__gt__", [](const Array& self, Val val) { return self > val; })
      .def(
          "__gt__",
          [](const Array& self, const Array& other) { return self > other; })
      .def(
          "__gt__",
          [](const Array& self, const Array& other) { return self > other; })
      .def(
          "coalesce",
          [](const Array& self, Val val) {
            return pyskip::coalesce(self, val);
          })
      .def("coalesce", [](const Array& self, const Array& other) {
        return pyskip::coalesce(self, other);
      });

  cls.def(
         "min",
         [](const Array& self, Val val) { return pyskip::min(self, val); })
      .def(
          "min",
          [](const Array& self, Val val) { return pyskip::min(val, self); })
      .def(
          "min",
          [](const Array& self, const Array& other) {
            return pyskip::min(self, other);
          })
      .def(
          "max",
          [](const Array& self, Val val) { return pyskip::max(self, val); })
      .def(
          "max",
          [](const Array& self, Val val) { return pyskip::max(val, self); })
      .def("max", [](const Array& self, const Array& other) {
        return pyskip::max(self, other);
      });

  // Add conversion operations.
  cls.def("int", [](const Array& self) { return pyskip::cast<int>(self); })
      .def("char", [](const Array& self) { return pyskip::cast<char>(self); })
      .def("float", [](const Array& self) { return pyskip::cast<float>(self); })
      .def("bool", [](const Array& self) { return pyskip::cast<bool>(self); });
}

template <size_t dim, typename Val>
inline void bind_tensor_class(py::module& m, const char* class_name) {
  using Tensor = pyskip::Tensor<dim, Val>;
  py::class_<Tensor>(m, class_name)
      .def(py::init([](const std::array<Pos, dim>& s, Val v) {
        return pyskip::make_tensor<dim, Val>(pyskip::make_shape<dim>(s), v);
      }))
      .def(py::init(
          [](const std::array<Pos, dim> s, const pyskip::Array<Val>& v) {
            return pyskip::make_tensor<dim, Val>(pyskip::make_shape<dim>(s), v);
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
          [](Tensor& self, std::array<Pos, dim> pos) { return self.get(pos); })
      .def(
          "__getitem__",
          [](Tensor& self, std::array<py::slice, dim> slices) {
            return self.get(convert_tensor_slice(self.shape(), slices));
          })
      .def(
          "__setitem__",
          [](Tensor& self, std::array<Pos, dim> pos, Val val) {
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
        return pyskip::from_buffer(array.size(), array.data());
      });
}
