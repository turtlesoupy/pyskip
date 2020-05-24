#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/skimpy.hpp>

namespace py = pybind11;

auto convert_band(skimpy::Pos length, py::slice slice) {
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

template <size_t dim>
auto convert_tensor_slice(
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
auto get_tensor_shape(const skimpy::TensorShape<dim> &shape) {
  std::array<int, dim> ret;
  for (int i = 0; i < dim; i += 1) {
    ret[i] = shape[i];
  }
  return std::tuple_cat(ret);
}

PYBIND11_MODULE(_skimpy_cpp_ext, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.0.1";

  // Module routines
  m.def("from_numpy", [](py::array_t<int>& array) {
    return skimpy::from_buffer(array.size(), array.data());
  });

  // ArrayBuilder class for int value types
  using IntArrayBuilder = skimpy::ArrayBuilder<int>;
  py::class_<IntArrayBuilder>(m, "IntArrayBuilder")
      .def(py::init<skimpy::Pos, int>())
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
            self.set(convert_band(self.len(), slice), val);
          })
      .def(
          "__setitem__",
          [](IntArrayBuilder& self,
             py::slice slice,
             const skimpy::Array<int>& other) {
            self.set(convert_band(self.len(), slice), other);
          })
      .def("build", &IntArrayBuilder::build);

  // Array class for int value types
  using IntArray = skimpy::Array<int>;
  py::class_<IntArray>(m, "IntArray")
      .def(py::init([](skimpy::Pos span, int fill) {
        return skimpy::make_array<int>(span, fill);
      }))
      .def("__len__", &IntArray::len)
      .def("__repr__", &IntArray::repr)
      .def("clone", &IntArray::clone)
      .def("eval", &IntArray::eval)
      .def("dumps", &IntArray::str)
      .def(
          "tensor",
          [](IntArray& self) { skimpy::make_tensor<1>({self.len()}, self); })
      .def(
          "to_numpy",
          [](IntArray& self) {
            int size, *buffer;
            skimpy::to_buffer(self, &size, &buffer);
            return py::array_t<int>(size, buffer);
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
      .def("__abs__", [](const IntArray& self) { return skimpy::abs(self); })
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
            return skimpy::min(self, other);
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
            return skimpy::min(self, other);
          })
      .def(
          "__pow__",
          [](const IntArray& self, int val) { return skimpy::pow(self, val); })
      .def(
          "__pow__",
          [](const IntArray& self, int val) { return skimpy::pow(val, self); })
      .def(
          "__pow__",
          [](const IntArray& self, const IntArray& other) {
            return skimpy::pow(self, other);
          })
      .def("abs", [](const IntArray& self) { return skimpy::abs(self); })
      .def("sqrt", [](const IntArray& self) { return skimpy::sqrt(self); })
      .def("exp", [](const IntArray& self) { return skimpy::exp(self); });

  // 1D Tensor class for int value types
  {
    using Tensor = skimpy::Tensor<1, int>;
    py::class_<Tensor>(m, "Tensor1i")
        .def(py::init([](int w, int val) {
          return skimpy::make_tensor<1, int>({w}, val);
        }))
        .def(py::init([](int w, const skimpy::Array<int>& vals) {
          return skimpy::make_tensor<1, int>({w}, vals);
        }))
        .def("__len__", &Tensor::len)
        .def("__repr__", &Tensor::repr)
        .def("shape", [](Tensor& self) { return get_tensor_shape(self.shape()); })
        .def("clone", &Tensor::clone)
        .def("eval", &Tensor::eval)
        .def("dumps", &Tensor::str)
        .def("array", &Tensor::array)
        .def(
            "__getitem__",
            [](Tensor& self, std::array<skimpy::Pos, 1> pos) {
              return self.get(pos);
            })
        .def(
            "__getitem__",
            [](Tensor& self, std::array<py::slice, 1> slices) {
              return self.get(convert_tensor_slice(self.shape(), slices));
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<skimpy::Pos, 1> pos, int val) {
              self.set(pos, val);
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<py::slice, 1> slices, int val) {
              self.set(convert_tensor_slice(self.shape(), slices), val);
            })
        .def(
            "__setitem__",
            [](Tensor& self,
               std::array<py::slice, 1> slices,
               const Tensor& other) {
              self.set(convert_tensor_slice(self.shape(), slices), other);
            });
  }

  // 2D Tensor class for int value types
  {
    using Tensor = skimpy::Tensor<2, int>;
    py::class_<Tensor>(m, "Tensor2i")
        .def(py::init([](std::array<skimpy::Pos, 2> shape, int val) {
          auto [w, h] = shape;
          return skimpy::make_tensor<2, int>({w, h}, val);
        }))
        .def(py::init([](std::array<skimpy::Pos, 2> shape,
                         const skimpy::Array<int>& vals) {
          auto [w, h] = shape;
          return skimpy::make_tensor<2, int>({w, h}, vals);
        }))
        .def("__len__", &Tensor::len)
        .def("__repr__", &Tensor::repr)
        .def("shape", [](Tensor& self) { return get_tensor_shape(self.shape()); })
        .def("clone", &Tensor::clone)
        .def("eval", &Tensor::eval)
        .def("dumps", &Tensor::str)
        .def("array", &Tensor::array)
        .def(
            "__getitem__",
            [](Tensor& self, std::array<skimpy::Pos, 2> pos) {
              return self.get(pos);
            })
        .def(
            "__getitem__",
            [](Tensor& self, std::array<py::slice, 2> slices) {
              return self.get(convert_tensor_slice(self.shape(), slices));
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<skimpy::Pos, 2> pos, int val) {
              self.set(pos, val);
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<py::slice, 2> slices, int val) {
              self.set(convert_tensor_slice(self.shape(), slices), val);
            })
        .def(
            "__setitem__",
            [](Tensor& self,
               std::array<py::slice, 2> slices,
               const Tensor& other) {
              self.set(convert_tensor_slice(self.shape(), slices), other);
            });
  }

  // 3D Tensor class for int value types
  {
    using Tensor = skimpy::Tensor<3, int>;
    py::class_<Tensor>(m, "Tensor3i")
        .def(py::init([](std::array<skimpy::Pos, 3> shape, int val) {
          auto [w, h, d] = shape;
          return skimpy::make_tensor<3, int>({w, h, d}, val);
        }))
        .def(py::init([](std::array<skimpy::Pos, 3> shape,
                         const skimpy::Array<int>& vals) {
          auto [w, h, d] = shape;
          return skimpy::make_tensor<3, int>({w, h, d}, vals);
        }))
        .def("__len__", &Tensor::len)
        .def("__repr__", &Tensor::repr)
        .def("shape", [](Tensor& self) { return get_tensor_shape(self.shape()); })
        .def("clone", &Tensor::clone)
        .def("eval", &Tensor::eval)
        .def("dumps", &Tensor::str)
        .def("array", &Tensor::array)
        .def(
            "__getitem__",
            [](Tensor& self, std::array<skimpy::Pos, 3> pos) {
              return self.get(pos);
            })
        .def(
            "__getitem__",
            [](Tensor& self, std::array<py::slice, 3> slices) {
              return self.get(convert_tensor_slice(self.shape(), slices));
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<skimpy::Pos, 3> pos, int val) {
              self.set(pos, val);
            })
        .def(
            "__setitem__",
            [](Tensor& self, std::array<py::slice, 3> slices, int val) {
              self.set(convert_tensor_slice(self.shape(), slices), val);
            })
        .def(
            "__setitem__",
            [](Tensor& self,
               std::array<py::slice, 3> slices,
               const Tensor& other) {
              self.set(convert_tensor_slice(self.shape(), slices), other);
            });
  }
}
