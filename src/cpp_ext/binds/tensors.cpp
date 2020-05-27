#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/tensor.hpp>
#include "helpers.hpp"

namespace py = pybind11;

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
auto get_tensor_shape(const skimpy::TensorShape<dim>& shape) {
  std::array<int, dim> ret;
  for (int i = 0; i < dim; i += 1) {
    ret[i] = shape[i];
  }
  return std::tuple_cat(ret);
}


template <size_t dim, typename Val>
void bind_tensor_class(py::module& m, const char* class_name) {
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


void bind_tensors(py::module &m) {
  bind_tensor_class<1, int>(m, "Tensor1i");
  bind_tensor_class<2, int>(m, "Tensor2i");
  bind_tensor_class<3, int>(m, "Tensor3i");
  bind_tensor_class<4, int>(m, "Tensor4i");
  bind_tensor_class<1, float>(m, "Tensor1f");
  bind_tensor_class<2, float>(m, "Tensor2f");
  bind_tensor_class<3, float>(m, "Tensor3f");
  bind_tensor_class<4, float>(m, "Tensor4f");
  bind_tensor_class<1, char>(m, "Tensor1c");
  bind_tensor_class<2, char>(m, "Tensor2c");
  bind_tensor_class<3, char>(m, "Tensor3c");
  bind_tensor_class<4, char>(m, "Tensor4c");
  bind_tensor_class<1, bool>(m, "Tensor1b");
  bind_tensor_class<2, bool>(m, "Tensor2b");
  bind_tensor_class<3, bool>(m, "Tensor3b");
  bind_tensor_class<4, bool>(m, "Tensor4b");

}