#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <skimpy/builder.hpp>

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


template <typename Val>
void bind_builder_class(py::module& m, const char* class_name) {
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

void bind_builders(py::module &m) {
  bind_builder_class<int>(m, "IntBuilder");
  bind_builder_class<float>(m, "FloatBuilder");
  bind_builder_class<char>(m, "CharBuilder");
  bind_builder_class<bool>(m, "BoolBuilder");
}