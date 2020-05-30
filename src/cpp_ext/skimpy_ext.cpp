#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <optional>
#include <skimpy/skimpy.hpp>

namespace py = pybind11;

void int_binds(py::module &m);
void float_binds(py::module &m);
void char_binds(py::module &m);
void bool_binds(py::module &m);

PYBIND11_MODULE(_skimpy_cpp_ext, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.1.5";

  int_binds(m);
  float_binds(m);
  char_binds(m);
  bool_binds(m);

  auto configModule =
      m.def_submodule("config", "Methods related to skimpy configuration");

  configModule.def("set_value", [](std::string key, std::optional<long> val) {
    if (val) {
      skimpy::GlobalConfig::get().setConfigVal<long>(key, *val);
    } else {
      skimpy::GlobalConfig::get().clearConfigVal(key);
    }
  });
  configModule.def("set_value", [](std::string key, std::optional<bool> val) {
    if (val) {
      skimpy::GlobalConfig::get().setConfigVal<bool>(key, *val);
    } else {
      skimpy::GlobalConfig::get().clearConfigVal(key);
    }
  });
  configModule.def("set_value", [](std::string key, std::optional<double> val) {
    if (val) {
      skimpy::GlobalConfig::get().setConfigVal<double>(key, *val);
    } else {
      skimpy::GlobalConfig::get().clearConfigVal(key);
    }
  });
  configModule.def("get_all_values", []() {
    return skimpy::GlobalConfig::get().getConfigMap();
  });

  configModule.def(
      "set_all_values",
      [](std::unordered_map<std::string, skimpy::ConfigTypes> &map) {
        skimpy::GlobalConfig::get().setConfigMap(map);
      });
}
