#if defined(__GNUC__) && defined(_WIN32)
#define _hypot hypot
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <optional>
#include <pyskip/pyskip.hpp>

namespace py = pybind11;

void int_binds(py::module& m);
void float_binds(py::module& m);
void char_binds(py::module& m);
void bool_binds(py::module& m);

using pyskip::detail::config::GlobalConfig;

PYBIND11_MODULE(_pyskip_cpp_ext, m) {
  m.doc() = "Space-optimized arrays";
  m.attr("__version__") = "0.1.5";

  int_binds(m);
  float_binds(m);
  char_binds(m);
  bool_binds(m);

  auto config_module =
      m.def_submodule("config", "Methods related to pyskip configuration");

  config_module.def(
      "set_int_value", [](std::string key, std::optional<long> val) {
        if (val) {
          GlobalConfig::get().set_config_val<int64_t>(key, *val);
        } else {
          GlobalConfig::get().clear_config_val(key);
        }
      });
  config_module.def(
      "set_bool_value", [](std::string key, std::optional<bool> val) {
        if (val) {
          GlobalConfig::get().set_config_val<bool>(key, *val);
        } else {
          GlobalConfig::get().clear_config_val(key);
        }
      });
  config_module.def(
      "set_float_value", [](std::string key, std::optional<double> val) {
        if (val) {
          GlobalConfig::get().set_config_val<double>(key, *val);
        } else {
          GlobalConfig::get().clear_config_val(key);
        }
      });

  config_module.def(
      "set_string_value", [](std::string key, std::optional<std::string> val) {
        if (val) {
          GlobalConfig::get().set_config_val<std::string>(key, *val);
        } else {
          GlobalConfig::get().clear_config_val(key);
        }
      });

  config_module.def(
      "get_all_values", []() { return GlobalConfig::get().get_config_map(); });

  config_module.def(
      "set_all_values",
      [](std::unordered_map<std::string, pyskip::detail::config::ConfigTypes>&
             map) { GlobalConfig::get().set_config_map(map); });
}
