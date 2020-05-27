#include "type_binds.hpp"

void bool_binds(py::module &m) {
    type_binds<bool>(m, "Bool", "b");
}