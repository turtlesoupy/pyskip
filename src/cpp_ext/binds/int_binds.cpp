#include "type_binds.hpp"

void int_binds(py::module &m) {
    type_binds<int>(m, "Int", "i");
}