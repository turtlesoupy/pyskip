#include "type_binds.hpp"

void char_binds(py::module &m) {
    type_binds<char>(m, "Char", "c");
}