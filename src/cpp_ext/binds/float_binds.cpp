#include "type_binds.hpp"

void float_binds(py::module &m) {
    type_binds<float>(m, "Float", "f");
}