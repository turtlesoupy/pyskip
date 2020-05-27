#pragma once

#include <pybind11/pybind11.h>
#include <skimpy/skimpy.hpp>
skimpy::Slice convert_slice(skimpy::Pos length, pybind11::slice slice);