#include "helpers.hpp"

namespace py = pybind11;

skimpy::Slice convert_slice(skimpy::Pos length, py::slice slice) {
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