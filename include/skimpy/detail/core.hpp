#pragma once

#include <cstdint>
#include <memory>

namespace skimpy::detail::core {

using Pos = int32_t;

template <typename Val>
struct Store {
  int size;
  std::unique_ptr<Pos[]> ends;
  std::unique_ptr<Val[]> vals;

  Store(int n) : size(n), ends(new Pos[n]), vals(new Val[n]) {}

  Store(int n, std::unique_ptr<Pos[]> ends, std::unique_ptr<Val[]> vals)
      : size(n), ends(std::move(ends)), vals(std::move(vals)) {}

  int index(Pos pos) const {
    return std::upper_bound(&ends[0], &ends[size], pos) - &ends[0];
  }
};

}  // namespace skimpy::detail::core
