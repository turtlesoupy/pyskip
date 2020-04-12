#pragma once

#include <cstdint>
#include <memory>

#include "errors.hpp"

namespace skimpy::detail::core {

using Pos = int32_t;

template <typename Val>
struct Store {
  int size;
  std::unique_ptr<Pos[]> ends;
  std::unique_ptr<Val[]> vals;

  Store(int n) : size(n), ends(new Pos[n]), vals(new Val[n]) {
    CHECK_ARGUMENT(n > 0);
  }

  Store(int n, std::unique_ptr<Pos[]> ends, std::unique_ptr<Val[]> vals)
      : size(n), ends(std::move(ends)), vals(std::move(vals)) {
    CHECK_ARGUMENT(n > 0);
  }

  Pos span() const {
    CHECK_STATE(size > 0);
    return ends[size - 1];
  }

  int index(Pos pos) const {
    CHECK_STATE(size > 0);
    return std::upper_bound(&ends[0], &ends[size], pos) - &ends[0];
  }
};

}  // namespace skimpy::detail::core
