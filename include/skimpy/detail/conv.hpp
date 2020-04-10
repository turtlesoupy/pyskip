#pragma once

#include <string>
#include <vector>

#include "core.hpp"

namespace skimpy::detail::conv {

// TODO: Implement these conversion routines.

template <typename Val, typename Allocator>
auto to_store(const std::vector<Val, Allocator>& vals) {
  core::Store<Val> ret;
  return ret;
}

template <typename Val>
auto to_vector(const core::Store<Val>& store) {
  std::vector<Val> ret;
  return ret;
}

template <typename Val>
auto to_string(const core::Store<Val>& store) {
  return "";
}

};  // namespace skimpy::detail::conv
