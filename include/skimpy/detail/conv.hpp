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
  auto span = store.span();
  auto ends_ptr = &store.ends[0];
  auto vals_ptr = &store.vals[0];

  std::vector<Val> ret;
  ret.reserve(span);
  for (int i = 0; i < store.span(); i += 1) {
    while (*ends_ptr <= i) {
      ++ends_ptr;
      ++vals_ptr;
    }
    ret.push_back(*vals_ptr);
  }
  return ret;
}

template <typename Val>
auto to_string(const core::Store<Val>& store) {
  return "";
}

};  // namespace skimpy::detail::conv
