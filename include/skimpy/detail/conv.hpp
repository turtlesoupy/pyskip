#pragma once

#include <fmt/core.h>

#include <string>
#include <vector>

#include "core.hpp"
#include "errors.hpp"

namespace skimpy::detail::conv {

using Pos = core::Pos;

template <typename Val, typename Out = Val>
auto to_store(size_t size, const Val* buffer) {
  CHECK_ARGUMENT(size > 0);

  // Work out how big of a store array is needed.
  auto store_size = 1;
  for (int i = 1; i < size; i += 1) {
    if (buffer[i] != buffer[i - 1]) {
      store_size += 1;
    }
  }

  // Create the store arrays.
  std::unique_ptr<Pos[]> ends(new Pos[store_size]);
  std::unique_ptr<Out[]> vals(new Out[store_size]);
  auto ends_ptr = &ends[0];
  auto vals_ptr = &vals[0];
  for (int i = 1; i < size; i += 1) {
    if (buffer[i] != buffer[i - 1]) {
      *ends_ptr++ = i;
      *vals_ptr++ = buffer[i - 1];
    }
  }
  *ends_ptr = size;
  *vals_ptr = buffer[size - 1];

  return std::make_shared<core::Store<Out>>(
      store_size, std::move(ends), std::move(vals));
}

template <typename Val, typename Out = Val>
auto to_store(const std::vector<Val>& array) {
  if constexpr (std::is_same_v<Val, bool>) {
    std::unique_ptr<bool[]> bools(new bool[array.size()]);
    for (int i = 0; i < array.size(); i += 1) {
      bools[i] = array[i];
    }
    return to_store<Val, Out>(array.size(), &bools[0]);
  } else {
    return to_store<Val, Out>(array.size(), array.data());
  }
}

template <typename Val, typename Out = Val>
auto to_store(const std::initializer_list<Val>& il) {
  return to_store<Val, Out>(std::vector<Val>(il));
}

template <typename Val, typename Out = Val>
void to_buffer(const core::Store<Val>& store, Out* buffer) {
  auto span = store.span();
  auto ends_ptr = &store.ends[0];
  auto vals_ptr = &store.vals[0];
  for (int i = 0; i < store.span(); i += 1) {
    while (*ends_ptr <= i) {
      ++ends_ptr;
      ++vals_ptr;
    }
    *buffer++ = *vals_ptr;
  }
}

template <typename Val, typename Out = Val>
auto to_vector(const core::Store<Val>& store) {
  auto span = store.span();
  auto ends_ptr = &store.ends[0];
  auto vals_ptr = &store.vals[0];

  std::vector<Out> ret;
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
  CHECK_ARGUMENT(store.size);
  auto ret = fmt::format("{}=>{}", store.ends[0], store.vals[0]);
  for (int i = 1; i < store.size; i += 1) {
    ret += fmt::format(", {}=>{}", store.ends[i], store.vals[i]);
  }
  return ret;
}

};  // namespace skimpy::detail::conv
