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
void to_buffer(const core::Range<Val>& range, Out* buffer) {
  auto span = range.span();
  auto store_index = range.start_index();
  for (int i = 0; i < span; ++i) {
    while (range.end(store_index) <= i) {
      ++store_index;
    }
    *buffer++ = range.store.vals[store_index];
  }
}

template <typename Val, typename Out = Val>
void to_buffer(const core::Store<Val>& store, Out* buffer) {
  to_buffer<Val, Out>(core::make_range(store), buffer);
}

template <typename Val, typename Out = Val>
auto to_vector(const core::Range<Val>& range) {
  auto span = range.span();

  std::vector<Out> ret;
  ret.reserve(span);
  auto store_index = range.start_index();
  for (auto i = 0; i < span; i += 1) {
    while (range.end(store_index) <= i) {
      store_index += 1;
    }
    ret.push_back(range.store.vals[store_index]);
  }

  return ret;
}

template <typename Val, typename Out = Val>
auto to_vector(const core::Store<Val>& store) {
  return to_vector<Val, Out>(core::make_range(store));
}

template <typename Val>
auto to_string(const core::Range<Val>& range) {
  CHECK_ARGUMENT(range.size() > 0);
  std::string ret = "";
  for (auto i = range.start_index(); i <= range.stop_index(); i += 1) {
    if (i == 0) {
      ret += fmt::format("{}=>{}", range.store.ends[i], range.store.vals[i]);
    } else {
      ret += fmt::format(", {}=>{}", range.store.ends[i], range.store.vals[i]);
    }
  }
  return ret;
}

template <typename Val>
auto to_string(const core::Store<Val>& store) {
  return to_string(core::make_range(store));
}

};  // namespace skimpy::detail::conv
