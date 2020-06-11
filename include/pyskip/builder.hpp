#pragma once

#include <memory>
#include <vector>

#include "array.hpp"
#include "detail/box.hpp"
#include "detail/core.hpp"
#include "detail/util.hpp"
#include "macros.hpp"

namespace pyskip {

namespace box = detail::box;
namespace core = detail::core;
namespace util = detail::util;

using Pos = core::Pos;

struct Band {
  core::Pos start;
  core::Pos stop;

  Band(core::Pos start, core::Pos stop) : start(start), stop(stop) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
  }

  explicit Band(core::Pos stop) : Band(0, stop) {}

  auto len() const {
    return stop - start;
  }
};

template <typename Val>
class Array;

template <typename Val>
class ArrayBuilder {
  static constexpr auto kBlockSize = 4096;

 public:
  // Value constructors
  ArrayBuilder(core::Pos len, Val val) : len_(len) {
    CHECK_ARGUMENT(len_ > 0);
    auto k = 1 + (len_ - 1) / kBlockSize;
    stores_.reserve(k);
    for (int i = 0; i < k; i += 1) {
      auto span = std::min(kBlockSize, len_ - i * kBlockSize);
      stores_.push_back(core::make_store(span, box::Box(val)));
    }
  }
  explicit ArrayBuilder(const Array<Val>& array)
      : ArrayBuilder(array.len(), 0) {
    set(array);
  }

  // Metadata methods
  auto len() const {
    return len_;
  }
  auto str() const {
    return build().str();
  }
  auto repr() const {
    CHECK_STATE(len() > 0);
    if (len() <= 10) {
      return fmt::format(
          "Builder<{}>([{}])",
          typeid(Val).name(),
          fmt::join(conv::to_vector<box::Box, Val>(stores_.at(0)), ", "));
    } else {
      auto range = core::make_range(stores_.at(0), 4);
      return fmt::format(
          "Builder<{}>([{}, ...])",
          typeid(Val).name(),
          fmt::join(conv::to_vector<box::Box, Val>(range), ", "));
    }
  }

  // Value assign methods
  ArrayBuilder<Val>& set(Val val) {
    set(Band(len_), make_store(len_, std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(core::Pos pos, Val val) {
    CHECK_ARGUMENT(pos < len_);
    auto i = pos / kBlockSize;
    auto& dst = stores_.at(i);
    core::set(dst, pos - i * kBlockSize, box::Box(val));
    reserve(dst);
    return *this;
  }
  ArrayBuilder<Val>& set(const Band& band, Val val) {
    set(band, Array<Val>::make(band.len(), std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Array<Val>& other) {
    set(Band(len_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Band& band, const Array<Val>& other) {
    CHECK_ARGUMENT(band.stop <= len());
    CHECK_ARGUMENT(band.len() == other.len());
    if (band.len() == 0) {
      return *this;
    }

    // Materialize the array.
    auto store = other.store();

    // Assign the array into each intersecting block.
    auto o = band.start;
    auto b = kBlockSize;
    for (auto s = o; s < band.stop; s += b - (s % b)) {
      auto l = std::min(band.stop - s, b - (s % b));
      auto& dst = stores_.at(s / b);
      core::insert(dst, core::Range(*store, s - o, s + l - o), s % b);
      reserve(dst);
    }
    return *this;
  }

  // Builder methods
  Array<Val> build() const {
    auto store = std::make_shared<box::BoxStore>(1, 1 + capacity());
    store->ends[0] = len_;
    for (int i = 0; i < stores_.size(); i += 1) {
      core::insert(*store, stores_[i], kBlockSize * i);
    }
    return Array<Val>(std::move(store));
  }

 private:
  auto capacity() const {
    int ret = 0;
    for (const auto& store : stores_) {
      ret += store.size;
    }
    return ret;
  }

  void reserve(box::BoxStore& store) {
    if (!util::is_power_of_two(store.capacity)) {
      store.reserve(util::round_up_to_power_of_two(store.capacity));
    }
  }

  core::Pos len_;
  std::vector<box::BoxStore> stores_;
};

// Convenience initializers.
template <typename Val>
auto make_builder(const Array<Val>& array) {
  return ArrayBuilder<Val>(array);
}
template <typename Val>
auto make_builder(core::Pos len, Val val) {
  return ArrayBuilder<Val>(Array<Val>::make(len, val));
}

}  // namespace pyskip
