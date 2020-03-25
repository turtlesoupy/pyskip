#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "errors.hpp"
#include "utils.hpp"

namespace skimpy {

// Utility class for storing sorted ranges associated with values.
template <typename Pos, typename Val>
struct RangeStore {
  std::vector<std::pair<Pos, Val>> ranges;

  explicit RangeStore(std::vector<std::pair<Pos, Val>> ranges)
      : ranges(std::move(ranges)) {}

  auto find(const Pos& pos) {
    CHECK_STATE(ranges.size() && ranges[0].first <= pos);
    auto iter = std::upper_bound(
        ranges.begin(),
        ranges.end(),
        pos,
        [](const Pos& pos, const std::pair<Pos, Val>& range) {
          return std::less<Pos>()(pos, range.first);
        });
    return --iter;
  }

  auto index(const Pos& pos) {
    return std::distance(ranges.begin(), find(pos));
  }
};

// Utility class for operating over range stores.
template <typename Pos, typename Val>
struct RangeMap {
  using Store = RangeStore<Pos, Val>;

  RangeMap(std::shared_ptr<Store> store, Pos start, Pos stop)
      : store_(std::move(store)),
        start_(std::move(start)),
        stop_(std::move(stop)) {
    CHECK_ARGUMENT(0 <= start_ && start_ <= stop_);
  }

  Pos size() const {
    return stop_ - start_;
  }

  auto scan(const Pos& pos) const {
    using Ret = std::tuple<Pos, Pos, Val>;
    auto iter = store_->find(store_pos(pos));
    return make_generator<Ret>(
        [this, iter = iter]() mutable -> std::optional<Ret> {
          if (iter == store_->ranges.end() || iter->first >= stop_) {
            return {};
          }
          auto curr = iter++;
          auto next = iter;
          auto start = std::max<Pos>(start_, curr->first);
          if (next != store_->ranges.end()) {
            auto stop = std::min<Pos>(next->first, stop_);
            return std::tuple(start - start_, stop - start_, curr->second);
          } else {
            return std::tuple(start - start_, stop_ - start_, curr->second);
          }
        });
  }

  Val get(const Pos& pos) const {
    return store_->find(store_pos(pos))->second;
  }

  void set(const Pos& pos, Val val) {
    slice(pos, pos + 1).assign(std::move(val));
  }

  void assign(Val val) {
    assign(make_range_map(stop_ - start_, std::move(val)));
  }

  void assign(const RangeMap<Pos, Val>& other) {
    CHECK_ARGUMENT(other.size() == size());
    if (size() == 0) {
      return;
    }

    // We will be copying values out from the old ranges vector.
    auto& ranges = store_->ranges;

    // Allocate space for the new ranges vector.
    std::vector<std::pair<Pos, Val>> buffer;
    buffer.reserve(2 + ranges.size() - range_count() + other.range_count());

    // Copy over everything from the store before the start position.
    auto iter = store_->find(start_);
    std::copy(ranges.begin(), iter, std::back_inserter(buffer));

    // Also copy over the start interval if its precedes the start position.
    if (iter->first < start_) {
      buffer.emplace_back(*iter);
    }

    // Set a new interval at the start of the other range (if required).
    auto other_iter = other.store_->find(other.start_);
    if (buffer.empty() || buffer.back().second != other_iter->second) {
      buffer.emplace_back(start_, other_iter->second);
    }

    // Copy over all of the others ranges strictly inside this range map.
    // NOTE: We need to adjust the starting position of these intervals.
    auto other_end = ++other.store_->find(other.stop_ - 1);
    for (++other_iter; other_iter != other_end; ++other_iter) {
      auto shift = start_ - other.start_;
      buffer.emplace_back(other_iter->first + shift, other_iter->second);
    }

    // Conditionally insert a new range at the start of this maps end.
    iter = store_->find(stop_);
    if (iter != ranges.end() && iter->second != buffer.back().second) {
      buffer.emplace_back(stop_, iter->second);
    }

    // Copy over the last sequence of ranges.
    if (iter != ranges.end()) {
      std::move(++iter, ranges.end(), std::back_inserter(buffer));
    }

    // Swap in the new set of ranges into our store.
    store_->ranges.swap(buffer);
  }

  RangeMap<Pos, Val> slice(const Pos& start, const Pos& stop) const {
    auto offset = store_pos(start);
    return RangeMap<Pos, Val>(store_, offset, offset - start + stop);
  }

  RangeMap<Pos, Val> clone() const {
    auto ret = make_range_map(stop_ - start_, 0);
    ret.assign(*this);
    return ret;
  }

 private:
  Pos store_pos(const Pos& pos) const {
    CHECK_ARGUMENT(0 <= pos && pos < stop_ - start_);
    return start_ + pos;
  }

  size_t range_count() const {
    return store_->index(stop_) - store_->index(start_);
  }

  std::shared_ptr<Store> store_;
  Pos start_;
  Pos stop_;
};

template <typename Pos, typename Val>
inline RangeMap<Pos, Val> make_range_map(Pos size, Val fill) {
  auto store = std::make_shared<RangeStore<Pos, Val>>(
      std::vector<std::pair<Pos, Val>>{std::pair(0, std::move(fill))});
  return RangeMap<Pos, Val>(std::move(store), 0, std::move(size));
}

}  // namespace skimpy
