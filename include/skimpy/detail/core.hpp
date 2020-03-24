#pragma once

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "defs.hpp"
#include "errors.hpp"
#include "utils.hpp"

namespace skimpy {

template <typename Index, typename Value>
using Range = std::tuple<Index, Index, Value>;

template <typename Index, typename Value, typename Cmp = std::less<Index>>
struct RangeStore {
  Index size;
  std::vector<Index> starts;
  std::vector<Value> values;

  RangeStore(Index size, std::vector<Index> starts, std::vector<Value> values)
      : size(std::move(size)),
        starts(std::move(starts)),
        values(std::move(values)) {}

  // Returns the position in the starts and values vectors corresponding to the
  // range containing the given index.
  size_t pos(const Index& index) const {
    CHECK_ARGUMENT(index < size);
    auto iter = std::upper_bound(starts.begin(), starts.end(), index, Cmp());
    CHECK_STATE(iter != starts.begin());
    return std::distance(starts.begin(), iter) - 1;
  }

  // Returns a generator over all ranges in the store from the index onward.
  auto scan(const Index& index) const {
    return make_generator<Range<Index, Value>>(
        [&, i = pos(index)]() mutable -> std::optional<Range<Index, Value>> {
          if (i++ < starts.size()) {
            auto end = i == starts.size() ? size : starts[i];
            return std::tuple(starts[i - 1], end, values[i - 1]);
          } else {
            return {};
          }
        });
  }

  // Replaces contiguous ranges with a common value to a single range.
  void compress() {
    int j = 1;
    for (int i = 1; i < starts.size(); i += 1) {
      if (values[i] != values[j - 1]) {
        starts[j] = starts[i];
        values[j] = values[i];
        j += 1;
      }
    }
    starts.resize(j);
    values.resize(j);
  }
};

template <typename Index, typename Value, typename Cmp = std::less<Index>>
struct BufferedRangeStore {
  RangeStore<Index, Value, Cmp> store;
  std::vector<Range<Index, Value>> buffer;

  BufferedRangeStore(
      Index size, std::vector<Index> starts, std::vector<Value> values)
      : store(std::move(size), std::move(starts), std::move(values)) {}

  // Returns a generator over all ranges in the from the index onward.
  auto scan(const Index& index) const {
    CHECK_ARGUMENT(index < store.size);

    // Initialize the store generator and the scan index.
    auto s_gen = store.scan(index);
    auto scan_index = std::get<0>(s_gen.get());

    // Search the buffer and update the scan_index if necessary.
    auto cmp = [](const Index& i, const Range<Index, Value>& r) {
      return i < std::get<1>(r);
    };
    auto b_iter = std::upper_bound(buffer.begin(), buffer.end(), index, cmp);
    if (b_iter != buffer.end() && std::get<0>(*b_iter) <= index) {
      scan_index = std::get<0>(*b_iter);
    } else if (b_iter != buffer.begin()) {
      scan_index = std::max<Index>(scan_index, std::get<1>(*std::prev(b_iter)));
    }

    // Initialize the buffer generator.
    auto b_gen = make_generator<Range<Index, Value>>(
        [&, b_iter]() mutable -> std::optional<Range<Index, Value>> {
          if (b_iter != buffer.end()) {
            return *b_iter++;
          } else {
            return {};
          }
        });

    // Return a generator over the ranges.
    using RetType = std::optional<Range<Index, Value>>;
    return make_generator<Range<Index, Value>>(
        [s_gen, b_gen, scan_index]() mutable -> RetType {
          while (!b_gen.done() || !s_gen.done()) {
            if (b_gen.done()) {
              auto s = scan_index;
              auto [_, e, v] = s_gen.next();
              if (s < e) {
                scan_index = e;
                return std::tuple(s, e, v);
              }
            } else if (s_gen.done()) {
              auto [s, e, v] = b_gen.next();
              scan_index = e;
              return std::tuple(s, e, v);
            } else {
              auto [b_s, b_e, b_v] = b_gen.get();
              if (scan_index == b_s) {
                b_gen.next();
                scan_index = b_e;
                return std::tuple(b_s, b_e, b_v);
              } else {
                auto s = scan_index;
                auto [_, s_e, s_v] = s_gen.get();
                if (s < s_e) {
                  scan_index = std::min<Index>(b_s, s_e);
                  return std::tuple(s, scan_index, s_v);
                } else {
                  s_gen.next();
                }
              }
            }
          }
          return {};
        });
  }

  // Merges the buffer into the store.
  void flush() {
    // Allocate space for the new ranges.
    std::vector<Index> new_starts;
    std::vector<Value> new_values;
    new_starts.reserve(2 * buffer.size() + store.starts.size());
    new_values.reserve(2 * buffer.size() + store.starts.size());
    CHECK_STATE(new_starts.size() == new_values.size());

    // Populate the new ranges.
    for (auto gen = scan(0); !gen.done(); gen.next()) {
      auto [s, e, v] = gen.get();
      new_starts.emplace_back(std::move(s));
      new_values.emplace_back(std::move(v));
    }

    // Assing the new ranges and clear the buffer.
    store.starts.swap(new_starts);
    store.values.swap(new_values);
    buffer.clear();
  }
};

template <typename T>
class RangeMap {
 public:
  using Store = BufferedRangeStore<index_t, T>;

  RangeMap(std::shared_ptr<Store> store, index_t start, index_t end)
      : store_(std::move(store)), start_(start), end_(end) {
    CHECK_ARGUMENT(start_ >= 0);
    CHECK_ARGUMENT(start_ <= end_);
  }

  auto scan(index_t index) const {
    CHECK_ARGUMENT(0 <= index && index < end_);
    return store_->scan(index);
  }

  T get(index_t index) const {
    return std::get<2>(scan(index).get());
  }

  void set(index_t index, T val) {
    slice(index, index + 1).assign(std::move(val));
  }

  void assign(T t) {
    assign(make(end - start, std::move(t)));
  }

  void assign(const RangeMap<T>& range) {
    // 1. If buffer has room, insert range into buffer.
    // 2. If buffer needs more room, flush everything into store.

    // How to do this?
    // 1. Allocate new key and val vectors with packed size.
    // 2. Merge store with range into new arrays.
    // 3. Swap vectors into store.
    //
    // Potential optimizations:
    // 1. Use a tiered vector and only re-allocate if there's no room.
    // 2. Leave gaps between elements and
    // 3. Use buffered store and flush once size is too large.
  }

  RangeMap<T> slice(index_t start, index_t end) const {
    return RangeMap<T>(store_, start_ + start, start_ + end);
  }

  std::shared_ptr<Store> store() const {
    return store_;
  }

  template <typename T>
  static RangeMap<T> make(index_t size, T fill) {
    auto store = std::make_shared<RangeMap<T>::Store>(
        size, std::vector<index_t>{0}, std::vector<T>{std::move(fill)});
    return RangeMap<T>(std::move(store), 0, size);
  }

 private:
  std::shared_ptr<Store> store_;
  index_t start_;
  index_t end_;
};

}  // namespace skimpy
