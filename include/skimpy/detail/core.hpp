#pragma once

#include <cstdint>
#include <memory>

#include "errors.hpp"

namespace skimpy::detail::core {

using Pos = int32_t;

template <typename Val>
struct Store {
  int size;
  int capacity;
  std::unique_ptr<Pos[]> ends;
  std::unique_ptr<Val[]> vals;

  Store(int n) : Store(n, n) {}

  Store(int n, int c)
      : Store(
            n,
            c,
            std::unique_ptr<Pos[]>(new Pos[c]),
            std::unique_ptr<Val[]>(new Val[c])) {}

  Store(int n, std::unique_ptr<Pos[]> ends, std::unique_ptr<Val[]> vals)
      : Store(n, n, std::move(ends), std::move(vals)) {}

  Store(int n, int c, std::unique_ptr<Pos[]> ends, std::unique_ptr<Val[]> vals)
      : size(n), capacity(c), ends(std::move(ends)), vals(std::move(vals)) {
    CHECK_ARGUMENT(n > 0);
    CHECK_ARGUMENT(c >= n);
  }

  void reserve(int capacity) {
    if (capacity > this->capacity) {
      std::unique_ptr<Pos[]> new_ends(new Pos[capacity]);
      std::unique_ptr<Val[]> new_vals(new Val[capacity]);
      std::copy(&ends[0], &ends[size], &new_ends[0]);
      std::copy(&vals[0], &vals[size], &new_vals[0]);
      ends = std::move(new_ends);
      vals = std::move(new_vals);
      this->capacity = capacity;
    }
  }

  int index(Pos pos) const {
    CHECK_STATE(size > 0);
    return std::upper_bound(&ends[0], &ends[size], pos) - &ends[0];
  }

  Pos span() const {
    CHECK_STATE(size > 0);
    return ends[size - 1];
  }

  Val get(Pos pos) const {
    return *std::upper_bound(&ends[0], &ends[size], pos);
  }
};

template <typename Val>
struct Range {
  const Store<Val>& store;
  int start;
  int stop;

  Range(const Store<Val>& store, int start, int stop)
      : store(store), start(start), stop(stop) {}

  int index(Pos pos) const {
    return store.index(start + pos);
  }

  int start_index() const {
    return store.index(start);
  }

  int stop_index() const {
    return store.index(stop - 1);
  }

  int size() const {
    return 1 + stop_index() - start_index();
  }

  Pos end(int index) const {
    CHECK_ARGUMENT(index < store.size);
    return std::min(stop, store.ends[index]) - start;
  }

  Pos span() const {
    return stop - start;
  }

  Val get(Pos pos) const {
    return *std::upper_bound(&store.ends[0], &store.ends[size], start + pos);
  }
};

template <typename Val>
auto make_store(Pos span, Val fill) {
  CHECK_ARGUMENT(span > 0);
  Store<Val> ret(1);
  ret.ends[0] = span;
  ret.vals[0] = std::move(fill);
  return ret;
}

template <typename Val>
auto make_shared_store(Pos span, Val fill) {
  return std::make_shared<Store<Val>>(make_store(span, std::move(fill)));
}

template <typename Val>
auto make_range(const Store<Val>& store, Pos start, Pos stop) {
  return Range<Val>(store, start, stop);
}

template <typename Val>
auto make_range(const Store<Val>& store, Pos stop) {
  return Range<Val>(store, 0, stop);
}

template <typename Val>
void set(Store<Val>& dst, Pos pos, Val val) {
  CHECK_ARGUMENT(pos < dst.span());

  const auto out = dst.index(pos);

  // Do nothing if the out range already has the same value.
  if (dst.vals[out] == val) {
    return;
  }

  auto l_shift = [](auto& s, auto i, auto k) {
    for (auto j = i + k; j < s.size; j += 1) {
      s.ends[j - k] = s.ends[j];
      s.vals[j - k] = s.vals[j];
    }
  };

  auto r_shift = [](auto& s, auto i, auto k) {
    s.reserve(s.size + 2);
    for (auto j = s.size - 1; j >= i; j -= 1) {
      s.ends[j + k] = s.ends[j];
      s.vals[j + k] = s.vals[j];
    }
  };

  // Extract the relevant conditions for case analysis.
  auto size = dst.size;
  auto prev_end = out > 0 ? dst.ends[out - 1] : 0;
  auto curr_end = dst.ends[out];
  auto l_adjacent = pos == prev_end;
  auto r_adjacent = pos + 1 == curr_end;
  auto width = curr_end - prev_end;
  auto l_compress = out > 0 && l_adjacent && dst.vals[out - 1] == val;
  auto r_compress = out < size - 1 && r_adjacent && dst.vals[out + 1] == val;

  // The cases are determeind based on how much shifting is required. We can
  // shift in either direction up to two position depending on compression.
  if (l_compress && r_compress) {
    dst.ends[out - 1] = dst.ends[out + 1];
    l_shift(dst, out, 2);
    dst.size -= 2;
  } else if (width == 1 && l_compress) {
    dst.ends[out - 1] = dst.ends[out];
    l_shift(dst, out, 1);
    dst.size -= 1;
  } else if (width == 1 && r_compress) {
    l_shift(dst, out, 1);
    dst.size -= 1;
  } else if (width == 1) {
    dst.vals[out] = std::move(val);
  } else if (l_compress) {
    ++dst.ends[out - 1];
  } else if (r_compress) {
    --dst.ends[out];
  } else if (l_adjacent) {
    r_shift(dst, out, 1);
    dst.ends[out] = pos + 1;
    dst.vals[out] = std::move(val);
    dst.size += 1;
  } else if (r_adjacent) {
    r_shift(dst, out, 1);
    --dst.ends[out];
    dst.vals[out + 1] = std::move(val);
    dst.size += 1;
  } else {
    r_shift(dst, out, 2);
    dst.ends[out] = pos;
    dst.ends[out + 1] = pos + 1;
    dst.vals[out + 1] = std::move(val);
    dst.size += 2;
  }
}

template <typename Val>
void insert(Store<Val>& dst, const Range<Val>& src, Pos pos) {
  CHECK_ARGUMENT(pos + src.span() <= dst.span());

  auto out_iter = dst.index(pos);
  auto dst_iter = dst.index(pos + src.span());
  auto src_iter = src.start_index();

  auto dst_size = dst.size - dst_iter;
  auto src_size = 1 + src.stop_index() - src_iter;
  auto new_size = dst.size + src_size + 1;

  // Shift destination ranges to make room for source ranges.
  dst.reserve(new_size);
  for (int i = new_size - 1; i > dst_iter + src_size; i -= 1) {
    dst.ends[i] = dst.ends[i - src_size - 1];
    dst.vals[i] = dst.vals[i - src_size - 1];
  }
  dst_iter += src_size + 1;

  // Insert the initial range up to the insertion position.
  auto prev_end = out_iter == 0 ? 0 : dst.ends[out_iter - 1];
  if (prev_end < pos) {
    dst.ends[out_iter++] = pos;
  }

  // A helper function to emit new ranges with value compression.
  auto emit = [](auto& dst, auto& out, Pos end, Val val) {
    if (out > 0 && dst.vals[out - 1] == val) {
      --out;
    }
    dst.ends[out] = end;
    dst.vals[out] = val;
    ++out;
  };

  // Insert all of the source ranges.
  for (int i = 0; i < src_size; i += 1) {
    auto src_end = pos + src.end(src_iter);
    if (prev_end < src_end) {
      emit(dst, out_iter, src_end, src.store.vals[src_iter]);
      prev_end = src_end;
    }
    ++src_iter;
  }

  // Insert all of the destination ranges.
  for (int i = 0; i < dst_size; i += 1) {
    auto dst_end = dst.ends[dst_iter];
    if (prev_end < dst_end) {
      emit(dst, out_iter, dst_end, dst.vals[dst_iter]);
      prev_end = dst_end;
    }
    ++dst_iter;
  }

  // Set the size of the destination store to the final number of elements.
  dst.size = out_iter;
}

template <typename Val>
void insert(Store<Val>& dst, const Store<Val>& src, Pos pos) {
  return insert(dst, Range(src, 0, src.span()), pos);
}

}  // namespace skimpy::detail::core
