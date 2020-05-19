#pragma once

#include <fmt/core.h>

#include <cmath>
#include <memory>
#include <vector>

#include "detail/box.hpp"
#include "detail/conv.hpp"
#include "detail/core.hpp"
#include "detail/lang.hpp"
#include "detail/util.hpp"

namespace skimpy {

namespace core = detail::core;
namespace conv = detail::conv;
namespace lang = detail::lang;
namespace step = detail::step;

using Pos = core::Pos;

template <typename Val>
using Store = core::Store<Val>;

struct Band {
  Pos start;
  Pos stop;

  Band(Pos start, Pos stop) : start(start), stop(stop) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
  }

  explicit Band(Pos stop) : Band(0, stop) {}

  Pos span() const {
    return stop - start;
  }
};

struct Slice {
  Pos start;
  Pos stop;
  Pos stride;

  Slice(Pos start, Pos stop, Pos stride)
      : start(start), stop(stop), stride(stride) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
    CHECK_ARGUMENT(stride > 0);
  }

  Slice(Pos start, Pos stop) : Slice(start, stop, 1) {}
  explicit Slice(Pos stop) : Slice(0, stop) {}

  Pos span() const {
    return 1 + (stop - start - 1) / stride;
  }

  auto step_fn() const {
    lang::StepFn ret;
    if (stride > 1) {
      ret = step::stride_fn(stride);
    }
    return ret;
  }
};

template <typename Val>
class Array;

template <typename Val>
class ArrayBuilder {
  static constexpr auto kBlockSize = 4096;

 public:
  // Value constructors
  ArrayBuilder(Pos span, Val fill) : span_(span) {
    auto k = 1 + (span - 1) / kBlockSize;
    stores_.reserve(k);
    for (int i = 0; i < k; i += 1) {
      auto span = std::min(kBlockSize, span_ - i * kBlockSize);
      stores_.push_back(core::make_store(span, fill));
    }
  }
  explicit ArrayBuilder(const Array<Val>& array)
      : ArrayBuilder(array.len(), 0) {
    set(array);
  }
  explicit ArrayBuilder(std::shared_ptr<Store<Val>> store)
      : ArrayBuilder(store.span(), 0) {
    set(ArrayVal<Val>(store));
  }

  // Metadata methods
  Pos len() const {
    return span_;
  }

  // Value assign methods
  ArrayBuilder<Val>& set(Val val) {
    set(Band(span_), make_store(span_, std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(Pos pos, Val val) {
    CHECK_ARGUMENT(pos < span_);
    auto i = pos / kBlockSize;
    auto& dst = stores_.at(i);
    core::set(dst, pos - i * kBlockSize, std::move(val));
    reserve(dst);
    return *this;
  }
  ArrayBuilder<Val>& set(const Band& band, Val val) {
    set(band, core::make_store(band.span(), std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Array<Val>& other) {
    set(Band(span_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Band& band, const Array<Val>& other) {
    set(band, *other.store());
    return *this;
  }
  ArrayBuilder<Val>& set(const Store<Val>& other) {
    set(Band(span_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Band& band, const Store<Val>& other) {
    CHECK_ARGUMENT(band.stop <= len());
    CHECK_ARGUMENT(band.span() == other.span());

    // Assign the array into each intersecting block.
    auto o = band.start;
    auto b = kBlockSize;
    for (auto s = o; s < band.stop; s += b - (s % b)) {
      auto l = std::min(band.stop - s, b - (s % b));
      auto& dst = stores_.at(s / b);
      core::insert(dst, core::Range(other, s - o, s + l - o), s % b);
      reserve(dst);
    }
    return *this;
  }

  // Builder methods
  Array<Val> build() {
    auto out_capacity = 1 + capacity();
    auto out_store = std::make_shared<Store<box::Box>>(1, out_capacity);
    out_store->ends[0] = span_;
    for (int i = 0; i < stores_.size(); i += 1) {
      core::insert(*out_store, stores_[i], kBlockSize * i);
    }
    return Array<Val>(out_store);
  }

 private:
  auto capacity() const {
    int ret = 0;
    for (const auto& store : stores_) {
      ret += store.size;
    }
    return ret;
  }

  void reserve(Store<Val>& store) {
    if (!detail::is_power_of_two(store.capacity)) {
      store.reserve(detail::round_up_to_power_of_two(store.capacity));
    }
  }

  core::Pos span_;
  std::vector<Store<Val>> stores_;
};

template <typename Val>
class Array {
 public:
  Array(std::shared_ptr<Store<box::Box>> store)
      : op_(lang::store(std::move(store))) {}

  // Copy and move constructors
  Array(const Array<Val>& other) : op_(other.op_) {}
  Array(Array<Val>&& other) : op_(std::move(other.op_)) {}

  // Copy and move assignment
  Array<Val>& operator=(const Array<Val>& other) {
    op_ = other.op_;
    return *this;
  }
  Array<Val>& operator=(Array<Val>&& other) {
    op_ = std::move(other.op_);
    return *this;
  }

  // Metadata methods
  Pos len() const {
    return op_->span();
  }
  std::string str() const {
    return conv::to_string(store());
  }
  std::string repr() const {
    if (len() <= 10) {
      return fmt::format(
          "Array<{}>([{}])",
          typeid(Val).name(),
          fmt::join(", ", conv::to_vector(store())));
    } else {
      return fmt::format(
          "Array<{}>([{}, ..., {}])",
          typeid(Val).name(),
          fmt::join(", ", conv::to_vector(get(Slice(4)).store())),
          get(len() - 1));
    }
  }

  // Utility methods
  auto store() const {
    return lang::materialize(op_);
  }
  auto clone() const {
    return Array<Val>(*this);
  }
  auto eval() const {
    return Array<Val>(lang::evaluate(op_));
  }

  // Value assign methods
  void set(Val val) {
    set(Slice(0, len()), val);
  }
  void set(Pos pos, Val val) {
    set(pos, Array<Val>(1, std::move(val)));
  }
  void set(const Slice& slice, Val val) {
    set(slice, Array<Val>(slice.span(), std::move(val)));
  }
  void set(const Array<Val>& other) {
    set(Slice(0, len()), other);
  }
  void set(Pos pos, const Array<Val>& other) {
    set(Slice(pos, pos + 1), other);
  }
  void set(const Slice& slice, const Array<Val>& other) {
    CHECK_ARGUMENT(slice.stop <= len());
    CHECK_ARGUMENT(slice.span() == other.len());
    *this = splat(
        mask::stride_mask(len(), slice.start, slice.stop, slice.stride),
        *this,
        lang::slice(
            other,
            0,
            other.len(),
            step::insert_fn(len(), slice.start, slice.stop, slice.stride)));
  }

  // Value access methods
  Val get(Pos pos) const {
    return get(Slice(pos, pos + 1)).store()->vals[0];
  }
  Array<Val> get(const Slice& slice) const {
    CHECK_ARGUMENT(slice.stop <= len());
    return Array<Val>(
        lang::slice(op_, slice.start, slice.stop, slice.step_fn()));
  }

  // General-purpose merge and apply operations
  template <Val (*fn)(Val)>
  Array<Val> merge() const {
    return Array<Val>(lang::merge(op_, fn));
  }
  template <Val (*fn)(Val, Val)>
  Array<Val> merge(const Array<Val>& other) const {
    return Array<Val>(lang::merge(op_, other.op_, fn));
  }
  template <Val (*fn)(Val, Val, Val)>
  Array<Val> merge(const Array<Val>& a, const Array<Val>& b) const {
    return Array<Val>(lang::merge(op_, a.op_, b.op_, fn));
  }

 private:
  explicit Array(lang::TypedExpr<Val> op) : op_(std::move(op)) {
    // The lang evaluation precedure benefits from lazy evaluation since it can
    // choose an optimal coaslescing of sources and expression. The overhead of
    // managing too large expressions however will eventually dominate the cost
    // of evaluation. We thus eagerly evaluate once expressions become too big.
    // TODO: Run experiments to measure the ideal threshold here.
    constexpr auto kFlushThreshold = 32;
    if (op_->data.size > kFlushThreshold) {
      op_ = lang::evaluate(op_);
    }
  }

  lang::TypedExpr<Val> op_;

  friend class ArrayBuilder<Val>;
};

// Convenience constructors
template <typename Val>
auto make_array(Pos span, Val fill) {
  return Array<Val>(lang::store(span, fill));
}
template <typename Val>
auto make_array(std::shared_ptr<Store<Val>> store) {
  return Array<Val>(lang::store(store));
}
template <typename Val>
auto make_array(const Store<Val>& store) {
  return Array<Val>(lang::store(store));
}

// Conversion routines
template <typename Val>
auto from_buffer(int size, const Val* data) {
  return Array<Val>(conv::to_store(size, data));
}

template <typename Val>
auto to_vector(const Array<Val>& array) {
  return conv::to_vector(array.store());
}

template <typename Val>
void to_buffer(const Array<Val>& array, int* size, Val** buffer) {
  auto store = array.store();
  *size = store->span();
  *buffer = new Val[*size];
  conv::to_buffer(*store, *buffer);
}

// Unary arithmetic operations
template <typename Val>
Array<Val> operator+(const Array<Val>& array) {
  return array.merge<[](Val a) { return +a; }>();
}

template <typename Val>
Array<Val> operator-(const Array<Val>& array) {
  return array.merge<[](Val a) { return -a; }>();
}

template <typename Val>
Array<Val> operator~(const Array<Val>& array) {
  return array.merge<[](Val a) { return ~a; }>();
}

// Binary arithmetic operations
template <typename Val>
Array<Val> operator+(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a + b; }>(rhs);
}
template <typename Val>
Array<Val> operator+(const Array<Val>& lhs, Val rhs) {
  return lhs + make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator+(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) + rhs;
}

template <typename Val>
Array<Val> operator-(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a - b; }>(rhs);
}
template <typename Val>
Array<Val> operator-(const Array<Val>& lhs, Val rhs) {
  return lhs - make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator-(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) - rhs;
}

template <typename Val>
Array<Val> operator*(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a * b; }>(rhs);
}
template <typename Val>
Array<Val> operator*(const Array<Val>& lhs, Val rhs) {
  return lhs * make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator*(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) * rhs;
}

template <typename Val>
Array<Val> operator/(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a / b; }>(rhs);
}
template <typename Val>
Array<Val> operator/(const Array<Val>& lhs, Val rhs) {
  return lhs / make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator/(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) / rhs;
}

template <typename Val>
Array<Val> operator%(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a % b; }>(rhs);
}
template <typename Val>
Array<Val> operator%(const Array<Val>& lhs, Val rhs) {
  return lhs % make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator%(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) % rhs;
}

// Binary bitwise operations
template <typename Val>
Array<Val> operator&(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a & b; }>(rhs);
}
template <typename Val>
Array<Val> operator&(const Array<Val>& lhs, Val rhs) {
  return lhs & make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator&(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) & rhs;
}

template <typename Val>
Array<Val> operator|(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a | b; }>(rhs);
}
template <typename Val>
Array<Val> operator|(const Array<Val>& lhs, Val rhs) {
  return lhs | make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator|(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) | rhs;
}

template <typename Val>
Array<Val> operator^(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a ^ b; }>(rhs);
}
template <typename Val>
Array<Val> operator^(const Array<Val>& lhs, Val rhs) {
  return lhs ^ make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator^(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) ^ rhs;
}

template <typename Val>
Array<Val> operator<<(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a << b; }>(rhs);
}
template <typename Val>
Array<Val> operator<<(const Array<Val>& lhs, Val rhs) {
  return lhs << make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator<<(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) << rhs;
}

template <typename Val>
Array<Val> operator>>(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return a >> b; }>(rhs);
}
template <typename Val>
Array<Val> operator>>(const Array<Val>& lhs, Val rhs) {
  return lhs >> make_array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator>>(Val lhs, const Array<Val>& rhs) {
  return make_array<Val>(rhs.len(), lhs) >> rhs;
}

// Unary math operations
template <typename Val>
Array<Val> abs(const Array<Val>& array) {
  return array.merge<[](Val a) { return std::abs(a); }>();
}

// Binary math operations
template <typename Val>
Array<Val> min(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return std::min(a, b); }>(rhs);
}
template <typename Val>
Array<Val> min(const Array<Val>& lhs, Val rhs) {
  return min(lhs, make_array<Val>(lhs.len(), rhs));
}
template <typename Val>
Array<Val> min(Val lhs, const Array<Val>& rhs) {
  return min(make_array<Val>(rhs.len(), lhs), rhs);
}

template <typename Val>
Array<Val> max(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge<[](Val a, Val b) { return std::max(a, b); }>(rhs);
}
template <typename Val>
Array<Val> max(const Array<Val>& lhs, Val rhs) {
  return max(lhs, make_array<Val>(lhs.len(), rhs));
}
template <typename Val>
Array<Val> max(Val lhs, const Array<Val>& rhs) {
  return max(make_array<Val>(rhs.len(), lhs), rhs);
}

// Ternary math oeprators
template <typename Val>
Array<Val> splat(
    const Array<bool>& mask, const Array<Val>& lhs, const Array<Val>& rhs) {
  return mask.merge<[](bool m, Val a, Val b) { return m ? a : b; }>(lhs, rhs);
}
template <typename Val>
Array<Val> splat(const Array<bool>& mask, const Array<Val>& lhs, Val rhs) {
  return splat(mask, lhs, make_array<Val>(mask.len(), rhs));
}
template <typename Val>
Array<Val> splat(const Array<bool>& mask, Val lhs, const Array<Val>& rhs) {
  return splat(mask, make_array<Val>(mask.len(), lhs), rhs);
}
template <typename Val>
Array<Val> splat(const Array<bool>& mask, Val lhs, Val rhs) {
  auto len = mask.len();
  return splat(mask, make_array<Val>(len, lhs), make_array<Val>(len, rhs));
}

// TODO: Add logical operations
// TODO: Add casting operations

}  // namespace skimpy
