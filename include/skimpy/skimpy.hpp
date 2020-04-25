#pragma once

#include <fmt/core.h>

#include <cmath>
#include <memory>
#include <vector>

#include "detail/conv.hpp"
#include "detail/core.hpp"
#include "detail/lang.hpp"
#include "detail/util.hpp"

namespace skimpy {

namespace core = detail::core;
namespace conv = detail::conv;
namespace lang = detail::lang;

using Pos = core::Pos;

template <typename Val>
using Store = core::Store<Val>;

struct Slice {
  Pos start;
  Pos stop;
  Pos stride;

  explicit Slice(Pos stop) : Slice(0, stop) {}
  Slice(Pos start, Pos stop) : Slice(start, stop, 1) {}
  Slice(Pos start, Pos stop, Pos stride)
      : start(start), stop(stop), stride(stride) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
    CHECK_ARGUMENT(stride > 0);
  }

  Pos span() const {
    return 1 + (stop - start - 1) / stride;
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
    set(Slice(span_), make_store(span_, std::move(val)));
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
  ArrayBuilder<Val>& set(const Slice& slice, Val val) {
    set(slice, core::make_store(slice.span(), std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Array<Val>& other) {
    set(Slice(span_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, const Array<Val>& other) {
    set(slice, *other.store());
    return *this;
  }
  ArrayBuilder<Val>& set(const Store<Val>& other) {
    set(Slice(span_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, const Store<Val>& other) {
    CHECK_ARGUMENT(slice.stride == 1);
    CHECK_ARGUMENT(0 <= slice.start && slice.stop <= len());
    CHECK_ARGUMENT(slice.span() == other.span());

    // Assign the array into each intersecting block.
    auto o = slice.start;
    auto b = kBlockSize;
    for (auto s = o; s < slice.stop; s += b - (s % b)) {
      auto l = std::min(slice.stop - s, b - (s % b));
      auto& dst = stores_.at(s / b);
      core::insert(dst, core::Range(other, s - o, s + l - o), s % b);
      reserve(dst);
    }
    return *this;
  }

  // Builder methods
  Array<Val> build() {
    auto out_capacity = 1 + capacity();
    auto out_store = std::make_shared<Store<Val>>(1, out_capacity);
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
  // Value constructors
  explicit Array(std::shared_ptr<Store<Val>> store) : op_(lang::store(store)) {}
  Array(Pos span, Val fill) : op_(lang::store(span, fill)) {}

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
    std::string ret = fmt::format("[{}", get(0));
    if (len() <= 10) {
      for (int i = 1; i < len(); i += 1) {
        ret += fmt::format(", {}", get(i));
      }
    } else {
      auto last = get(len() - 1);
      ret += fmt::format(", {}, {}, {}, ..., {}", get(1), get(2), get(3), last);
    }
    return fmt::format("{}]", ret);
  }

  // Utility methods
  std::shared_ptr<Store<Val>> store() const {
    return lang::materialize(op_);
  }
  Array<Val> clone() const {
    return Array<Val>(op_);
  }
  void eval() {
    *this = Array<Val>(lang::materialize(op_));
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
    // TODO: Implement strided assignment.
    CHECK_ARGUMENT(slice.stride == 1);
    CHECK_ARGUMENT(slice.span() == other.len());
    *this = Array<Val>(lang::stack(
        lang::slice(op_, 0, slice.start, 1),
        other.op_,
        lang::slice(op_, slice.stop, len(), 1)));
  }

  // Value access methods
  Val get(Pos pos) const {
    return get(Slice(pos, pos + 1)).store()->vals[0];
  }
  Array<Val> get(const Slice& slice) const {
    CHECK_ARGUMENT(0 <= slice.start && slice.stop <= len());
    return Array<Val>(lang::slice(op_, slice.start, slice.stop, slice.stride));
  }

  // General-purpose merge and apply operations
  Array<Val> merge(const Array<Val>& other, Val (*fn)(Val, Val)) const {
    return Array<Val>(lang::merge(op_, other.op_, fn));
  }
  Array<Val> apply(Val (*fn)(Val)) const {
    return Array<Val>(lang::apply(op_, fn));
  }

  // Component-wise operations
  // TODO: Add logical operations
  // TODO: Add casting operations
  // TODO: Need to modify eval to support distinct input/output types
  // TODO: Figure out how to handle cross-type math operations
  template <typename ArrayOrVal>
  Array<Val> min(const ArrayOrVal& other) const {
    return skimpy::min(*this, other);
  }
  template <typename ArrayOrVal>
  Array<Val> max(const ArrayOrVal& other) const {
    return skimpy::max(*this, other);
  }
  template <typename ArrayOrVal>
  Array<Val> pow(const ArrayOrVal& other) const {
    return skimpy::pow(*this, other);
  }
  Array<Val> abs() const {
    return skimpy::abs(*this);
  }
  Array<Val> sqrt() const {
    return skimpy::sqrt(*this);
  }
  Array<Val> exp() const {
    return skimpy::exp(*this);
  }

 private:
  explicit Array(lang::OpPtr<Val> op) : op_(std::move(op)) {
    // There is some cost-criterion for deciding when it's better to evaluate
    // an op. One reason to eagerly evaluate is to share work across separate
    // arrays with shared histories. Another reason to eagerly evaluate is to
    // account for the non-linear costs of lang parsing and array evaluation.
    // TODO: Figure out the right way to model the cost threshold here.
    constexpr auto kFlushThreshold = 32;
    if (op_->count() > kFlushThreshold) {
      op_ = lang::store(lang::materialize(op_));
    }
  }

  lang::OpPtr<Val> op_;

  friend class ArrayBuilder<Val>;
};

// Unary arithmetic operations
template <typename Val>
Array<Val> operator+(const Array<Val>& array) {
  return array.apply([](Val a) { return +a; });
}

template <typename Val>
Array<Val> operator-(const Array<Val>& array) {
  return array.apply([](Val a) { return -a; });
}

template <typename Val>
Array<Val> operator~(const Array<Val>& array) {
  return array.apply([](Val a) { return ~a; });
}

// Binary arithmetic operations
template <typename Val>
Array<Val> operator+(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a + b; });
}
template <typename Val>
Array<Val> operator+(const Array<Val>& lhs, Val rhs) {
  return lhs + Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator+(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) + rhs;
}

template <typename Val>
Array<Val> operator-(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a - b; });
}
template <typename Val>
Array<Val> operator-(const Array<Val>& lhs, Val rhs) {
  return lhs - Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator-(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) - rhs;
}

template <typename Val>
Array<Val> operator*(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a * b; });
}
template <typename Val>
Array<Val> operator*(const Array<Val>& lhs, Val rhs) {
  return lhs * Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator*(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) * rhs;
}

template <typename Val>
Array<Val> operator/(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a / b; });
}
template <typename Val>
Array<Val> operator/(const Array<Val>& lhs, Val rhs) {
  return lhs / Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator/(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) / rhs;
}

template <typename Val>
Array<Val> operator%(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a % b; });
}
template <typename Val>
Array<Val> operator%(const Array<Val>& lhs, Val rhs) {
  return lhs % Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator%(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) % rhs;
}

// Binary bitwise operations
template <typename Val>
Array<Val> operator&(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a & b; });
}
template <typename Val>
Array<Val> operator&(const Array<Val>& lhs, Val rhs) {
  return lhs & Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator&(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) & rhs;
}

template <typename Val>
Array<Val> operator|(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a | b; });
}
template <typename Val>
Array<Val> operator|(const Array<Val>& lhs, Val rhs) {
  return lhs | Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator|(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) | rhs;
}

template <typename Val>
Array<Val> operator^(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a ^ b; });
}
template <typename Val>
Array<Val> operator^(const Array<Val>& lhs, Val rhs) {
  return lhs ^ Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator^(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) ^ rhs;
}

template <typename Val>
Array<Val> operator<<(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a << b; });
}
template <typename Val>
Array<Val> operator<<(const Array<Val>& lhs, Val rhs) {
  return lhs << Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator<<(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) << rhs;
}

template <typename Val>
Array<Val> operator>>(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return a >> b; });
}
template <typename Val>
Array<Val> operator>>(const Array<Val>& lhs, Val rhs) {
  return lhs >> Array<Val>(lhs.len(), rhs);
}
template <typename Val>
Array<Val> operator>>(Val lhs, const Array<Val>& rhs) {
  return Array<Val>(rhs.len(), lhs) >> rhs;
}

// Unary math operations
template <typename Val>
Array<Val> abs(const Array<Val>& array) {
  return array.apply([](Val a) { return std::abs(a); });
}

template <typename Val>
Array<Val> sqrt(const Array<Val>& array) {
  return array.apply([](Val a) {
    auto d_a = static_cast<double>(a);
    return static_cast<Val>(std::sqrt(d_a));
  });
}

template <typename Val>
Array<Val> exp(const Array<Val>& array) {
  return array.apply([](Val a) {
    auto d_a = static_cast<double>(a);
    return static_cast<Val>(std::exp(d_a));
  });
}

// Binary math operations
template <typename Val>
Array<Val> min(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return std::min(a, b); });
}
template <typename Val>
Array<Val> min(const Array<Val>& lhs, Val rhs) {
  return min(lhs, Array<Val>(lhs.len(), rhs));
}
template <typename Val>
Array<Val> min(Val lhs, const Array<Val>& rhs) {
  return min(Array<Val>(rhs.len(), lhs), rhs);
}

template <typename Val>
Array<Val> max(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) { return std::max(a, b); });
}
template <typename Val>
Array<Val> max(const Array<Val>& lhs, Val rhs) {
  return max(lhs, Array<Val>(lhs.len(), rhs));
}
template <typename Val>
Array<Val> max(Val lhs, const Array<Val>& rhs) {
  return max(Array<Val>(rhs.len(), lhs), rhs);
}

template <typename Val>
Array<Val> pow(const Array<Val>& lhs, const Array<Val>& rhs) {
  return lhs.merge(rhs, [](Val a, Val b) {
    auto d_a = static_cast<double>(a);
    auto d_b = static_cast<double>(b);
    return static_cast<Val>(std::pow(d_a, d_b));
  });
}
template <typename Val>
Array<Val> pow(const Array<Val>& lhs, Val rhs) {
  return pow(lhs, Array<Val>(lhs.len(), rhs));
}
template <typename Val>
Array<Val> pow(Val lhs, const Array<Val>& rhs) {
  return pow(Array<Val>(rhs.len(), lhs), rhs);
}

}  // namespace skimpy
