#pragma once

#include <fmt/core.h>

#include <cmath>
#include <memory>
#include <vector>

#include "detail/core.hpp"
#include "detail/lang.hpp"

namespace skimpy {

namespace core = detail::core;
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
 public:
  // Value constructors
  explicit ArrayBuilder(std::shared_ptr<Store<Val>> store)
      : op_(lang::store(store)), store_size_(store->size) {}
  explicit ArrayBuilder(const Array<Val>& array)
      : ArrayBuilder(lang::materialize(array.op_)) {}
  ArrayBuilder(Pos span, Val fill) : op_(lang::store(span, fill)) {}

  // Metadata methods
  Pos len() const {
    return op_->span();
  }
  std::string str() const {
    return lang::str(op_);
  }
  std::string plan() const {
    return lang::str(lang::normalize(op_));
  }

  // Value assign methods
  ArrayBuilder<Val>& set(Pos pos, Val val) {
    set(pos, Array<Val>(1, std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, Val val) {
    set(slice, Array<Val>(slice.span(), std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(Pos pos, const Array<Val>& other) {
    set(Slice(pos, pos + 1), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, const Array<Val>& other) {
    // TODO: Implement strided assignment.
    CHECK_ARGUMENT(slice.stride == 1);
    CHECK_ARGUMENT(slice.span() == other.len());
    auto l = lang::slice(op_, 0, slice.start, 1);
    auto r = lang::slice(op_, slice.stop, len(), 1);
    op_ = lang::stack(l, other.op_, r);
    maybe_materialize();
    return *this;
  }

  // Builder methods
  Array<Val> build() {
    return Array<Val>(lang::materialize(op_));
  }

 private:
  void maybe_materialize() {
    auto op_count = count(op_);
    if (op_count * op_count > store_size_) {
      auto store = lang::materialize(op_);
      store_size_ = store->size;
      op_ = lang::store(store);
    }
  }

  lang::OpPtr<Val> op_;
  int store_size_;
};

template <typename Val>
class Array {
 public:
  // Value constructors
  explicit Array(std::shared_ptr<Store<Val>> store) : op_(lang::store(store)) {}
  Array(Pos span, Val fill) : op_(lang::store(span, fill)) {}

  // Copy and move constructors
  Array(const Array<Val>& other) : Array<Val>(lang::materialize(other.op_)) {}
  Array(Array<Val>&& other) : Array<Val>(other) {}

  // Copy and move assignment
  Array<Val>& operator=(const Array<Val>& other) {
    op_ = lang::store(lang::materialize(other.op_));
    return *this;
  }
  Array<Val>& operator=(Array<Val>&& other) {
    return *this = other;
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

  // Value assign methods
  void set(Pos pos, Val val) {
    set(pos, Array<Val>(1, std::move(val)));
  }
  void set(const Slice& slice, Val val) {
    set(slice, Array<Val>(slice.span(), std::move(val)));
  }
  void set(Pos pos, const Array<Val>& other) {
    set(Slice(pos, pos + 1), other);
  }
  void set(const Slice& slice, const Array<Val>& other) {
    // TODO: Implement strided assignment.
    CHECK_ARGUMENT(slice.stride == 1);
    CHECK_ARGUMENT(slice.span() == other.len());
    auto l = lang::slice(op_, 0, slice.start, 1);
    auto r = lang::slice(op_, slice.stop, len(), 1);
    op_ = lang::store(lang::materialize(lang::stack(l, other.op_, r)));
  }

  // Value access methods
  Val get(Pos pos) const {
    return lang::materialize(get(Slice(pos, pos + 1)).op_)->vals[0];
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
  explicit Array(lang::OpPtr<Val> op) : op_(std::move(op)) {}

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
