#pragma once

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

  Slice(Pos stop) : Slice(0, stop) {}
  Slice(Pos start, Pos stop) : Slice(start, stop, 1) {}
  Slice(Pos start, Pos stop, Pos stride)
      : start(start), stop(stop), stride(stride) {}

  Pos span() const {
    return 1 + (stop - start - 1) / stride;
  }
};

template <typename Val>
class Array {
 public:
  // Value constructors
  explicit Array(std::shared_ptr<Store<Val>> store) : op_(lang::store(store)) {}
  Array(Pos span, Val fill) : Array<Val>(lang::store(span, fill)) {}

  // Copy and move constructors
  Array(const Array<Val>& other) : Array<Val>(lang::materialize(other.op_)) {}
  Array(Array<Val>&& other) : Array<Val>(other) {}

  // Copy and move assignment
  Array<Val>& operator=(const Array<Val>& other) {
    op_ = lang::store(lang::materialize(other.op));
    return *this;
  }
  Array<Val>& operator=(Array<Val>&& other) {
    return *this = other;
  }

  // Metadata methods
  Pos len() const {
    return op_->span();
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
    return min(*this, other);
  }
  template <typename ArrayOrVal>
  Array<Val> max(const ArrayOrVal& other) const {
    return max(*this, other);
  }
  Array<Val> abs() const {
    return abs(*this);
  }
  Array<Val> sqrt() const {
    return sqrt(*this);
  }
  Array<Val> exp() const {
    return exp(*this);
  }

 private:
  explicit Array(lang::OpPtr<Val> op) : op_(std::move(op)) {}

  lang::OpPtr<Val> op_;
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

}  // namespace skimpy
