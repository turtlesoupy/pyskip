#pragma once

#include <fmt/core.h>

#include <cmath>
#include <memory>
#include <vector>

#include "detail/conv.hpp"
#include "detail/core.hpp"
#include "detail/lang.hpp"

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
 public:
  // Value constructors
  explicit ArrayBuilder(const Array<Val>& array) : span_(array.len()) {
    set(array);
  }
  explicit ArrayBuilder(std::shared_ptr<Store<Val>> store)
      : ArrayBuilder<Val>(Array<Val>(std::move(store))) {}
  ArrayBuilder(Pos span, Val fill)
      : ArrayBuilder<Val>(Array<Val>(span, fill)) {}

  // Metadata methods
  Pos len() const {
    return span_;
  }

  // Value assign methods
  ArrayBuilder<Val>& set(Val val) {
    set(Slice(span_), std::move(val));
    return *this;
  }
  ArrayBuilder<Val>& set(Pos pos, Val val) {
    set(pos, Array<Val>(1, std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, Val val) {
    set(slice, Array<Val>(slice.span(), std::move(val)));
    return *this;
  }
  ArrayBuilder<Val>& set(const Array<Val>& other) {
    set(Slice(span_), other);
    return *this;
  }
  ArrayBuilder<Val>& set(Pos pos, const Array<Val>& other) {
    set(Slice(pos, pos + 1), other);
    return *this;
  }
  ArrayBuilder<Val>& set(const Slice& slice, const Array<Val>& other) {
    // TODO: Implement strided assignment.
    CHECK_ARGUMENT(slice.stride == 1);
    CHECK_ARGUMENT(0 <= slice.start && slice.stop <= len());
    CHECK_ARGUMENT(slice.span() == other.len());
    sets_.emplace_back(sets_.size(), slice, other.op_);
    return *this;
  }

  // Builder methods
  Array<Val> build() {
    // TODO: This algorithm below should actually be incorporated into the lang
    // expression normalization routine. It's here now as a temporary hack.
    auto rank_fn = [](const auto& p1, const auto& p2) {
      auto s1 = std::get<1>(p1).start;
      auto s2 = std::get<1>(p2).start;
      auto i1 = std::get<0>(p1);
      auto i2 = std::get<0>(p2);
      return s1 == s2 ? i1 < i2 : s1 > s2;
    };

    auto pop_heap = [&](auto& b, auto& e) {
      auto ret = std::move(*b);
      std::pop_heap(b, e--, rank_fn);
      return ret;
    };

    auto push_heap = [&](auto& b, auto& e, auto v) {
      *e++ = std::move(v);
      std::push_heap(b, e, rank_fn);
    };

    // Prepare a heap with all set operations.
    std::make_heap(sets_.begin(), sets_.end(), rank_fn);

    // Initialize the output slices composing the output stack.
    std::vector<lang::OpPtr<Val>> slices;

    // Track the relative slice offsets of each set.
    std::unordered_map<lang::OpPtr<Val>, core::Pos> offsets;
    auto emit_slice = [&](const auto& op, auto span) {
      auto& offset = offsets[op];
      slices.push_back(slice(op, offset, offset + span, 1));
      offset += span;
    };

    // Emit all slices composing the output stack.
    auto b = sets_.begin();
    auto e = sets_.end();
    auto x = pop_heap(b, e);
    while (b != e) {
      auto y = pop_heap(b, e);
      auto [xi, xs, xo] = x;
      auto [yi, ys, yo] = y;

      if (xi > yi && xs.stop > ys.start) {
        if (ys.stop > xs.stop) {
          ys.start = xs.stop;
          push_heap(b, e, std::make_tuple(yi, ys, yo));
        }
      } else {
        CHECK_STATE(xi < yi || xs.stop == ys.start);
        CHECK_STATE(xs.start < ys.start);
        emit_slice(xo, ys.start - xs.start);
        if (xs.stop > ys.start) {
          xs.start = ys.start;
          push_heap(b, e, std::make_tuple(xi, xs, xo));
        }
        x = y;
      }
    }
    emit_slice(std::get<2>(x), len() - std::get<1>(x).start);

    Array<Val> ret(lang::materialize(lang::stack(slices)));
    sets_.clear();
    set(ret);
    return ret;
  }

 private:
  core::Pos span_;
  std::vector<std::tuple<size_t, Slice, lang::OpPtr<Val>>> sets_;
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
  std::shared_ptr<Store<Val>> store() const {
    return lang::materialize(op_);
  }
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
