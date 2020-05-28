#pragma once

#include <fmt/format.h>

#include <cmath>
#include <memory>
#include <vector>

#include "detail/box.hpp"
#include "detail/conv.hpp"
#include "detail/core.hpp"
#include "detail/lang.hpp"
#include "detail/mask.hpp"
#include "detail/step.hpp"
#include "macros.hpp"

namespace skimpy {

namespace box = detail::box;
namespace conv = detail::conv;
namespace core = detail::core;
namespace lang = detail::lang;
namespace mask = detail::mask;
namespace step = detail::step;

struct Slice {
  core::Pos start;
  core::Pos stop;
  core::Pos stride;

  Slice(core::Pos start, core::Pos stop, core::Pos stride)
      : start(start), stop(stop), stride(stride) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
    CHECK_ARGUMENT(stride > 0);
  }

  // Convenience constructors
  Slice(core::Pos start, core::Pos stop) : Slice(start, stop, 1) {}
  explicit Slice(core::Pos stop) : Slice(0, stop) {}

  // Array constructors
  Slice(std::array<core::Pos, 3> array) : Slice(array[0], array[1], array[2]) {}
  Slice(std::array<core::Pos, 2> array) : Slice(array[0], array[1]) {}
  Slice(std::array<core::Pos, 1> array) : Slice(array[1]) {}

  auto len() const {
    return (stop - start + stride - 1) / stride;
  }

  auto get_fn() const {
    return step::cyclic::slice(start, stop, step::cyclic::stride_fn(stride));
  }

  auto set_fn(core::Pos len) const {
    return step::cyclic::insert_fn(len, start, stop, stride);
  }

  auto mask(core::Pos len) const {
    return mask::stride_mask<box::Box, bool>(len, start, stop, stride);
  }
};

template <typename Val>
class Array {
 public:
  Array() : op_{nullptr} {}
  Array(std::shared_ptr<box::BoxStore> store) : op_(lang::store<Val>(store)) {}

  // Metadata methods
  auto len() const {
    return op_ ? lang::span(op_) : 0;
  }
  auto empty() const {
    return len() == 0;
  }
  auto str() const {
    return empty() ? std::string("") : conv::to_string(*store());
  }
  auto repr() const {
    if (empty()) {
      return fmt::format("Array<{}>([])", typeid(Val).name());
    } else if (len() <= 10) {
      return fmt::format(
          "Array<{}>([{}])",
          typeid(Val).name(),
          fmt::join(conv::to_vector(*store()), ", "));
    } else {
      return fmt::format(
          "Array<{}>([{}, ..., {}])",
          typeid(Val).name(),
          fmt::join(conv::to_vector(*get(Slice(4)).store()), ", "),
          get(len() - 1));
    }
  }

  // Utility methods
  auto store() const {
    return empty() ? nullptr : lang::materialize(op_);
  }
  auto clone() const {
    return Array<Val>(*this);
  }
  auto eval() const {
    return Array<Val>(lang::evaluate(op_));
  }

  // Value access methods
  auto get(core::Pos pos) const {
    return get(Slice(pos, pos + 1)).store()->vals[0];
  }
  auto get(const Slice& slice) const {
    CHECK_ARGUMENT(slice.stop <= len());
    if (slice.len() == 0) {
      return Array<Val>();
    } else {
      return Array<Val>(lang::slice(op_, slice.get_fn()));
    }
  }

  // Value assign methods
  void set(core::Pos pos, Val val) {
    set(pos, Array<Val>(lang::store(1, val)));
  }
  void set(const Slice& slice, Val val) {
    set(slice, Array<Val>(lang::store(slice.len(), val)));
  }
  void set(core::Pos pos, const Array<Val>& other) {
    set(Slice(pos, pos + 1), other);
  }
  void set(const Slice& slice, const Array<Val>& other) {
    CHECK_ARGUMENT(slice.stop <= len());
    CHECK_ARGUMENT(slice.len() == other.len());
    if (slice.len() > 0) {
      constexpr auto fn = [](bool m, Val a, Val b) { return m ? a : b; };
      *this = Array<Val>(lang::merge(
          lang::store<bool>(slice.mask(len())),
          lang::slice(other.op_, slice.set_fn(len())),
          op_,
          fn));
    }
  }

  // General-purpose merge operations
  template <typename Out, Out (*fn)(Val)>
  auto merge() const {
    if (empty()) {
      return Array<Out>(lang::cast<Out>(op_));
    }
    return Array<Out>(lang::merge<Out, Val, fn>(op_));
  }
  template <typename Out, typename Arg, Out (*fn)(Val, Arg)>
  auto merge(const Array<Arg>& other) const {
    CHECK_ARGUMENT(len() == other.len());
    if (empty()) {
      return Array<Out>(lang::cast<Out>(op_));
    }
    return Array<Out>(lang::merge<Out, Val, Arg, fn>(op_, other.op_));
  }
  template <
      typename Out,
      typename Arg1,
      typename Arg2,
      Out (*fn)(Val, Arg1, Arg2)>
  auto merge(const Array<Arg1>& a, const Array<Arg2>& b) const {
    CHECK_ARGUMENT(len() == a.len());
    CHECK_ARGUMENT(len() == b.len());
    if (empty()) {
      return Array<Out>(lang::cast<Out>(op_));
    }
    return Array<Out>(lang::merge<Out, Val, Arg1, Arg2, fn>(op_, a.op_, b.op_));
  }

  // Type-preserving merge operations
  template <Val (*fn)(Val)>
  auto merge() const {
    return merge<Val, fn>();
  }
  template <Val (*fn)(Val, Val)>
  auto merge(const Array<Val>& other) const {
    return merge<Val, Val, fn>(other);
  }
  template <Val (*fn)(Val, Val, Val)>
  auto merge(const Array<Val>& a, const Array<Val>& b) const {
    return merge<Val, Val, Val, fn>(a, b);
  }

  // Static initializer routines
  static auto make(const core::Store<Val>& store) {
    return Array<Val>(std::make_shared<box::BoxStore>(box::box_store(store)));
  }
  static auto make(core::Pos len, Val val) {
    if (len == 0) {
      return Array<Val>();
    }
    return make(core::make_store(len, val));
  }

 private:
  explicit Array(lang::TypedExpr<Val> op) : op_(std::move(op)) {
    // The lang evaluation precedure benefits from lazy evaluation since it can
    // choose an optimal coaslescing of sources and expression. The overhead of
    // managing too large expressions however will eventually dominate the cost
    // of evaluation. We thus eagerly evaluate once expressions become too big.
    // TODO: Run experiments to measure the ideal threshold here.
    constexpr auto kFlushThreshold = 32;
    if (op_ && op_->data.size > kFlushThreshold) {
      op_ = lang::evaluate(op_);
    }
  }

  lang::TypedExpr<Val> op_;

  template <typename T>
  friend class Array;

  template <size_t dim, typename T>
  friend class Tensor;
};

// Convenience constructors
template <typename Val>
auto make_array(core::Pos len, Val val) {
  return Array<Val>::make(len, val);
}
template <typename Val>
auto make_array(const core::Store<Val>& store) {
  return Array<Val>::make(store);
}

// Conversion routines
template <typename Val>
auto from_vector(const std::vector<Val>& vals) {
  if (vals.empty()) {
    return Array<Val>();
  }
  return Array<Val>(conv::to_store<Val, box::Box>(vals));
}
template <typename Val>
auto from_buffer(int size, const Val* data) {
  if (size == 0) {
    return Array<Val>();
  }
  return Array<Val>(conv::to_store<Val, box::Box>(size, data));
}

template <typename Val>
auto to_vector(const Array<Val>& array) {
  if (array.empty()) {
    return std::vector<Val>();
  }
  return conv::to_vector(*array.store());
}
template <typename Val>
void to_buffer(const Array<Val>& array, int* size, Val** buffer) {
  if (array.empty()) {
    *size = 0;
    *buffer = nullptr;
    return;
  }
  auto store = array.store();
  *size = store->span();
  *buffer = new Val[*size];
  conv::to_buffer(*store, *buffer);
}

template <typename Val>
auto to_string(const Array<Val>& array) {
  return array.str();
}

// Casting operations
template <typename Out, typename Val>
Array<Out> cast(const Array<Val>& array) {
  constexpr Out (*fn)(Val) = [](Val a) { return static_cast<Out>(a); };
  return array.template merge<Out, fn>();
}

// Unary arithmetic operations
UNARY_ARRAY_OP_SIMPLE(operator+, [](Val a) { return +a; })
UNARY_ARRAY_OP_SIMPLE(operator-, [](Val a) { return -a; })
UNARY_ARRAY_OP_SIMPLE(operator~, [](Val a) { return ~a; })

// Binary arithmetic operations
BINARY_ARRAY_OP_SIMPLE(operator+, [](Val a, Val b) { return a + b; })
BINARY_ARRAY_OP_SIMPLE(operator-, [](Val a, Val b) { return a - b; })
BINARY_ARRAY_OP_SIMPLE(operator*, [](Val a, Val b) { return a * b; })
BINARY_ARRAY_OP_SIMPLE(operator/, [](Val a, Val b) { return a / b; })
BINARY_ARRAY_OP_SIMPLE(operator%, [](Val a, Val b) { return a % b; })

template <>
inline Array<float> operator%(const Array<float>& lhs, const Array<float>& rhs) {
  constexpr auto fn = [](float a, float b) { return fmodf(a, b); };
  return lhs.template merge<fn>(rhs);
}

// Binary bitwise operations
BINARY_ARRAY_OP_SIMPLE(operator&, [](Val a, Val b) { return a & b; })
BINARY_ARRAY_OP_SIMPLE(operator|, [](Val a, Val b) { return a | b; })
BINARY_ARRAY_OP_SIMPLE(operator^, [](Val a, Val b) { return a ^ b; })
BINARY_ARRAY_OP_SIMPLE(operator<<, [](Val a, Val b) { return a << b; })
BINARY_ARRAY_OP_SIMPLE(operator>>, [](Val a, Val b) { return a >> b; })

// Unary math operations
UNARY_ARRAY_OP_SIMPLE(abs, [](Val a) { return std::abs(a); })
UNARY_ARRAY_OP_SIMPLE(exp, [](Val a) { return static_cast<Val>(std::exp(a)); })
UNARY_ARRAY_OP_SIMPLE(sqrt, [](Val a) {
  return static_cast<Val>(std::sqrt(a));
})

// Binary math operations
BINARY_ARRAY_OP_SIMPLE(min, [](Val a, Val b) { return std::min(a, b); })
BINARY_ARRAY_OP_SIMPLE(max, [](Val a, Val b) { return std::max(a, b); })
BINARY_ARRAY_OP_SIMPLE(pow, [](Val a, Val b) {
  return static_cast<Val>(std::pow(a, b));
})

// Ternary math operations
TERNARY_ARRAY_OP(splat, bool, Val, Val, Val, [](bool m, Val a, Val b) {
  return m ? a : b;
})

// Logical operations
UNARY_ARRAY_OP(operator!, bool, [](Val a) { return !a; })
BINARY_ARRAY_OP(operator&&, Val, Val, bool, [](Val a, Val b) { return a && b; })
BINARY_ARRAY_OP(operator||, Val, Val, bool, [](Val a, Val b) { return a || b; })
BINARY_ARRAY_OP(operator==, Val, Val, bool, [](Val a, Val b) { return a == b; })
BINARY_ARRAY_OP(operator!=, Val, Val, bool, [](Val a, Val b) { return a != b; })

// Comparison operations
BINARY_ARRAY_OP(operator<, Val, Val, bool, [](Val a, Val b) { return a < b; })
BINARY_ARRAY_OP(operator>, Val, Val, bool, [](Val a, Val b) { return a > b; })
BINARY_ARRAY_OP(operator<=, Val, Val, bool, [](Val a, Val b) { return a <= b; })
BINARY_ARRAY_OP(operator>=, Val, Val, bool, [](Val a, Val b) { return a >= b; })

}  // namespace skimpy
