#pragma once

#include <fmt/format.h>

#include <array>
#include <memory>
#include <tuple>
#include <vector>

#include "array.hpp"
#include "detail/box.hpp"
#include "detail/conv.hpp"
#include "detail/core.hpp"
#include "detail/lang.hpp"
#include "detail/mask.hpp"
#include "detail/step.hpp"
#include "detail/util.hpp"

namespace skimpy {

namespace box = detail::box;
namespace conv = detail::conv;
namespace core = detail::core;
namespace lang = detail::lang;
namespace mask = detail::mask;
namespace step = detail::step;
namespace util = detail::util;

template <size_t dim>
using TensorPos = std::array<core::Pos, dim>;

template <size_t dim>
struct TensorShape : public TensorPos<dim> {
  inline auto len() const {
    auto ret = 1;
    for (int i = 0; i < dim; i += 1) {
      ret *= this->at(i);
    }
    return ret;
  }
};

template <size_t dim>
inline auto make_shape(const std::array<core::Pos, dim>& shape) {
  TensorShape<dim> ret;
  for (int i = 0; i < dim; i += 1) {
    ret[i] = shape[i];
  }
  return ret;
}

template <size_t dim>
struct TensorSlice {
  static_assert(dim > 0, "Tensor slices must have positive dimension.");

  std::array<std::array<core::Pos, 3>, dim> components;

  TensorSlice(std::array<std::array<core::Pos, 3>, dim> components)
      : components(std::move(components)) {
    for (auto [c_0, c_1, c_s] : components) {
      CHECK_ARGUMENT(0 <= c_0);
      CHECK_ARGUMENT(c_0 <= c_1);
      CHECK_ARGUMENT(c_s > 0);
    }
  }

  TensorSlice(std::initializer_list<std::array<core::Pos, 3>> il) {
    CHECK_ARGUMENT(il.size() == dim);
    std::copy(il.begin(), il.end(), components.begin());
  }

  auto valid(TensorShape<dim> shape) const {
    for (int i = 0; i < dim; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      if (c_1 > shape[i]) {
        return false;
      }
    }
    return true;
  }

  auto shape() const {
    TensorShape<dim> ret;
    for (int i = 0; i < dim; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      ret[i] = (c_1 - c_0 + c_s - 1) / c_s;
    }
    return ret;
  }

  auto len() const {
    return shape().len();
  }

  auto get_fn(const TensorShape<dim>& shape) const {
    namespace sc = step::cyclic;

    auto scale = 1;
    auto i_0 = 0, i_1 = 1;
    sc::ExprNode::Ptr expr = nullptr;
    for (int i = 0; i < dim; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      i_0 += c_0 * scale;
      i_1 += (c_1 - 1) * scale;
      if (i == 0) {
        expr = sc::strided(c_1 - c_0, c_s);
      } else {
        auto reps = 1 + (c_1 - c_0 - 1) / c_s;
        auto tail = scale - expr->data.span + (c_s - 1) * scale;
        auto span = i_1 - i_0;
        if (tail > 0) {
          expr = sc::clamp(span, sc::stack(reps, expr, sc::fixed<0>(tail)));
        } else if (reps > 1) {
          expr = sc::clamp(span, sc::stack(reps, expr));
        }
      }
      scale *= shape[i];
    }

    return sc::build(i_0, i_1, expr);
  }

  auto set_fn(const TensorShape<dim>& shape) const {
    namespace sc = step::cyclic;

    auto scale = 1;
    auto i_0 = 0, i_1 = 1;
    sc::ExprNode::Ptr expr = nullptr;
    for (int i = 0; i < dim; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      i_0 += c_0 * scale;
      i_1 += (c_1 - 1) * scale;
      if (i == 0) {
        auto reps = (c_1 - c_0 - 1) / c_s;
        if (reps > 0) {
          expr = sc::scaled(reps, c_s);
        }
      } else {
        auto reps = (c_1 - c_0 - 1) / c_s;
        if (reps > 0) {
          auto prev = expr ? expr->data.span : 0;
          auto step = scale - prev + (c_s - 1) * scale;
          if (expr) {
            auto head = sc::stack(reps, sc::stack(expr, sc::scaled(1, step)));
            expr = sc::stack(head, expr);
          } else {
            expr = sc::stack(sc::stack(reps, sc::scaled(1, step)), expr);
          }
        }
      }
      scale *= shape[i];
    }

    // Shift everything up by the start position, and extend the final position
    // up to the span of the source array (i.e. the span of the given shape).
    if (expr == nullptr) {
      return sc::build(0, shape.len(), sc::scaled(1, shape.len()));
    } else {
      expr = sc::stack(sc::shift(i_0), expr);
      expr = sc::stack(expr, sc::scaled(1, shape.len() - expr->data.step));
      return sc::build(0, shape.len(), expr);
    }
  }

  auto mask(const TensorShape<dim>& shape) const {
    auto scale = 1;
    auto i_0 = 0, i_1 = 1;
    mask::Expr<box::Box> body{nullptr};
    for (int i = 0; i < dim; i += 1) {
      auto [c_0, c_1, c_s] = components[i];
      i_0 += c_0 * scale;
      i_1 += (c_1 - 1) * scale;
      if (i == 0) {
        body = mask::strided<box::Box>(c_1 - c_0, c_s);
      } else {
        auto reps = 1 + (c_1 - c_0 - 1) / c_s;
        auto tail = scale - body->data.span + (c_s - 1) * scale;
        auto span = i_1 - i_0;
        if (tail > 0) {
          body = mask::stack(reps, body, mask::range<box::Box>(tail, false));
          body = mask::clamp(span, body);
        } else {
          body = mask::clamp(span, mask::stack(reps, body));
        }
      }
      scale *= shape[i];
    }

    auto head = mask::range<box::Box>(i_0, false);
    auto tail = mask::range<box::Box>(shape.len() - i_1, false);
    return mask::build(stack(head, stack(body, tail)));
  }
};

template <size_t dim>
inline auto to_slice(const TensorPos<dim>& pos) {
  std::array<std::array<core::Pos, 3>, dim> components;
  for (int i = 0; i < dim; i += 1) {
    components[i] = {pos[i], pos[i] + 1, 1};
  }
  return TensorSlice<dim>(std::move(components));
}

template <size_t dim, typename Val>
class Tensor {
  static_assert(dim > 0, "Tensors must have positive dimension.");

 public:
  Tensor(const TensorShape<dim>& shape, std::shared_ptr<box::BoxStore> store)
      : Tensor(shape, lang::store<Val>(std::move(store))) {}

  // Metadata methods
  auto shape() const {
    return shape_;
  }
  auto len() const {
    return op_ ? lang::span(op_) : 0;
  }
  auto empty() const {
    return len() == 0;
  }
  auto str() const {
    return empty() ? "" : conv::to_string(*store());
  }
  auto repr() const {
    if (empty()) {
      return fmt::format("Tensor<{}, {}>([])", dim, typeid(Val).name());
    } else if (len() <= 10) {
      return fmt::format(
          "Tensor<{}, {}>([{}])",
          dim,
          typeid(Val).name(),
          fmt::join(conv::to_vector(*store()), ", "));
    } else {
      auto prefix = lang::materialize(lang::slice(op_, 4));
      auto suffix = lang::materialize(lang::slice(op_, len() - 1, len()));
      return fmt::format(
          "Tensor<{}, {}>([{}, ..., {}])",
          dim,
          typeid(Val).name(),
          fmt::join(conv::to_vector(*prefix), ", "),
          suffix->vals[0]);
    }
  }

  // Utility methods
  auto store() const {
    return empty() ? nullptr : lang::materialize(op_);
  }
  auto clone() const {
    return Tensor<dim, Val>(*this);
  }
  auto eval() const {
    return Tensor<dim, Val>(shape(), lang::evaluate(op_));
  }
  auto array() const {
    return Array<Val>(op_);
  }

  // Value access methods
  auto get(const TensorPos<dim>& pos) const {
    return get(to_slice(pos)).store()->vals[0];
  }
  auto get(const TensorSlice<dim>& slice) const {
    CHECK_ARGUMENT(slice.valid(shape()));
    if (slice.len() == 0) {
      return Tensor<dim, Val>();
    } else {
      return Tensor<dim, Val>(
          slice.shape(), lang::slice(op_, slice.get_fn(shape())));
    }
  }

  // Value assignment methods
  void set(const TensorPos<dim>& pos, Val val) {
    set(to_slice(pos), val);
  }
  void set(const TensorPos<dim>& pos, const Tensor<dim, Val>& other) {
    set(to_slice(pos), other);
  }
  void set(const TensorSlice<dim>& slice, Val val) {
    set(slice, Tensor<dim, Val>::make(slice.shape(), val));
  }
  void set(const TensorSlice<dim>& slice, const Tensor<dim, Val>& other) {
    CHECK_ARGUMENT(slice.valid(shape()));
    CHECK_ARGUMENT(slice.shape() == other.shape());
    if (slice.len() > 0) {
      constexpr auto fn = [](bool m, Val a, Val b) { return m ? a : b; };
      *this = Tensor<dim, Val>(
          shape(),
          lang::merge(
              lang::store<bool>(slice.mask(shape())),
              lang::slice(other.op_, slice.set_fn(shape())),
              op_,
              fn));
    }
  }

  // Static initializer routines
  static auto make(
      const TensorShape<dim>& shape, const core::Store<Val>& store) {
    auto boxes = std::make_shared<box::BoxStore>(box::box_store(store));
    return Tensor<dim, Val>(shape, boxes);
  }
  static auto make(const TensorShape<dim>& shape, Val val) {
    return make(shape, core::make_store(shape.len(), val));
  }
  static auto make(const TensorShape<dim>& shape, const Array<Val>& array) {
    return Tensor<dim, Val>(shape, array.op_);
  }

 private:
  explicit Tensor(const TensorShape<dim>& shape, lang::TypedExpr<Val> op)
      : shape_(shape), op_(std::move(op)) {
    // The lang evaluation precedure benefits from lazy evaluation since it can
    // choose an optimal coaslescing of sources and expression. The overhead of
    // managing too-large expressions however will eventually dominate the cost
    // of evaluation. We thus eagerly evaluate once expressions become too big.
    // TODO: Run experiments to measure the ideal threshold here.
    constexpr auto kFlushThreshold = 32;
    if (op_->data.size > kFlushThreshold) {
      op_ = lang::evaluate(op_);
    }
  }

  // Constructs an empty tensor
  Tensor() : shape_(TensorShape<dim>()), op_{nullptr} {}

  TensorShape<dim> shape_;
  lang::TypedExpr<Val> op_;
};

// Convenience initializers
template <size_t dim, typename Val>
inline auto make_tensor(const TensorShape<dim>& shape, Val val) {
  return Tensor<dim, Val>::make(shape, std::move(val));
}
template <size_t dim, typename Val>
inline auto make_tensor(
    const TensorShape<dim>& shape, const core::Store<Val>& store) {
  return Tensor<dim, Val>::make(shape, std::move(store));
}
template <size_t dim, typename Val>
inline auto make_tensor(
    const TensorShape<dim>& shape, const Array<Val>& array) {
  return Tensor<dim, Val>::make(shape, std::move(array));
}

// Conversion routines
template <size_t dim, typename Val>
auto from_vector(const TensorShape<dim>& shape, const std::vector<Val>& vals) {
  CHECK_ARGUMENT(len(shape) == vals.size());
  return Tensor<dim, Val>(shape, conv::to_store<Val, box::Box>(vals));
}
template <size_t dim, typename Val>
auto from_buffer(const TensorShape<dim>& shape, int size, const Val* data) {
  CHECK_ARGUMENT(len(shape) == size);
  return Tensor<dim, Val>(shape, conv::to_store<Val, box::Box>(size, data));
}

template <size_t dim, typename Val>
inline auto to_vector(const Tensor<dim, Val>& tensor) {
  return conv::to_vector(tensor.store());
}
template <size_t dim, typename Val>
inline auto to_buffer(const Tensor<dim, Val>& tensor, int* size, Val** buffer) {
  auto store = tensor.store();
  *size = store->span();
  *buffer = new Val[*size];
  conv::to_buffer(*store, *buffer);
}

template <size_t dim, typename Val>
auto to_string(const Tensor<dim, Val>& tensor) {
  return tensor.str();
}

}  // namespace skimpy
