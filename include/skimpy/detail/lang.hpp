#pragma once

#include <memory>
#include <vector>

#include "core.hpp"
#include "errors.hpp"
#include "eval.hpp"

namespace skimpy::detail::lang {

template <typename Val>
struct Op {
  virtual ~Op() = default;
  virtual core::Pos size() const = 0;
  virtual const std::type_info& type() const = 0;
  virtual std::string str() const = 0;

  template <typename OpType>
  const OpType* as() const {
    return dynamic_cast<const OpType*>(this);
  };

  template <typename OpType>
  const OpType& to() const {
    CHECK_ARGUMENT(is<OpType>());
    return dynamic_cast<const OpType&>(*this);
  };

  template <typename T>
  bool is() const {
    return as<T>();
  };
};

template <typename Val>
using OpPtr = std::shared_ptr<Op<Val>>;

template <typename Val>
struct Store : public Op<Val> {
  const std::shared_ptr<core::Store<Val>> store;

  Store(std::shared_ptr<core::Store<Val>> store) : store(std::move(store)) {}

  core::Pos size() const override {
    return store->span();
  }

  const std::type_info& type() const override {
    return typeid(Store);
  }

  std::string str() const override {
    CHECK_STATE(store->size);
    if (store->size == 1) {
      return fmt::format("store({}=>{})", store->ends[0], store->vals[0]);
    } else {
      return fmt::format("store({}=>{}, ...)", store->ends[0], store->vals[0]);
    }
  }
};

template <typename Val>
struct Slice : public Op<Val> {
  const OpPtr<Val> input;
  const int start;
  const int stop;
  const int stride;

  Slice(std::shared_ptr<Op<Val>> input, int start, int stop, int stride)
      : input(input), start(start), stop(stop), stride(stride) {
    CHECK_ARGUMENT(0 <= start);
    CHECK_ARGUMENT(start <= stop);
    CHECK_ARGUMENT(stop <= input->size());
    CHECK_ARGUMENT(stride > 0);
  }

  core::Pos size() const override {
    return 1 + (stop - start - 1) / stride;
  }

  const std::type_info& type() const override {
    return typeid(Slice);
  }

  std::string str() const override {
    return fmt::format(
        "slice({}, {}:{}:{})", input->str(), start, stop, stride);
  }
};

template <typename Val>
struct Stack : public Op<Val> {
  const std::vector<OpPtr<Val>> inputs;

  Stack(std::vector<OpPtr<Val>> inputs) : inputs(std::move(inputs)) {
    CHECK_ARGUMENT(this->inputs.size());
  }

  core::Pos size() const override {
    core::Pos ret = 0;
    for (const auto& input : inputs) {
      ret += input->size();
    }
    return ret;
  }

  const std::type_info& type() const override {
    return typeid(Stack);
  }

  std::string str() const override {
    std::string ret = "stack(" + inputs.at(0)->str();
    for (int i = 1; i < inputs.size(); i += 1) {
      ret += ", " + inputs[i]->str();
    }
    return ret + ")";
  }
};

template <typename Val>
struct Merge : public Op<Val> {
  using Fn = int (*)(int, int);

  const OpPtr<Val> lhs;
  const OpPtr<Val> rhs;
  const Fn fn;

  Merge(OpPtr<Val> lhs, OpPtr<Val> rhs, Fn fn) : lhs(lhs), rhs(rhs), fn(fn) {
    CHECK_ARGUMENT(lhs->size() == rhs->size());
  }

  core::Pos size() const override {
    return lhs->size();
  }

  const std::type_info& type() const override {
    return typeid(Merge);
  }

  std::string str() const override {
    return fmt::format("merge({}, {})", lhs->str(), rhs->str());
  }
};

template <typename Val>
struct Apply : public Op<Val> {
  using Fn = int (*)(int);

  const OpPtr<Val> input;
  const Fn fn;

  Apply(OpPtr<Val> input, Fn fn) : input(input), fn(fn) {}

  core::Pos size() const override {
    return input->size();
  }

  const std::type_info& type() const override {
    return typeid(Apply);
  }

  std::string str() const override {
    return fmt::format("apply({})", input->str());
  }
};

template <typename Val>
inline OpPtr<Val> store(core::Pos size, Val fill) {
  CHECK_ARGUMENT(size > 0);
  auto s = std::make_shared<core::Store<Val>>(1);
  s->ends[0] = size;
  s->vals[0] = fill;
  return store(std::move(s));
}

template <typename Val>
inline OpPtr<Val> store(std::shared_ptr<core::Store<Val>> store) {
  return std::make_shared<Store<Val>>(std::move(store));
}

template <typename Val>
inline OpPtr<Val> slice(OpPtr<Val> input, int start, int stop, int stride = 1) {
  return std::make_shared<Slice<Val>>(std::move(input), start, stop, stride);
}

template <typename Val>
inline OpPtr<Val> stack(std::vector<OpPtr<Val>> inputs) {
  return std::make_shared<Stack<Val>>(std::move(inputs));
}

template <typename Val, typename... Tail>
inline OpPtr<Val> stack(OpPtr<Val> head, Tail... tail) {
  std::vector<OpPtr<Val>> inputs{std::move(head), std::forward<Tail>(tail)...};
  return stack(std::move(inputs));
}

template <typename Val, typename Fun>
inline OpPtr<Val> merge(OpPtr<Val> lhs, OpPtr<Val> rhs, Fun&& func) {
  return std::make_shared<Merge<Val>>(
      std::move(lhs), std::move(rhs), std::forward<Fun>(func));
}

template <typename Val, typename Fun>
inline OpPtr<Val> apply(OpPtr<Val> input, Fun&& func) {
  return std::make_shared<Apply<Val>>(
      std::move(input), std::forward<Fun>(func));
}

template <typename Val, typename Fn>
void recurse(const OpPtr<Val>& op, Fn&& fn) {
  if (auto p = op->as<Slice<Val>>()) {
    fn(p->input);
  } else if (auto p = op->as<Stack<Val>>()) {
    for (const auto& input : p->inputs) {
      fn(input);
    }
  } else if (auto p = op->as<Merge<Val>>()) {
    fn(p->lhs);
    fn(p->rhs);
  } else if (auto p = op->as<Apply<Val>>()) {
    fn(p->input);
  }
}

template <typename Val, typename Fn>
auto substitute(const OpPtr<Val>& op, Fn&& fn) {
  if (auto p = op->as<Store<Val>>()) {
    return store(p->store);
  } else if (auto p = op->as<Slice<Val>>()) {
    return slice(fn(p->input), p->start, p->stop, p->stride);
  } else if (auto p = op->as<Stack<Val>>()) {
    std::vector<OpPtr<Val>> inputs;
    for (const auto& input : p->inputs) {
      inputs.push_back(fn(input));
    }
    return stack(std::move(inputs));
  } else if (auto p = op->as<Merge<Val>>()) {
    return merge(fn(p->lhs), fn(p->rhs), p->fn);
  } else if (auto p = op->as<Apply<Val>>()) {
    return apply(fn(p->input), p->fn);
  } else {
    CHECK_UNREACHABLE("Upsupported op type");
  }
}

template <typename Val>
auto clone(const OpPtr<Val>& op) {
  return substitute(op, [](OpPtr<Val> op) { return op; });
}

template <typename Val>
auto normalize(const OpPtr<Val>& op) {
  OpPtr<Val> ret = op;

  // Pull all stack operations to the top of the tree.
  ret = Fix([](auto fn, const OpPtr<char>& op) -> OpPtr<char> {
    // Handle the base case by wrapping the leaf store in a stack.
    if (op->is<Store<char>>()) {
      return stack(op);
    }

    // Recurse to pull all stack in all sub ops.
    auto new_op = substitute(op, fn);

    // Handle remaining cases by pulling each child stack up.
    if (auto p = new_op->as<Slice<char>>()) {
      const auto& c = p->input->to<Stack<char>>();

      int offset = 0;
      std::vector<OpPtr<char>> inputs;
      for (const auto& input : c.inputs) {
        auto rel_s = p->start - offset;
        auto start = std::max(rel_s, (p->stride + rel_s) % p->stride);
        auto stop = std::min(p->stop - offset, input->size());
        auto stride = p->stride;
        if (start < stop && offset + start >= p->start) {
          inputs.push_back(slice(input, start, stop, stride));
        }
        offset += input->size();
      }

      return stack(inputs);
    } else if (auto p = new_op->as<Stack<char>>()) {
      std::vector<OpPtr<char>> inputs;
      for (const auto& p_input : p->inputs) {
        const auto& c = p_input->to<Stack<char>>();
        for (const auto& c_input : c.inputs) {
          inputs.push_back(c_input);
        }
      }
      return stack(inputs);
    } else if (auto p = new_op->as<Merge<char>>()) {
      auto c_1 = p->lhs->to<Stack<char>>();
      auto c_2 = p->rhs->to<Stack<char>>();
      auto i_1 = c_1.inputs.begin();
      auto i_2 = c_1.inputs.begin();
      auto o_1 = 0;
      auto o_2 = 0;
      auto s_1 = 0;
      auto s_2 = 0;

      std::vector<OpPtr<char>> inputs;
      while (i_1 != c_1.inputs.end() && i_2 != c_2.inputs.end()) {
        const auto& size_1 = (*i_1)->size();
        const auto& size_2 = (*i_2)->size();
        if (o_1 + size_1 <= o_2 + size_2) {
          auto lhs = slice(*i_1, s_1, size_1, 1);
          auto rhs = slice(*i_2, s_2, size_1, 1);
          inputs.push_back(merge(lhs, rhs, p->fn));
          o_1 += size_1;
          s_1 = 0;
          s_2 = size_1;
          ++i_1;
        } else {
          auto lhs = slice(*i_1, s_1, size_2, 1);
          auto rhs = slice(*i_2, s_2, size_2, 1);
          inputs.push_back(merge(lhs, rhs, p->fn));
          o_2 += size_2;
          s_2 = 0;
          s_1 = size_2;
          ++i_2;
        }
      }
      return stack(inputs);
    } else {
      auto parent = new_op->to<Apply<char>>();
      auto c = parent.input->to<Stack<char>>();

      std::vector<OpPtr<char>> inputs;
      for (const auto& input : c.inputs) {
        inputs.push_back(apply(input, parent.fn));
      }

      return stack(inputs);
    }
  })(ret);

  // Push all slice operations to the bottom of the tree.
  ret = Fix([](auto fn, const OpPtr<char>& op) -> OpPtr<char> {
    // Handle the base case by wrapping the leaf store in a slice.
    if (op->is<Store<char>>()) {
      return slice(op, 0, op->size(), 1);
    }

    // If the op is a slice, we push it down one level in the tree.
    if (auto p = op->as<Slice<char>>()) {
      if (auto c = p->input->as<Store<char>>()) {
        return op;
      } else if (auto c = p->input->as<Slice<char>>()) {
        auto start = c->start + c->stride * p->start;
        auto stop = c->start + c->stride * p->stop;
        auto stride = c->stride * p->stride;
        return fn(slice(c->input, start, stop, stride));
      } else if (auto c = p->input->as<Merge<char>>()) {
        auto lhs = slice(c->lhs, p->start, p->stop, p->stride);
        auto rhs = slice(c->rhs, p->start, p->stop, p->stride);
        return substitute(merge(lhs, rhs, c->fn), fn);
      } else if (auto c = p->input->as<Apply<char>>()) {
        auto input = slice(c->input, p->start, p->stop, p->stride);
        return substitute(apply(input, c->fn), fn);
      } else {
        CHECK_UNREACHABLE(fmt::format("Invalid op {}", op->type().name()));
      }
    } else {
      return substitute(op, fn);
    }
  })(ret);

  return ret;
}

template <typename Val>
inline auto eval(std::shared_ptr<Op<Val>> input) {
  std::shared_ptr<Store<Val>> ret;
  return ret;
}

}  // namespace skimpy::detail::lang
