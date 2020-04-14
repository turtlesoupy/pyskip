#pragma once

#include <fmt/core.h>

#include <memory>
#include <vector>

#include "core.hpp"
#include "errors.hpp"
#include "eval.hpp"

namespace skimpy::detail::lang {

template <typename Val>
struct Op {
  virtual ~Op() = default;
  virtual core::Pos span() const = 0;
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

  core::Pos span() const override {
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
    CHECK_ARGUMENT(stop <= input->span());
    CHECK_ARGUMENT(stride > 0);
  }

  core::Pos span() const override {
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

  Stack(std::vector<OpPtr<Val>> inputs) : inputs(std::move(inputs)) {}

  core::Pos span() const override {
    core::Pos ret = 0;
    for (const auto& input : inputs) {
      ret += input->span();
    }
    return ret;
  }

  const std::type_info& type() const override {
    return typeid(Stack);
  }

  std::string str() const override {
    std::string ret = "stack(";
    if (inputs.size()) {
      ret += inputs.at(0)->str();
    }
    for (int i = 1; i < inputs.size(); i += 1) {
      ret += ", " + inputs[i]->str();
    }
    return ret + ")";
  }
};

template <typename Val>
struct Merge : public Op<Val> {
  using Fn = Val (*)(Val, Val);

  const OpPtr<Val> lhs;
  const OpPtr<Val> rhs;
  const Fn fn;

  Merge(OpPtr<Val> lhs, OpPtr<Val> rhs, Fn fn) : lhs(lhs), rhs(rhs), fn(fn) {
    CHECK_ARGUMENT(lhs->span() == rhs->span());
  }

  core::Pos span() const override {
    return lhs->span();
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
  using Fn = Val (*)(Val);

  const OpPtr<Val> input;
  const Fn fn;

  Apply(OpPtr<Val> input, Fn fn) : input(input), fn(fn) {}

  core::Pos span() const override {
    return input->span();
  }

  const std::type_info& type() const override {
    return typeid(Apply);
  }

  std::string str() const override {
    return fmt::format("apply({})", input->str());
  }
};

template <typename Val>
inline OpPtr<Val> store(core::Pos span, Val fill) {
  CHECK_ARGUMENT(span > 0);
  auto s = std::make_shared<core::Store<Val>>(1);
  s->ends[0] = span;
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
auto depth(const OpPtr<Val>& op) {
  auto max_depth = 0;
  Fix([&, depth = 0](auto& fn, const OpPtr<Val>& op) mutable -> void {
    max_depth = std::max(max_depth, ++depth);
    recurse(op, fn);
    --depth;
  })(op);
  return max_depth;
}

template <typename Val>
auto normalize(const OpPtr<Val>& op) {
  OpPtr<Val> ret = op;

  // Pull all stack operations to the top of the tree.
  ret = Fix([](auto fn, const OpPtr<Val>& op) -> OpPtr<Val> {
    // Handle the base case by wrapping the leaf store in a stack.
    if (op->is<Store<Val>>()) {
      return stack(op);
    }

    // Recurse to pull all stack in all sub ops.
    auto new_op = substitute(op, fn);

    // Handle remaining cases by pulling each child stack up.
    if (auto p = new_op->as<Slice<Val>>()) {
      const auto& c = p->input->to<Stack<Val>>();

      int offset = 0;
      std::vector<OpPtr<Val>> inputs;
      for (const auto& input : c.inputs) {
        auto rel_s = p->start - offset;
        auto start = std::max(rel_s, (p->stride + rel_s) % p->stride);
        auto stop = std::min(p->stop - offset, input->span());
        auto stride = p->stride;
        if (start < stop && offset + start >= p->start) {
          inputs.push_back(slice(input, start, stop, stride));
        }
        offset += input->span();
      }

      return stack(inputs);
    } else if (auto p = new_op->as<Stack<Val>>()) {
      std::vector<OpPtr<Val>> inputs;

      // Flatten the stack of stacks into a single stack.
      for (const auto& p_input : p->inputs) {
        const auto& c = p_input->to<Stack<Val>>();
        for (const auto& c_input : c.inputs) {
          inputs.push_back(c_input);
        }
      }

      return stack(inputs);
    } else if (auto p = new_op->as<Merge<Val>>()) {
      auto c_1 = p->lhs->to<Stack<Val>>();
      auto c_2 = p->rhs->to<Stack<Val>>();
      auto i_1 = c_1.inputs.begin();
      auto i_2 = c_2.inputs.begin();
      auto o_1 = 0;
      auto o_2 = 0;
      auto s_1 = 0;
      auto s_2 = 0;

      // Merge across both the lhs and rhs stacks.
      std::vector<OpPtr<Val>> inputs;
      while (i_1 != c_1.inputs.end() && i_2 != c_2.inputs.end()) {
        auto span_1 = (*i_1)->span();
        auto span_2 = (*i_2)->span();
        if (o_1 + span_1 < o_2 + span_2) {
          auto lhs = slice(*i_1, s_1, span_1, 1);
          auto rhs = slice(*i_2, s_2, s_2 + span_1 - s_1, 1);
          inputs.push_back(merge(lhs, rhs, p->fn));
          s_2 += span_1 - s_1;
          o_1 += span_1;
          s_1 = 0;
          ++i_1;
        } else if (o_1 + span_1 > o_2 + span_2) {
          auto lhs = slice(*i_1, s_1, s_1 + span_2 - s_2, 1);
          auto rhs = slice(*i_2, s_2, span_2, 1);
          inputs.push_back(merge(lhs, rhs, p->fn));
          s_1 += span_2 - s_2;
          o_2 += span_2;
          s_2 = 0;
          ++i_2;
        } else {
          auto lhs = slice(*i_1, s_1, span_1, 1);
          auto rhs = slice(*i_2, s_2, span_2, 1);
          inputs.push_back(merge(lhs, rhs, p->fn));
          s_1 = 0;
          s_2 = 0;
          o_1 += span_1;
          o_2 += span_2;
          ++i_1;
          ++i_2;
        }
      }
      CHECK_STATE(i_1 == c_1.inputs.end() && i_2 == c_2.inputs.end());

      return stack(inputs);
    } else {
      auto parent = new_op->to<Apply<Val>>();
      auto c = parent.input->to<Stack<Val>>();

      std::vector<OpPtr<Val>> inputs;
      for (const auto& input : c.inputs) {
        inputs.push_back(apply(input, parent.fn));
      }

      return stack(inputs);
    }
  })(ret);

  // Push all slice operations to the bottom of the tree.
  ret = Fix([](auto fn, const OpPtr<Val>& op) -> OpPtr<Val> {
    // Handle the base case by wrapping the leaf store in a slice.
    if (op->is<Store<Val>>()) {
      return slice(op, 0, op->span(), 1);
    }

    // If the op is a slice, we push it down one level in the tree.
    if (auto p = op->as<Slice<Val>>()) {
      if (auto c = p->input->as<Store<Val>>()) {
        return op;
      } else if (auto c = p->input->as<Slice<Val>>()) {
        auto span = c->input->span();
        auto start = std::min(c->start + c->stride * p->start, span);
        auto stop = std::min(c->start + c->stride * p->stop, span);
        auto stride = c->stride * p->stride;
        return fn(slice(c->input, start, stop, stride));
      } else if (auto c = p->input->as<Merge<Val>>()) {
        auto lhs = slice(c->lhs, p->start, p->stop, p->stride);
        auto rhs = slice(c->rhs, p->start, p->stop, p->stride);
        return substitute(merge(lhs, rhs, c->fn), fn);
      } else if (auto c = p->input->as<Apply<Val>>()) {
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
inline auto materialize(const OpPtr<Val>& input) {
  auto normal_form = normalize(input);
  CHECK_STATE(normal_form->is<Stack<Val>>());

  eval::EvalPlan<Val> plan;

  auto offset = 0;
  for (const auto& root : normal_form->to<Stack<Val>>().inputs) {
    struct EvalNode {
      enum { STORE, MERGE, APPLY } tag;
      union {
        size_t index;
        Apply<Val>::Fn apply_fn;
        Merge<Val>::Fn merge_fn;
      };
    };
    struct EvalFunc {
      std::vector<EvalNode> nodes;
      std::unique_ptr<Val[]> stack;
    };

    // Initialize the state of the eval fn.
    constexpr auto kCacheLineSize = std::hardware_destructive_interference_size;
    auto ef = std::make_shared<EvalFunc>();
    ef->stack.reset(new Val[depth(root), kCacheLineSize]);

    // Initialize the step.
    eval::EvalStep<Val> step(offset, offset + root->span(), [ef](Val* inputs) {
      auto sp = &ef->stack[0];
      for (auto cn : ef->nodes) {
        switch (cn.tag) {
          case EvalNode::STORE:
            *sp++ = inputs[cn.index];
            break;
          case EvalNode::MERGE:
            *(sp - 2) = cn.merge_fn(*(sp - 2), *(sp - 1));
            --sp;
            break;
          case EvalNode::APPLY:
            *(sp - 1) = cn.apply_fn(*(sp - 1));
            break;
        }
      }
      return *(sp - 1);
    });

    // Traverse the op graph to populate the eval step.
    Fix([&](auto fn, const OpPtr<Val>& op) -> void {
      recurse(op, fn);
      if (auto p = op->as<Slice<Val>>()) {
        const auto& store = p->input->to<Store<Val>>();
        step.sources.emplace_back(store.store, p->start, p->stop, p->stride);
        ef->nodes.emplace_back();
        ef->nodes.back().tag = EvalNode::STORE;
        ef->nodes.back().index = step.sources.size() - 1;
      } else if (auto p = op->as<Merge<Val>>()) {
        ef->nodes.emplace_back();
        ef->nodes.back().tag = EvalNode::MERGE;
        ef->nodes.back().merge_fn = p->fn;
      } else if (auto p = op->as<Apply<Val>>()) {
        ef->nodes.emplace_back();
        ef->nodes.back().tag = EvalNode::APPLY;
        ef->nodes.back().apply_fn = p->fn;
      }
    })(root);

    // Add the fully populated step to the plan and loop.
    plan.steps.push_back(std::move(step));
    offset += root->span();
  }

  return eval::eval_plan(plan);
}

}  // namespace skimpy::detail::lang
