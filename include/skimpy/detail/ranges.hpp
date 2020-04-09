
#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "errors.hpp"
#include "utils.hpp"

namespace skimpy::detail::ranges {

template <typename Val>
using Range = std::tuple<int, int, Val>;

template <typename Val>
class Op;

// Represents contiguous sequence of ranges, each with an associated value.
template <typename Val>
class Store {
 public:
  Store(int size, std::unique_ptr<int[]> ends, std::unique_ptr<Val[]> vals)
      : size_(size), ends_(std::move(ends)), vals_(std::move(vals)) {
    CHECK_ARGUMENT(size > 0);
  }

  auto size() const {
    return size_;
  }

  auto scan(int pos) {
    CHECK_ARGUMENT(0 <= pos && pos < ends[size - 1]);
    auto iter = std::upper_bound(ends_.get(), ends_.get() + size_, pos);
    auto rank = iter - ends_.get();
    return make_generator<Range<Val>>([&, rank] -> std::optional<Range<Val>> {
      if (rank < size) {
        Range<Val> ret(rank == 0 ? 0 : ends[rank - 1], ends[rank], vals[rank]);
        rank += 1;
        return ret;
      } else {
        return {};
      }
    });
  }

 private:
  int size_;
  std::unique_ptr<int[]> ends_;
  std::unique_ptr<Val[]> vals_;
};

template <typename Val>
class Op {
 public:
  virtual ~Op() = default;
  virtual size_t size() const = 0;
  virtual std::shared_ptr<Chain<Val>> eval() const = 0;
  virtual void emit(std::shared_ptr<Store<Val>> store) const = 0;
};

template <typename Val>
using OpPtr = std::shared_ptr<Op<Val>>;

template <typename Val>
class StoreOp : public Op<Val> {
 public:
  StoreOp(std::shared_ptr<Store<Val>> store) : store_(std::move(store)) {}

  size_t size() const override {
    return store_->size();
  }

  std::shared_ptr<Chain<Val>> eval() const override {
    return nullptr;
  }

  void emit(std::shared_ptr<Store<Val>> store) const override {}

 private:
  std::shared_ptr<Store<Val>> store_;
};

template <typename Val>
class SliceOp : public Op<Val> {
 public:
  SliceOp(std::shared_ptr<Op<Val>> input, int start, int end)
      : input_(std::move(input)), start_(start), end_(end) {}

  size_t size() const override {
    return end_ - start_;
  }

  std::shared_ptr<Chain<Val>> eval() const override {
    return nullptr;
  }

  void emit(std::shared_ptr<Store<Val>> store) const override {}

 private:
  std::shared_ptr<Op<Val>> input_;
  int start_;
  int end_;
};

template <typename Val>
class StackOp : public Op<Val> {
 public:
  StackOp(std::shared_ptr<Op<Val>> head, std::shared_ptr<Op<Val>> tail)
      : head_(std::move(head)), tail_(std::move(tail)) {}

  size_t size() const override {
    return head_->size() + tail_->size();
  }

  std::shared_ptr<Chain<Val>> eval() const override {
    return nullptr;
  }

  void emit(std::shared_ptr<Store<Val>> store) const override {}

 private:
  std::shared_ptr<Op<Val>> head_;
  std::shared_ptr<Op<Val>> tail_;
};

template <typename Val, typename Fun>
class MergeOp : public Op<Val> {
 public:
  MergeOp(
      std::shared_ptr<Op<Val>> lhs, std::shared_ptr<Op<Val>> rhs, Fun&& func)
      : head_(std::move(head)),
        tail_(std::move(tail)),
        func_(std::forward<Fun>(func)) {
    CHECK_ARGUMENT(head_->size() == tail_->size());
  }

  size_t size() const override {
    return head_->size();
  }

  std::shared_ptr<Chain<Val>> eval() const override {
    return nullptr;
  }

  void emit(std::shared_ptr<Store<Val>> store) const override {}

 private:
  std::shared_ptr<Op<Val>> lhs_;
  std::shared_ptr<Op<Val>> rhs_;
  Fun func_;
};

template <typename Val, typename Fun>
class ApplyOp : public Op<Val> {
 public:
  ApplyOp(std::shared_ptr<Op<Val>> input, Fun&& func)
      : input_(std::move(input)), func_(std::forward<Fun>(func)) {}

  size_t size() const override {
    return head_->size();
  }

  std::shared_ptr<Chain<Val>> eval() const override {
    return nullptr;
  }

  void emit(std::shared_ptr<Store<Val>> store) const override {}

 private:
  std::shared_ptr<Op<Val>> lhs_;
  std::shared_ptr<Op<Val>> rhs_;
  Fun func_;
};

template <typename Val>
inline OpPtr<Val> store(int size, Val fill) {
  std::unique_ptr<int[]> ends(new int[1]);
  std::unique_ptr<Val[]> vals(new Val[1]);
  ends[0] = size;
  vals[0] = fill;
  return std::make_shared<StoreOp<Val>>(
      std::make_shared<Store<Val>>(size, std::move(ends), std::move(vals)));
}

template <typename Val>
inline OpPtr<Val> slice(OpPtr<Val> input, int start, int end) {
  return std::make_shared<SliceOp<Val>>(std::move(input), start, end);
}

template <typename Val>
inline OpPtr<Val> stack(OpPtr<Val> head, OpPtr<Val> tail) {
  return std::make_shared<StackOp<Val>>(std::move(head), std::move(tail));
}

template <typename Val, typename Fun>
inline OpPtr<Val> merge(OpPtr<Val> lhs, OpPtr<Val> rhs, Fun&& func) {
  return std::make_shared<MergeOp<Val>>(
      std::move(lhs), std::move(rhs), std::forward<Fun>(func));
}

template <typename Val, typename Fun>
inline OpPtr<Val> apply(OpPtr<Val> input, Fun&& func) {
  return std::make_shared<ApplyOp<Val>>(
      std::move(input), std::forward<Fun>(func));
}

template <typename Val>
inline auto emit(std::shared_ptr<Op<Val>> input) {
  // Evaluate the operation into a chain to compute the output size.
  auto chain = input->eval();

  // Allocate the output store and emit ranges to it.
  std::shared_ptr<Store<Val>> ret;
  input->emit(ret);

  // Compress contiguous same-valued ranges in the output store.

  return ret;
}

template <typename Val>
inline auto to_vector(std::shared_ptr<Op<Val>> input) {
  std::vector<Range<Val>> ret;
  return ret;
}

template <typename Val>
inline auto to_string(std::shared_ptr<Op<Val>> input) {
  std::string ret;
  return ret;
}

}  // namespace skimpy::detail::ranges
