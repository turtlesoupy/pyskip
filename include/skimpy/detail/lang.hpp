
#pragma once

#include <fmt/format.h>

#include <algorithm>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

#include "core.hpp"
#include "errors.hpp"
#include "util.hpp"

namespace skimpy::detail::lang {

template <typename Val>
class Op;

template <typename Val>
class Op {
 public:
  virtual ~Op() = default;
  virtual size_t size() const = 0;
};

template <typename Val>
using OpPtr = std::shared_ptr<Op<Val>>;

template <typename Val>
struct Store {};

template <typename Val>
class StoreOp : public Op<Val> {
 public:
  StoreOp(std::shared_ptr<Store<Val>> store) : store_(std::move(store)) {}

  size_t size() const override {
    return store_->size();
  }

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

 private:
  std::shared_ptr<Op<Val>> lhs_;
  std::shared_ptr<Op<Val>> rhs_;
  Fun func_;
};

template <typename Val>
inline OpPtr<Val> store(int size, Val fill) {
  return std::make_shared<StoreOp<Val>>(std::make_shared<Store<Val>>());
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
inline auto eval(std::shared_ptr<Op<Val>> input) {
  std::shared_ptr<Store<Val>> ret;
  return ret;
}

}  // namespace skimpy::detail::lang
