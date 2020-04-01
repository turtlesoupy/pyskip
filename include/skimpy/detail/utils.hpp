#pragma once

#include <optional>

#include "errors.hpp"

namespace skimpy::detail {

// Provides forward iteration over an unbounded sequence of elements.
template <typename Val, typename Fn>
class Generator {
 public:
  explicit Generator(Fn&& fn) : fn_(std::forward<Fn>(fn)) {
    val_ = fn_();
  }
  const Val& get() const {
    return val_;
  }
  bool done() const {
    return false;
  }
  Val next() {
    auto ret = std::move(val_);
    val_ = fn_();
    return ret;
  }

 private:
  Fn fn_;
  Val val_;
};

template <typename Val, typename Fn>
inline auto make_generator(Fn&& fn) {
  return Generator<Val, Fn>(std::forward<Fn>(fn));
}

template <typename T>
inline auto make_array_ptr(std::initializer_list<T> vals) {
  std::unique_ptr<T[]> ret(new T[vals.size()]);
  std::move(vals.begin(), vals.end(), ret.get());
  return ret;
}

}  // namespace skimpy::detail
