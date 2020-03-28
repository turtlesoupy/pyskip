#pragma once

#include <optional>

#include "errors.hpp"

namespace skimpy::detail {

template <typename Val, typename Fn>
class Generator {
 public:
  explicit Generator(Fn&& fn) : fn_(std::forward<Fn>(fn)) {
    val_ = fn_();
  }
  bool done() const {
    return !val_;
  }
  const Val& get() const {
    CHECK_STATE(!done());
    return *val_;
  }
  Val next() {
    CHECK_STATE(!done());
    auto ret = std::move(*val_);
    val_ = fn_();
    return ret;
  }

 private:
  Fn fn_;
  std::optional<Val> val_;
};

template <typename Val, typename Fn>
class InfGenerator {
 public:
  explicit InfGenerator(Fn&& fn) : fn_(std::forward<Fn>(fn)) {
    val_ = fn_();
  }
  const Val& get() const {
    return val_;
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

template <typename Val, typename Fn>
inline auto make_inf_generator(Fn&& fn) {
  return InfGenerator<Val, Fn>(std::forward<Fn>(fn));
}

template <typename T>
inline auto make_array_ptr(std::initializer_list<T> vals) {
  std::unique_ptr<T[]> ret(new T[vals.size()]);
  std::move(vals.begin(), vals.end(), ret.get());
  return ret;
}

}  // namespace skimpy::detail
