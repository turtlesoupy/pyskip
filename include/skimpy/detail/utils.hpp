#pragma once

#include <optional>

#include "errors.hpp"

namespace skimpy {

template <typename Val, typename Fn>
class Generator {
 public:
  explicit Generator(Fn&& fn) : fn_(std::forward<Fn>(fn)) {
    val_ = fn_();
  }
  bool done() {
    return !val_;
  }
  Val& get() {
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
inline auto make_generator(Fn&& fn) {
  return Generator<Val, Fn>(std::forward<Fn>(fn));
}

}  // namespace skimpy
