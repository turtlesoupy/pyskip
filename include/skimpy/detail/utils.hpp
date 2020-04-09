#pragma once

#include <optional>

#include "errors.hpp"

namespace skimpy::detail {

inline constexpr bool is_power_of_two(uint32_t x) {
  return (x & (x - 1)) == 0;
}

inline constexpr uint32_t round_up_to_power_of_two(uint32_t x) {
  // NOTE: Zero is mapped to zero.
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x;
}

inline constexpr uint32_t lg2(uint32_t x) {
  return x < 2 ? 0 : 1 + lg2(x >> 1);
}

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
