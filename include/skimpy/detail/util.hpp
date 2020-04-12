#pragma once

#include <optional>
#include <type_traits>

#include "errors.hpp"

namespace skimpy::detail {

inline constexpr bool is_power_of_two(uint32_t x) {
  // NOTE: Zero return true.
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

template <typename T>
inline auto make_array_ptr(std::initializer_list<T> vals) {
  std::unique_ptr<T[]> ret(new T[vals.size()]);
  std::move(vals.begin(), vals.end(), ret.get());
  return ret;
}

template <typename Fn>
class Fix {
 public:
  Fix(Fn&& fn) : fn_(std::forward<Fn>(fn)) {}

  template <typename... Args>
  auto operator()(Args&&... args) const -> decltype(
      fn_(std::declval<const Fix<Fn>&>(), std::forward<Args>(args)...)) {
    if constexpr (std::is_same_v<decltype(fn_), void>) {
      fn_(*this, std::forward<Args>(args)...);
    } else {
      return fn_(*this, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  auto operator()(Args&&... args)
      -> decltype(fn_(std::declval<Fix<Fn>&>(), std::forward<Args>(args)...)) {
    if constexpr (std::is_same_v<decltype(fn_), void>) {
      fn_(*this, std::forward<Args>(args)...);
    } else {
      return fn_(*this, std::forward<Args>(args)...);
    }
  }

 private:
  Fn fn_;
};

template <typename Fn>
inline auto make_fix(Fn&& fn) {
  return Fix<Fn>(std::foward<Fn>(fn));
}

}  // namespace skimpy::detail
