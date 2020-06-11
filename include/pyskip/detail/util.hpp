#pragma once

#include <functional>
#include <new>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include "errors.hpp"

namespace pyskip::detail::util {

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

template <typename Fn>
class Fix {
 public:
  explicit Fix(Fn&& fn) : fn_(std::forward<Fn>(fn)) {}

  template <typename... Args>
  auto operator()(Args&&... args) const -> decltype(std::declval<Fn>()(
      std::declval<const Fix<Fn>&>(), std::forward<Args>(args)...)) {
    if constexpr (std::is_same_v<decltype(fn_), void>) {
      fn_(*this, std::forward<Args>(args)...);
    } else {
      return fn_(*this, std::forward<Args>(args)...);
    }
  }

  template <typename... Args>
  auto operator()(Args&&... args) -> decltype(std::declval<Fn>()(
      std::declval<Fix<Fn>&>(), std::forward<Args>(args)...)) {
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
  return Fix<Fn>(std::forward<Fn>(fn));
}

template <typename Head>
inline auto hash_combine(Head&& head) {
  return std::hash<std::decay_t<Head>>()(std::forward<Head>(head));
}

template <typename Head, typename... Tail>
inline auto hash_combine(Head&& head, Tail&&... tail) {
  auto head_hash = std::hash<std::decay_t<Head>>()(std::forward<Head>(head));
  auto tail_hash = hash_combine(std::forward<Tail>(tail)...);
  return head_hash ^ (0x9e3779b9 + (tail_hash << 6) + (tail_hash >> 2));
}

template <typename T>
inline auto hash_combine(const std::vector<T>& v) {
  CHECK_ARGUMENT(v.size());
  auto ret = hash_combine(v[0]);
  for (auto t : v) {
    ret = hash_combine(ret, t);
  }
  return ret;
}

}  // namespace pyskip::detail::util
