#pragma once

#include <array>
#include <functional>
#include <skimpy/skimpy.hpp>
#include <tuple>

namespace skimpy_3d {

// Enables lambda-based static visitor pattern for std::variant
template <class... Ts>
struct Overloaded : Ts... {
  using Ts::operator()...;
};

// explicit deduction guide for Overloaded visitor pattern
template <class... Ts>
explicit Overloaded(Ts...)->Overloaded<Ts...>;

// Basic geometric types.
using Vec3i = std::array<int, 3>;
using Vec3f = std::array<float, 3>;

template <typename Vec3>
inline auto add(Vec3 u, Vec3 v) {
  return Vec3{u[0] + v[0], u[1] + v[1], u[2] + v[2]};
}

template <typename S, typename Vec3>
inline auto scale(S a, Vec3 u) {
  return Vec3{a * u[0], a * u[1], a * u[2]};
}

// Masks for each cube face direction
enum DirMask {
  X_NEG = 0b000001,
  X_POS = 0b000010,
  Y_NEG = 0b000100,
  Y_POS = 0b001000,
  Z_NEG = 0b010000,
  Z_POS = 0b100000,
};

// Integer log base 2
inline constexpr uint32_t lg2(uint32_t x) {
  return x < 2 ? 0 : 1 + lg2(x >> 1);
}

// Generic routine for applying a function to each component of a tuple.
template <typename... Args, typename Fn, std::size_t... Idx>
inline constexpr void for_each(
    const std::tuple<Args...>& t, Fn&& f, std::index_sequence<Idx...>) {
  (f(std::get<Idx>(t)), ...);
}

// Provides efficient traversal over tensor cells via a boolean mask.
template <typename Mask, typename... Arrays, typename Fn>
inline void array_walk(Fn&& fn, Mask&& mask, Arrays&&... a) {
  CHECK_ARGUMENT(((mask.len() == a.len()) && ...));

  auto store = mask.store();
  auto iters = std::make_tuple(std::tuple(0, a.store())...);

  int pos = 0;
  auto advance = [&](auto& it) {
    auto& index = std::get<0>(it);
    auto& store = std::get<1>(it);
    while (store->ends[index] <= pos) {
      ++index;
    }
    return store->vals[index];
  };

  for (int i = 0; i < store->size; i += 1) {
    auto end = store->ends[i];
    auto val = store->vals[i];
    if (!val) {
      pos = end;
      continue;
    }
    for (; pos < end; ++pos) {
      // Advance tensor iterators.
      auto vals = std::apply(
          [&](auto&... it) { return std::tuple(advance(it)...); }, iters);

      // Invoke the traversal fn.
      std::apply(fn, std::tuple_cat(std::tuple(pos), vals));
    }
  }
}

auto pack_bytes(uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  uint32_t ret = static_cast<uint32_t>(d);
  ret |= static_cast<uint32_t>(c) << 8;
  ret |= static_cast<uint32_t>(b) << 16;
  ret |= static_cast<uint32_t>(a) << 24;
  return ret;
}

auto unpack_bytes(uint32_t bytes) {
  uint8_t a = (bytes & 0xFF000000) >> 24;
  uint8_t b = (bytes & 0x00FF0000) >> 16;
  uint8_t c = (bytes & 0x0000FF00) >> 8;
  uint8_t d = (bytes & 0x000000FF);
  return std::tuple(a, b, c, d);
}

}  // namespace skimpy_3d
