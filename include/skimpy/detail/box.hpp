#pragma once

#include <cstdint>
#include <type_traits>

#include "core.hpp"

namespace skimpy::detail::box {

// We store data in boxes between  merge function calls to unify the value type.
class Box {
 public:
  Box() = default;

  template <typename T>
  Box(T t) {
    put(t);
  }

  template <typename T>
  constexpr inline const auto& ref() const {
    if constexpr (std::is_same<T, Box>()) {
      return *this;
    } else if constexpr (std::is_same<T, char>()) {
      return i8;
    } else if constexpr (std::is_same<T, int8_t>()) {
      return i8;
    } else if constexpr (std::is_same<T, short>()) {
      return i16;
    } else if constexpr (std::is_same<T, int16_t>()) {
      return i16;
    } else if constexpr (std::is_same<T, int>()) {
      return i32;
    } else if constexpr (std::is_same<T, int32_t>()) {
      return i32;
    } else if constexpr (std::is_same<T, bool>()) {
      return i32;
    } else if constexpr (std::is_same<T, unsigned char>()) {
      return u8;
    } else if constexpr (std::is_same<T, uint8_t>()) {
      return u8;
    } else if constexpr (std::is_same<T, unsigned short>()) {
      return u16;
    } else if constexpr (std::is_same<T, uint16_t>()) {
      return u16;
    } else if constexpr (std::is_same<T, unsigned int>()) {
      return u32;
    } else if constexpr (std::is_same<T, uint32_t>()) {
      return u32;
    } else if constexpr (std::is_same<T, float>()) {
      return f32;
    }
  }

  template <typename T>
  constexpr inline auto& ref() {
    auto const_this = const_cast<const Box*>(this);
    using R = std::decay_t<decltype(const_this->ref<T>())>;
    return const_cast<R&>(const_this->ref<T>());
  }

  auto clear() {
    ref<uint32_t>() = 0;
  }

  template <typename T>
  auto get() const {
    return static_cast<T>(ref<T>());
  }

  template <typename T>
  auto& put(T v) {
    clear();
    using A = std::decay_t<T>;
    using B = std::decay_t<decltype(ref<std::decay_t<T>>())>;
    return ref<A>() = static_cast<B>(v);
  }

  template <typename T>
  auto& operator=(T v) {
    return put<T>(v);
  }

  bool operator==(const Box& other) {
    return get<uint32_t>() == other.get<uint32_t>();
  }

  bool operator!=(const Box& other) {
    return get<uint32_t>() != other.get<uint32_t>();
  }

 private:
  union {
    int_fast8_t i8;
    int_fast16_t i16;
    int_fast32_t i32;
    uint_fast8_t u8;
    uint_fast16_t u16;
    uint_fast32_t u32;
    float f32;
  };
};

using BoxStore = core::Store<box::Box>;

template <typename Val>
inline auto box_store(const core::Store<Val>& store) {
  core::Store<box::Box> ret(store.size);
  for (int i = 0; i < store.size; i += 1) {
    ret.ends[i] = store.ends[i];
    ret.vals[i] = store.vals[i];
  }
  return ret;
}

}  // namespace skimpy::detail::box
