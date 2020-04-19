#pragma once

#include <functional>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

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

template <typename Range, typename Fn>
auto map(Range&& r, Fn&& fn) {
  std::vector<decltype(fn(*r.begin()))> ret;
  for (const auto& x : r) {
    ret.push_back(fn(x));
  }
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

template <typename T>
class Deferred {
  static_assert(!std::is_void_v<T>, "Deferred cannot return void type.");

 public:
  Deferred() : state_(nullptr) {}
  Deferred(T val) : state_(std::make_shared<State>(std::move(val))) {}
  Deferred(std::function<T()> fn)
      : state_(std::make_shared<State>(std::move(fn))) {}

  auto get() const {
    CHECK_STATE(state_);
    if (!state_->val) {
      CHECK_STATE(state_->fn);
      state_->val = state_->fn();
    }
    return state_->val.value();
  }

  template <typename Fn>
  auto then(Fn&& fn) const& {
    return Deferred<decltype(fn(get()))>(
        [this_deferred = *this, fn = std::forward<Fn>(fn)] {
          return fn(this_deferred.get());
        });
  }

  template <typename Fn>
  auto then(Fn&& fn) && {
    return Deferred<decltype(fn(get()))>(
        [this_deferred = std::move(*this), fn = std::forward<Fn>(fn)] {
          return fn(this_deferred.get());
        });
  }

 private:
  struct State {
    std::function<T()> fn;
    std::optional<T> val;

    State(std::function<T()> fn) : fn(std::move(fn)) {}
    State(T val) : fn(nullptr), val(std::move(val)) {}
  };

  std::shared_ptr<State> state_;
};

template <>
class Deferred<void> {
 public:
  Deferred() : state_(nullptr) {}
  Deferred(std::function<void()> fn)
      : state_(std::make_shared<State>(std::move(fn))) {}

  void get() const {
    CHECK_STATE(state_);
    if (!state_->set) {
      state_->fn();
      state_->set = true;
    }
  }

  template <typename Fn>
  auto then(Fn&& fn) const& {
    return Deferred<decltype(fn())>(
        [this_deferred = *this, fn = std::forward<Fn>(fn)] {
          this_deferred.get();
          return fn();
        });
  }

 private:
  struct State {
    bool set;
    std::function<void()> fn;

    State(std::function<void()> fn) : set(false), fn(std::move(fn)) {}
  };

  std::shared_ptr<State> state_;
};

template <typename Fn>
inline auto make_deferred(Fn&& fn) {
  return Deferred<decltype(fn())>(std::forward<Fn>(fn));
};

template <typename T>
auto chain(std::vector<Deferred<T>> inputs) {
  return Deferred<std::vector<T>>([inputs = std::move(inputs)] {
    std::vector<T> ret;
    ret.reserve(inputs.size());
    for (const auto& input : inputs) {
      ret.push_back(input.get());
    }
    return ret;
  });
}

template <>
auto chain(std::vector<Deferred<void>> inputs) {
  return Deferred<void>([inputs = std::move(inputs)] {
    for (const auto& input : inputs) {
      input.get();
    }
  });
}

template <
    typename... T,
    typename = std::enable_if_t<(!std::is_same_v<Deferred<void>, T> && ...)>>
auto chain(T&&... inputs) {
  return Deferred<std::tuple<decltype(inputs.get())...>>(
      [inputs = std::make_tuple(std::forward<T>(inputs)...)] {
        return std::apply(
            [](const auto&... args) { return std::make_tuple(args.get()...); },
            inputs);
      });
}

template <typename... T>
auto chain(Deferred<void> d, T&&... inputs) {
  if constexpr (sizeof...(inputs) == 0) {
    return d;
  } else {
    auto tail = chain(std::forward<T>(inputs)...);
    return std::move(d).then([tail] { return tail.get(); });
  }
}

template <typename Head>
auto hash_combine(Head&& head) {
  return std::hash<std::decay_t<Head>>()(std::forward<Head>(head));
}

template <typename Head, typename... Tail>
auto hash_combine(Head&& head, Tail&&... tail) {
  auto head_hash = std::hash<std::decay_t<Head>>()(std::forward<Head>(head));
  auto tail_hash = hash_combine(std::forward<Tail>(tail)...);
  return head_hash ^ (0x9e3779b9 + (tail_hash << 6) + (tail_hash >> 2));
}

template <typename T>
auto hash_combine(const std::vector<T>& v) {
  CHECK_ARGUMENT(v.size());
  auto ret = hash_combine(v[0]);
  for (auto t : v) {
    ret = hash_combine(ret, t);
  }
  return ret;
}

}  // namespace skimpy::detail
