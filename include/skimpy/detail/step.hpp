#pragma once

#include "core.hpp"
#include "util.hpp"

namespace skimpy::detail::step {

using StepFn = std::function<core::Pos(core::Pos)>;

StepFn run_skip_fn(core::Pos run, core::Pos skip) {
  // TODO: Jit these functions.
  const auto width = run + skip;
  CHECK_ARGUMENT(run > 0);
  CHECK_ARGUMENT(skip >= 0);
  CHECK_ARGUMENT(width > 0);
  if (skip == 0) {
    return [](core::Pos pos) { return pos; };
  } else if (is_power_of_two(width)) {
    if (run == 1) {
      return [s = lg2(width)](core::Pos pos) { return 1 + (pos - 1 >> s); };
    } else {
      return [run, s = lg2(width), t = width - 1](core::Pos pos) {
        auto q = pos >> s;
        auto r = pos & t;
        return std::min(run, r) + run * q;
      };
    }
  } else {
    if (run == 1) {
      return [width](core::Pos pos) { return 1 + (pos - 1) / width; };
    } else {
      return [run, width](core::Pos pos) {
        auto d = std::div(pos, width);
        return std::min(run, d.rem) + run * d.quot;
      };
    }
  }
}

StepFn stride_fn(core::Pos stride) {
  return run_skip_fn(1, stride - 1);
}

template <typename F, typename G>
auto compose(F&& f, G&& g) {
  return [f = std::forward<F>(f), g = std::forward<G>(g)](core::Pos pos) {
    return g(f(pos));
  }
}

core::Pos span(core::Pos start, core::Pos stop, const StepFn& fn) {
  return fn(stop - start);
}

}  // namespace skimpy::detail::step
