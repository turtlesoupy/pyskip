#pragma once

#include "core.hpp"
#include "util.hpp"

namespace skimpy::detail::step {

using Pos = core::Pos;

using SimpleStepFn = Pos (*)(Pos);

// Provides parameterization of monotonic integer functions that correspond
// naturally to ways of stepping over sequential data. In particular, these
// parameterizations can be used to represent n-dimensional array slicing.
class StepFn {
 public:
  StepFn() : hash_(0), fn_([](Pos p) { return p; }) {}

  StepFn(Pos run, Pos skip, Pos jump, Pos lead)
      : StepFn(StepFn(), run, skip, jump, lead) {}

  StepFn(StepFn input, Pos run, Pos skip, Pos jump, Pos lead)
      : hash_(hash_combine(input.hash_, run, skip, jump, lead)) {
    CHECK_ARGUMENT(run > 0);
    CHECK_ARGUMENT(skip >= 0);
    CHECK_ARGUMENT(jump >= 0);

    // Initialize arguments.
    args_.swap(input.args_);
    args_.emplace_back(run, skip, jump, lead);
  }

  StepFn compose(StepFn input) const {
    StepFn ret = std::move(input);
    for (const auto& args : args_) {
      ret = StepFn(ret, args.run, args.skip, args.jump, args.lead);
    }
    return ret;
  }

  Pos operator()(Pos pos) const {
    if (!fn_) {
      init_fn();
    }
    return fn_(pos);
  }

  size_t hash() const {
    return hash_;
  }

  bool operator==(const StepFn& other) const {
    if (args_.size() != other.args_.size()) {
      return false;
    }
    for (int i = 0; i < args_.size(); i += 1) {
      const auto& a = args_[i];
      const auto& b = other.args_[i];
      if (a.run != b.run) {
        return false;
      }
      if (a.skip != b.skip) {
        return false;
      }
      if (a.jump != b.jump) {
        return false;
      }
      if (a.lead != b.lead) {
        return false;
      }
    }
    return true;
  }

 private:
  void init_fn() const {
    // Initialize the component functions.
    std::vector<std::function<Pos(Pos)>> fns;
    for (const auto& args : args_) {
      auto ru = args.run;
      auto sk = args.skip;
      auto ju = args.jump;
      auto le = args.lead;
      auto wi = ru + sk;
      if (sk == 0 && ju == 0 && le == 0) {
        fns.emplace_back([](Pos pos) { return pos; });
      } else if (sk == 0 && ju == 0) {
        fns.emplace_back([le](Pos pos) { return pos + le; });
      } else if (is_power_of_two(wi)) {
        auto sh = lg2(wi);
        fns.emplace_back([ru, wi, sh, ju, le](Pos pos) {
          auto q = (le + pos - 1) >> sh;
          auto r = (le + pos - 1) & (wi - 1);
          return 1 + std::min<Pos>(ru - 1, r) + (ru + ju) * q;
        });
      } else {
        fns.emplace_back([ru, wi, ju, le](Pos pos) {
          auto d = std::div(le + pos - 1, wi);
          return 1 + std::min<Pos>(ru - 1, d.rem) + (ru + ju) * d.quot;
        });
      }
    }

    // Initialize the composite function.
    // TODO: Generate the entire composite function using asmjit.
    fn_ = [fns = std::move(fns)](Pos pos) {
      for (const auto& fn : fns) {
        pos = fn(pos);
      }
      return pos;
    };
  }

  struct Args {
    Pos run;
    Pos skip;
    Pos jump;
    Pos lead;

    Args(Pos run, Pos skip, Pos jump, Pos lead)
        : run(run), skip(skip), jump(jump), lead(lead) {}
  };

  size_t hash_;
  std::vector<Args> args_;
  mutable std::function<Pos(Pos)> fn_;
};

auto identity() {
  return [](Pos p) { return p; };
}

StepFn step_fn(Pos run = 1, Pos skip = 0, Pos jump = 0, Pos lead = 0) {
  return StepFn(run, skip, jump, lead);
}

StepFn stride_fn(Pos stride) {
  CHECK_ARGUMENT(stride > 0);
  return step_fn(1, stride - 1);
}

StepFn stride_fn(Pos start, Pos stride) {
  CHECK_ARGUMENT(stride > 0);
  return step_fn(1, stride - 1, 0, -start);
}

StepFn compose(const StepFn& f, StepFn g) {
  return f.compose(std::move(g));
}

StepFn shift(StepFn f, Pos p) {
  return StepFn(std::move(f), 1, 0, 0, p);
}

template <typename Fn>
Pos span(Pos start, Pos stop, Fn&& fn) {
  return fn(stop) - fn(start);
}

template <typename Fn>
Pos invert(Pos pos, Pos start, Pos stop, Fn&& step) {
  CHECK_ARGUMENT(start <= stop);
  auto l = start;
  auto h = stop;
  while (l < h) {
    auto m = (l + h) >> 1;
    if (step(m) < pos) {
      l = m + 1;
    } else {
      h = m;
    }
  }
  return l;
}

}  // namespace skimpy::detail::step
