#pragma once

#include <fmt/core.h>

#include <stdexcept>

#define CHECK_STATE(cond)                                                     \
  do {                                                                        \
    if (!(cond)) {                                                            \
      throw std::logic_error(                                                 \
          fmt::format("Failed \"{}\" at {}:{}.", #cond, __FILE__, __LINE__)); \
    }                                                                         \
  } while (0)

#define CHECK_ARGUMENT(cond)                                                  \
  do {                                                                        \
    if (!(cond)) {                                                            \
      throw std::invalid_argument(                                            \
          fmt::format("Failed \"{}\" at {}:{}.", #cond, __FILE__, __LINE__)); \
    }                                                                         \
  } while (0)

#define CHECK_UNREACHABLE(msg)                       \
  do {                                               \
    fmt::print(                                      \
        "Unreachable control flow \"{}\" at {}:{}.", \
        #msg,                                        \
        __FILE__,                                    \
        __LINE__);                                   \
    std::terminate();                                \
  } while (0)
