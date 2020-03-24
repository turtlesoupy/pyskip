#pragma once

#include <stdexcept>

#define CHECK_STATE(cond)            \
  do {                               \
    if (!(cond)) {                   \
      throw std::logic_error(#cond); \
    }                                \
  } while (0)

#define CHECK_ARGUMENT(cond)              \
  do {                                    \
    if (!(cond)) {                        \
      throw std::invalid_argument(#cond); \
    }                                     \
  } while (0)
