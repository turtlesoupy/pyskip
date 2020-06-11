#pragma once

#define UNARY_ARRAY_OP(op, out, impl)       \
  template <typename Val>                   \
  Array<out> op(const Array<Val>& array) {  \
    constexpr out (*fn)(Val) = (impl);      \
    return array.template merge<out, fn>(); \
  }

#define BINARY_ARRAY_OP(op, in1, in2, out, impl)                \
  template <typename Val>                                       \
  Array<out> op(const Array<in1>& lhs, const Array<in2>& rhs) { \
    constexpr out (*fn)(in1, in2) = (impl);                     \
    return lhs.template merge<out, in2, fn>(rhs);               \
  }                                                             \
  template <typename Val>                                       \
  Array<out> op(const Array<in1>& lhs, in2 rhs) {               \
    return op(lhs, make_array<in2>(lhs.len(), rhs));            \
  }                                                             \
  template <typename Val>                                       \
  Array<out> op(in1 lhs, const Array<in2>& rhs) {               \
    return op(make_array<in1>(rhs.len(), lhs), rhs);            \
  }

#define TERNARY_ARRAY_OP(op, in1, in2, in3, out, impl)                     \
  template <typename Val>                                                  \
  Array<out> op(                                                           \
      const Array<in1>& a1, const Array<in2>& a2, const Array<in3>& a3) {  \
    constexpr out (*fn)(in1, in2, in3) = (impl);                           \
    return a1.template merge<out, in2, in3, fn>(a2, a3);                   \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(const Array<in1>& a1, const Array<in3>& a2, in3 a3) {      \
    return op(a1, a2, make_array<in3>(a2.len(), a3));                      \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(const Array<in1>& a1, in2 a2, const Array<in3>& a3) {      \
    return op(a1, make_array<in2>(a1.len(), a2), a3);                      \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(in1 a1, const Array<in2>& a2, const Array<in3>& a3) {      \
    return op(make_array<in1>(a3.len(), a1), a2, a3);                      \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(const Array<in1>& a1, in2 a2, in3 a3) {                    \
    return op(                                                             \
        a1, make_array<in2>(a1.len(), a2), make_array<in3>(a1.len(), a3)); \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(in1 a1, const Array<in2>& a2, in3 a3) {                    \
    return op(                                                             \
        make_array<in1>(a2.len(), a1), a2, make_array<in3>(a2.len(), a3)); \
  }                                                                        \
  template <typename Val>                                                  \
  Array<out> op(in1 a1, in2 a2, const Array<in3>& a3) {                    \
    return op(                                                             \
        make_array<in1>(a3.len(), a1), make_array<in2>(a3.len(), a2), a3); \
  }

#define UNARY_ARRAY_OP_SIMPLE(op, impl) UNARY_ARRAY_OP(op, Val, impl)
#define BINARY_ARRAY_OP_SIMPLE(op, impl) \
  BINARY_ARRAY_OP(op, Val, Val, Val, impl)
#define TERNNARY_ARRAY_OP_SIMPLE(op, impl) \
  TERNARY_ARRAY_OP(op, Val, Val, Val, Val, impl)
