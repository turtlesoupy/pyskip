import numpy as np
import re

import _skimpy_cpp_ext
import skimpy.util as util
from .exceptions import (
    InvalidTensorError,
    IncompatibleTensorError,
    UnimplementedOperationError,
    TypeConversionError,
)

_type_mapping = {int: "i", float: "f"}
_type_mapping_reverse = {v: k for k, v in _type_mapping.items()}


class Tensor:
  @classmethod
  def wrap(cls, cpp_tensor):
    return cls(cpp_tensor = cpp_tensor)

  @classmethod
  def from_numpy(cls, np_arr):
    np_shape = np_arr.shape
    skimpy_arr = _skimpy_cpp_ext.from_numpy(np_arr.flatten("C"))
    return cls(shape = tuple(reversed(np_shape)), val = skimpy_arr)

  def __init__(self, shape = None, val = 0, dtype = int, cpp_tensor = None):
    if cpp_tensor:
      self._init_from_cpp_tensor(cpp_tensor)
      return

    if shape is None:
      raise InvalidTensorError("Must specify shape")

    if isinstance(shape, int):
      shape = (shape, )

    dimensionality = len(shape)
    if dimensionality < 1:
      raise InvalidTensorError("Dimensionality must be >= 1")
    elif dimensionality > 3:
      raise InvalidTensorError("Only 1D, 2D and 3D tensors are supported")

    if dtype not in _type_mapping:
      raise InvalidTensorError("Only int32 tensors are supported")

    klass = f"Tensor{dimensionality}{_type_mapping[dtype]}"
    tensor = getattr(_skimpy_cpp_ext, klass)(shape, val)

    self._init_from_cpp_tensor(tensor)

  def _init_from_cpp_tensor(self, cpp_tensor):
    self._tensor = cpp_tensor

    m = re.search(r"Tensor(\d)(\w+)$", self._tensor.__class__.__name__)
    self.dimensionality = int(m.group(1))
    self.dtype = _type_mapping_reverse[m.group(2)]
    self.shape = self._tensor.shape()
    return self

  @classmethod
  def _validate_or_cast(cls, a, b):
    if a.shape != b.shape and b.shape != (1, ):
      raise IncompatibleTensorError(
          f"Incompatible shapes: {a.shape} and {b.shape}"
      )

    return (a, b)

  # Binary operators
  @classmethod
  def _forward_to_binary_array_op(cls, a, b, op):
    if isinstance(b, Tensor):
      a, b = Tensor._validate_or_cast(a, b)
      return Tensor.wrap(
          a._tensor.__class__(
              a.shape,
              getattr(a._tensor.array(), op)(b._tensor.array())
          )
      )
    else:
      return Tensor.wrap(
          a._tensor.__class__(a.shape,
                              getattr(a._tensor.array(), op)(b))
      )

  def __add__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__add__")

  def __radd__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__radd__")

  def __iadd__(self, other):
    return self._init_from_cpp_tensor(self.__add__(other)._tensor)

  def __sub__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__sub__")

  def __rsub__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rsub__")

  def __isub__(self, other):
    return self._init_from_cpp_tensor(self.__sub__(other)._tensor)

  def __mul__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__mul__")

  def __rmul__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rmul__")

  def __imul__(self, other):
    return self._init_from_cpp_tensor(self.__mul__(other)._tensor)

  def __truediv__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__truediv__")

  def __rtruediv__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rtruediv__")

  def __idiv__(self, other):
    return self._init_from_cpp_tensor(self.__truediv__(other)._tensor)

  def __floordiv__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__floordiv__")

  def __rfloordiv__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rfloordiv__")

  def __ifloordiv__(self, other):
    return self._init_from_cpp_tensor(self.__floordiv__(other)._tensor)

  def __mod__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__mod__")

  def __rmod__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rmod__")

  def __imod__(self, other):
    return self._init_from_cpp_tensor(self.__mod__(other)._tensor)

  def __and__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__and__")

  def __rand__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rand__")

  def __iand__(self, other):
    return self._init_from_cpp_tensor(self.__and__(other)._tensor)

  def __xor__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__xor__")

  def __rxor__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rxor__")

  def __ixor__(self, other):
    return self._init_from_cpp_tensor(self.__xor__(other)._tensor)

  def __or__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__or__")

  def __ror__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__ror__")

  def __ior__(self, other):
    return self._init_from_cpp_tensor(self.__or__(other)._tensor)

  def __pow__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__pow__")

  def __rpow__(self, other):
    return Tensor._forward_to_binary_array_op(self, other, "__rpow__")

  def __ipow__(self, other):
    return self._init_from_cpp_tensor(self.__pow__(other)._tensor)

  def __setitem__(self, slices, value):
    slices = util.unify_slices(slices)
    if isinstance(value, Tensor):
      self._tensor[slices] = value._tensor
    else:
      self._tensor[slices] = value

  def __getitem__(self, slices):
    slices = util.unify_slices(slices)
    ret = self._tensor[slices]
    if isinstance(ret, self.dtype):
      return ret
    else:
      return Tensor.wrap(ret)

  # Unary Operators
  def _forward_to_unary_array_op(self, op):
    return Tensor.wrap(
        self._tensor.__class__(self.shape,
                               getattr(self._tensor.array(), op)())
    )

  def __neg__(self):
    return self._forward_to_unary_array_op("__neg__")

  def __pos__(self):
    return self._forward_to_unary_array_op("__pos__")

  def __abs__(self):
    return self._forward_to_unary_array_op("__abs__")

  def __invert__(self):
    return self._forward_to_unary_array_op("__invert__")

  def __len__(self):
    return len(self._tensor)

  def __str__(self):
    return self.to_string()

  def __repr__(self):
    type_str = f"Tensor(shape={self.shape}, dtype={self.dtype.__name__})"
    vals_str = self.to_string(separator = ", ")
    indented = f"\n{vals_str}".replace("\n", "\n    ")
    return f"{type_str}:{indented}"

  def to(self, dtype):
    if isinstance(dtype, int):
      return self._forward_to_unary_array_op("int")
    elif isinstance(dtype, float):
      return self._forward_to_unary_array_op("float")
    elif isinstance(dtype, bool):
      return self._forward_to_unary_array_op("bool")
    else:
      raise TypeConversionError(f"No conversion to dtype='{dtype}' exists.")

  def to_numpy(self):
    np_arr = self._tensor.array().to_numpy()
    return np_arr.reshape(tuple(reversed(self.shape)))

  def to_string(self, threshold = 20, separator = " "):
    truncated = self
    for i, l in enumerate(self.shape):
      if l > threshold:
        head = util.take(truncated, slice(threshold), axis = i)
        tail = util.take(truncated, slice(-threshold, None), axis = i)
        truncated = util.stack(head, tail)
    return np.array2string(
        truncated.to_numpy(), threshold = threshold, separator = separator
    )

  def eval(self):
    return self._forward_to_unary_array_op("eval")
