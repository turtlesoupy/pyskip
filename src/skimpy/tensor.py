import re
from typing import Iterable, Union, Tuple

import functools
import operator
import numpy as np
import _skimpy_cpp_ext
from .util import unify_slices
from .exceptions import (
    InvalidTensorError,
    IncompatibleTensorError,
    UnimplementedOperationError,
    TypeConversionError,
)

_postfix_type_mapping = {int: "i", float: "f", bool: "b"}
_postfix_type_mapping_reverse = {v: k for k, v in _postfix_type_mapping.items()}
_prefix_type_mapping = {int: "Int", float: "Float", bool: "Bool"}
_prefix_type_mapping_reverse = {v: k for k, v in _prefix_type_mapping.items()}


class TensorBuilder:
    def __init__(self, shape, val, dtype):
        if len(shape) > 3:
            raise UnimplementedOperationError("Builders only support up to three dimensions")

        self.shape = shape
        self.val = val
        self.dtype = dtype

        max_size = functools.reduce(operator.mul, shape)
        self._builder = getattr(_skimpy_cpp_ext, f"{_prefix_type_mapping[dtype]}Builder")(max_size, val)

        self._scale = [1]
        for shape in shape[:-1]:
            self._scale.append(self._scale[-1] * shape)

    def __setitem__(self, items, value):
        # TODO: make this support slice ranges
        if not isinstance(items, Iterable):
            items = (items, )
        elif not all(isinstance(e, int) for e in items):
            raise UnimplementedOperationError("Builder __setitem__ only works with integers for now")

        idx = sum(i * s for i, s in zip(items, self._scale))
        self._builder[idx] = value

    def build(self):
        val = self._builder.build()
        return Tensor(
            shape=self.shape,
            val=val,
            dtype=self.dtype,
        )


# Scalar defines a static type of the possible scalar values composing a Tensor.
Scalar = Union[int, float, bool]


class Tensor:
    @classmethod
    def builder(cls, shape, val=0, dtype=int):
        return TensorBuilder(shape, val, dtype)

    @classmethod
    def wrap(cls, cpp_tensor):
        return cls(cpp_tensor=cpp_tensor)

    @classmethod
    def force_tensor(cls, item, dtype):
        if isinstance(item, Tensor):
            return item
        else:
            return cls(shape=(1,), val=item, dtype=dtype)

    @classmethod
    def from_numpy(cls, np_arr):
        np_shape = np_arr.shape
        skimpy_arr = _skimpy_cpp_ext.from_numpy(np_arr.flatten("C"))
        return cls(shape=tuple(reversed(np_shape)), val=skimpy_arr)

    @classmethod
    def from_list(cls, values, dtype=None):
        return cls.from_numpy(np.array(values, dtype=dtype))

    def __init__(self, shape=None, val=0, dtype=int, cpp_tensor=None):
        if cpp_tensor:
            self._init_from_cpp_tensor(cpp_tensor)
            return

        if shape is None:
            raise InvalidTensorError("Must specify shape")

        # Map scalar shape onto 1-dimensional tensor
        if isinstance(shape, int):
            shape = (shape, )

        ndim = len(shape)
        if ndim < 1:
            raise InvalidTensorError("Dimensionality must be >= 1")
        elif ndim > 3:
            raise InvalidTensorError("Only 1D, 2D and 3D tensors are supported")

        self.dtype = dtype
        tensor = self.__class__._cpp_class(shape, dtype)(shape, val)

        self._init_from_cpp_tensor(tensor)

    def _init_from_cpp_tensor(self, cpp_tensor):
        self._tensor = cpp_tensor

        m = re.search(r"Tensor(\d)(\w+)$", self._tensor.__class__.__name__)
        self.ndim = int(m.group(1))
        self.dtype = _postfix_type_mapping_reverse[m.group(2)]
        self.shape = self._tensor.shape()
        return self

    @classmethod
    def _cpp_class(self, shape, typ):
        name = f"Tensor{len(shape)}{_postfix_type_mapping[typ]}"
        return getattr(_skimpy_cpp_ext, name)

    @classmethod
    def _validate_or_cast(cls, a, b):
        if a.shape != b.shape and b.shape != (1, ):
            raise IncompatibleTensorError(f"Incompatible shapes: {a.shape} and {b.shape}")
        return a, b

    # Unary Operators
    def _forward_to_unary_array_op(self, op):
        return Tensor.wrap(self._tensor.__class__(self.shape, getattr(self._tensor.array(), op)()))

    def __neg__(self):
        return self._forward_to_unary_array_op("__neg__")

    def __pos__(self):
        return self._forward_to_unary_array_op("__pos__")

    def __abs__(self):
        return self._forward_to_unary_array_op("__abs__")

    def __invert__(self):
        return self._forward_to_unary_array_op("__invert__")

    def abs(self):
        return self._forward_to_unary_array_op("abs")

    # Binary operators
    @classmethod
    def _forward_to_binary_array_op(cls, a, b, op, klass=None):
        klass = klass or a._tensor.__class__
        if isinstance(b, Tensor):
            a, b = Tensor._validate_or_cast(a, b)
            return Tensor.wrap(klass(a.shape, getattr(a._tensor.array(), op)(b._tensor.array())))
        else:
            return Tensor.wrap(klass(a.shape, getattr(a._tensor.array(), op)(b)))


    @property
    def _bool_tensor_class(self):
        return self._cpp_class(self.shape, bool)

    def __eq__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__eq__", klass=self._bool_tensor_class)

    def __ne__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__ne__", klass=self._bool_tensor_class)

    def __lt__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__lt__", klass=self._bool_tensor_class)

    def __le__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__le__", klass=self._bool_tensor_class)

    def __gt__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__gt__", klass=self._bool_tensor_class)

    def __ge__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__ge__", klass=self._bool_tensor_class)

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

    def coalesce(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "coalesce")

    def __setitem__(self, slices, value):
        slices = unify_slices(slices)
        if isinstance(value, Tensor):
            self._tensor[slices] = value._tensor
        else:
            self._tensor[slices] = value

    def __getitem__(self, slices):
        slices = unify_slices(slices)
        ret = self._tensor[slices]
        if isinstance(ret, self.dtype):
            return ret
        else:
            return Tensor.wrap(ret)

    def __len__(self):
        return len(self._tensor)

    def rle_length(self):
        return self._tensor.array().rle_length()

    def empty(self):
        return len(self) == 0

    def clone(self):
        return self._forward_to_unary_array_op("clone")

    def to(self, typ):
        return Tensor.wrap(
            self._cpp_class(self.shape, typ)(self.shape, getattr(self._tensor.array(), typ.__name__)())
        )

    def to_numpy(self):
        np_arr = self._tensor.array().to_numpy()
        return np_arr.reshape(tuple(reversed(self.shape)))

    def to_list(self):
        return self.to_numpy().tolist()

    def to_string(self, threshold=20, separator=" "):
        from .io import format_tensor
        return format_tensor(self, threshold, separator)

    def eval(self):
        return self._forward_to_unary_array_op("eval")

    def item(self):
        assert len(self) == 1
        return self._tensor.array()[0]

    def reshape(self, shape: Union[int, Tuple[int]]):
        from .manipulate import reshape
        return reshape(self, shape)

    def flatten(self, shape: Union[int, Tuple[int]]):
        from .manipulate import flatten
        return flatten(self)
