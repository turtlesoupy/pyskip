import re
import _skimpy_cpp_ext
from .exceptions import InvalidTensorError, IncompatibleTensorError, UnimplementedOperationError

_type_mapping = {"int32": "i"}
_type_mapping_reverse = {v: k for k, v in _type_mapping.items()}


class Tensor:
    @classmethod
    def wrap(cls, cpp_tensor):
        return cls(cpp_tensor=cpp_tensor)

    @classmethod
    def from_numpy(cls, np_arr):
        np_shape = np_arr.shape
        skimpy_arr = _skimpy_cpp_ext.from_numpy(np_arr.flatten("C"))
        return cls(shape=np_shape, val=skimpy_arr)

    def __init__(self, shape=None, val=0, dtype="int32", cpp_tensor=None):
        if cpp_tensor:
            self._init_from_cpp_tensor(cpp_tensor)
            return

        if shape is None:
            raise InvalidTensorError("Must specify shape")

        if isinstance(shape, int):
            shape = (shape,)

        dimensionality = len(shape)

        if dimensionality == 1:
            shape = shape[0]

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
        if a.dtype != b.dtype:
            raise IncompatibleTensorError(f"Incompatible types: {a.dtype} and {b.dtype}")

        if a.shape != b.shape and b.shape != (1,):
            raise IncompatibleTensorError(f"Incompatible shapes: {a.shape} and {b.shape}")

        return (a, b)

    # Binary operators
    @classmethod
    def _forward_to_binary_array_op(cls, a, b, op):
        if isinstance(b, int):
            return Tensor.wrap(a._tensor.__class__(a.shape, getattr(a._tensor.array(), op)(b)))
        else:
            a, b = Tensor._validate_or_cast(a, b)
            return Tensor.wrap(a._tensor.__class__(a.shape, getattr(a._tensor.array(), op)(b._tensor.array())))
    
    def __add__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__add__")

    def __iadd__(self, other):
        return self._init_from_cpp_tensor(self.__add__(other)._tensor)

    def __sub__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__sub__")

    def __isub__(self, other):
        return self._init_from_cpp_tensor(self.__sub__(other)._tensor)

    def __mul__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__mul__")

    def __imul__(self, other):
        return self._init_from_cpp_tensor(self.__mul__(other)._tensor)

    def __truediv__(self, other):
        raise UnimplementedOperationError("True division is unsupported, try integer division (//)")

    def __idiv__(self, other):
        return self._init_from_cpp_tensor(self.__truediv__(other)._tensor)

    def __floordiv__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__floordiv__")

    def __ifloordiv__(self, other):
        return self._init_from_cpp_tensor(self.__floordiv__(other)._tensor)

    def __mod__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__mod__")

    def __imod__(self, other):
        return self._init_from_cpp_tensor(self.__mod__(other)._tensor)
    
    def __and__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__and__")

    def __iand__(self, other):
        return self._init_from_cpp_tensor(self.__and__(other)._tensor)

    def __xor__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__xor__")

    def __ixor__(self, other):
        return self._init_from_cpp_tensor(self.__xor__(other)._tensor)

    def __or__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__or__")

    def __ior__(self, other):
        return self._init_from_cpp_tensor(self.__or__(other)._tensor)

    def __pow__(self, other):
        return Tensor._forward_to_binary_array_op(self, other, "__pow__")

    def __ipow__(self, other):
        return self._init_from_cpp_tensor(self.__pow__(other)._tensor)

    def __setitem__(self, index, value):
        self._tensor[index] = value

    def __getitem__(self, index):
        return Tensor.wrap(self._tensor[index])

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


    def to_numpy(self):
        np_arr = self._tensor.array().to_numpy()
        return np_arr.reshape(self.shape)

    @classmethod
    def _array_str(cls, arr):
        if len(arr) > 10:
            ret = []
            for i in range(8):
                ret.append(str(arr[i]))
            ret.append("...")
            for i in range(len(arr) - 2, len(arr)):
                ret.append(str(arr[i]))
            return ",".join(ret)
        else:
            return ",".join(str(arr[i]) for i in range(len(arr)))

    def __str__(self):
        def print_2d(n_rows, n_cols):
            ret = []
            if n_rows < 10:
                for i in range(n_rows):
                    ret.append("\t" + self._array_str(self._tensor.array()[i * n_cols : ((i + 1) * n_cols)]))
            else:
                for i in range(8):
                    ret.append("\t" + self._array_str(self._tensor.array()[i * n_cols : ((i + 1) * n_cols)]))
                ret.append("\t...")
                for i in range(n_rows - 2, n_rows):
                    ret.append("\t" + self._array_str(self._tensor.array()[i * n_cols : ((i + 1) * n_cols)]))

            return "\n".join(ret)

        if self.dimensionality == 1:
            return self._array_str(self._tensor.array())
        if self.dimensionality == 2:
            n_rows = self.shape[1]  # SWAPME
            n_cols = self.shape[0]
            return print_2d(n_rows, n_cols)
        elif self.dimensionality == 3:
            n_rows = self.shape[1]  # SWAPME
            n_cols = self.shape[0]
            st = print_2d(n_rows, n_cols)
            return f"\tz=0\n{st}\n\tz=..."
        else:
            return str(self._tensor)

    def __repr__(self):
        return f"Tensor (dtype={self.dtype}, shape={self.shape}) with values\n{str(self)}"
