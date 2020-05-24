import re
import _skimpy_cpp_ext

from . exceptions import InvalidTensorError

_type_mapping = {
    "int32": "i"
}
_type_mapping_reverse = {v: k for k, v in _type_mapping.items()}


class Tensor:
    @classmethod
    def wrap(cls, cpp_tensor):
        return cls(cpp_tensor=cpp_tensor)

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
        self.dimensionality = m.group(1)
        self.dtype = _type_mapping_reverse[m.group(2)]

