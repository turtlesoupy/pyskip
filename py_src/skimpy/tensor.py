import _skimpy_cpp_ext

from . exceptions import InvalidTensorError

_type_mapping = {
    "int32": "i"
}


class Tensor:
    def __init__(self, shape, val=0, dtype="int32"):
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
        self._tensor = getattr(_skimpy_cpp_ext, klass)(shape, val)