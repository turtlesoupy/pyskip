from typing import List, Optional, Union, Tuple
from collections.abc import Iterable


def unify_slices(slices: Union[int, slice, List[Union[slice]]]) -> slice:
    """Maps a mixed tuple of slices and int indices to a tuple of all slices."""
    if not isinstance(slices, Iterable):
        return (slices, )
    if all(isinstance(sl, int) for sl in slices):
        return slices
    ret = []
    for sl in slices:
        if isinstance(sl, int):
            ret.append(slice(sl, sl + 1))
        else:
            assert isinstance(sl, slice)
            ret.append(sl)
    return tuple(ret)


def broadcast_slice(ndim: int, sl: slice, axis: Optional[int] = None) -> Tuple[slice]:
    """Returns a tuple of slices by broadcasting the given slice.

     The tuple has size matching the given Tensor dimensionality "dim" and has
     default slices at each axis except with "sl" at the specified axis. If the
     "axis" argument is None (the default), "sl" is inserted at each axis."""
    assert axis is None or 0 <= axis < ndim
    return tuple(sl if axis is None or i == axis else slice(None) for i in range(ndim))


def broadcast_shape(ndim: int, shape: Union[int, Tuple[int]]) -> Tuple[int]:
    """Broadcasts shape if necessary the provided number of dimensions."""
    if isinstance(shape, int):
        return tuple(shape for _ in range(ndim))
    if len(shape) == 1:
        return tuple(shape[0] for _ in range(ndim))
    assert len(shape) == ndim
    return shape
