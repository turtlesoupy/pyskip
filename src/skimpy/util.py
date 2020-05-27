from typing import List, Optional, Union, Tuple
from collections.abc import Iterable

from . import tensor


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


def broadcast_slice(dim: int, sl: slice, axis: Optional[int] = None) -> Tuple[slice]:
    """Returns a tuple of slices by broadcasting the given slice.

     The tuple has size matching the given Tensor dimensionality "dim" and has
     default slices at each axis except with "sl" at the specified axis. If the
     "axis" argument is None (the default), "sl" is inserted at each axis."""
    assert axis is None or 0 <= axis < dim
    return tuple(sl if axis is None or i == axis else slice(None) for i in range(dim))


def take(t: "Tensor", sl: slice, axis: Optional[int] = None) -> "Tensor":
    """Returns the sub-tensor by applying the given slice to axis.
  
     If axis is None (the default), the slice is applied to all axes."""
    return t[broadcast_slice(t.dimensionality, sl, axis)]


def stack(t1: "Tensor", t2: "Tensor") -> "Tensor":
    assert t1.dimensionality == t2.dimensionality
    assert t1.dtype == t2.dtype
    t3 = tensor.Tensor(tuple(l1 + l2 for l1, l2 in zip(t1.shape, t2.shape)))
    t3[tuple(slice(None, l) for l in t1.shape)] = t1
    t3[tuple(slice(l, None) for l in t1.shape)] = t2
    return t3
