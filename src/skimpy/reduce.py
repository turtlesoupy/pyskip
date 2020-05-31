import operator
from typing import Callable, Union
from skimpy.index import take
from skimpy.manipulate import flatten, squeeze
from skimpy.tensor import Scalar, Tensor


def reduce(
    op: Callable[[Scalar, Scalar], Scalar],
    t: Tensor,
    axis: int = None,
    keepdims=False,
) -> Union[Tensor, Scalar]:
    assert len(t) > 0
    if axis is None:
        return reduce(op, flatten(t), axis=0, keepdims=keepdims)
    assert 0 <= axis < len(t.shape)
    if len(t) == 1:
        return t if keepdims else t.item()
    if t.shape[axis] == 1:
        return t if keepdims else squeeze(t)
    else:
        n = t.shape[axis]
        if n % 2 == 1:
            last = take(t, slice(-1, n), axis)
            if len(t.shape) == 1:
                last = last.item()

            return op(reduce(op, take(t, slice(-1), axis)), last)
        else:
            lo = take(t, slice(0, n, 2), axis)
            hi = take(t, slice(1, n, 2), axis)
            return reduce(op, op(lo, hi), axis, keepdims)


def add(t: Tensor, axis: int = None, keepdims=False) -> Tensor:
    if t.empty():
        return t.dtype(0)
    return reduce(operator.add, t, axis, keepdims)


def mul(t: Tensor, axis: int = None, keepdims=False) -> Tensor:
    if t.empty():
        return t.dtype(1)
    return reduce(operator.mul, t, axis, keepdims)


# Convenience aliases for reduction operators
sum = add
prod = mul
