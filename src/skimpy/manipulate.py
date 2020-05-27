import operator
from functools import reduce
from typing import Union, Tuple, List
from skimpy.tensor import Tensor


def shape_len(shape: Tuple[int]) -> int:
  return reduce(operator.mul, shape)


def reshape(t: Tensor, shape: Union[int, Tuple[int]]) -> Tensor:
  if isinstance(shape, int):
    shape = (shape, )
  assert (len(t) == shape_len(shape))
  return Tensor(shape, val = t._tensor.array())


def flatten(t: Tensor) -> Tensor:
  return reshape(t, len(t))


def squeeze(t: Tensor, axis = None) -> Tensor:
  new_shape = []
  for i, s in enumerate(t.shape):
    if s != 1 or (axis is not None and axis != i):
      new_shape.append(s)
  if not new_shape:
    new_shape = (1, )
  return reshape(t, tuple(new_shape))


def stack(t1: Tensor, t2: Tensor) -> Tensor:
  """Returns a new tensor by concatenating the values of the two input tensors.

     The output shape is the sum of the shapes of the input tensors. The first
     range of indices are filled with t1, and the rest are filled with t2."""
  assert t1.ndim == t2.ndim
  assert t1.dtype == t2.dtype
  t3 = Tensor(tuple(l1 + l2 for l1, l2 in zip(t1.shape, t2.shape)))
  t3[tuple(slice(None, l) for l in t1.shape)] = t1
  t3[tuple(slice(l, None) for l in t1.shape)] = t2
  return t3


def concat(tensors: List[Tensor]) -> Tensor:
  """Stacks the values in the given tensors into one output tensor."""
  assert len(tensors) > 0
  if len(tensors) == 1:
    return tensors[0].clone()
  elif len(tensors) == 2:
    return stack(tensors[0], tensors[1])
  else:
    return stack(tensors[0], concat(tensors[1:]))
