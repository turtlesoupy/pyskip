from typing import Optional
from pyskip.util import broadcast_slice
from pyskip.tensor import Tensor


def take(t: Tensor, sl: slice, axis: Optional[int] = None) -> Tensor:
  """Returns the sub-tensor by applying the given slice to axis.
  
     If axis is None (the default), the slice is applied to all axes."""
  return t[broadcast_slice(t.ndim, sl, axis)]
