import numpy as np
from skimpy.index import take
from skimpy.manipulate import stack
from skimpy.tensor import Tensor


def format_tensor(tensor: Tensor, threshold: int = 20, separator: str = " "):
    """Formats tensor into a string for printing / visualizing.

    The threshold controls truncation of the output (i.e. when to add ellipses)
    and values are joined by the separator string."""
    truncated = tensor
    for i, l in enumerate(tensor.shape):
        if l > threshold:
            head = take(truncated.ndim, slice(threshold), axis=i)
            tail = take(truncated.ndim, slice(-threshold, None), axis=i)
            truncated = stack(head, tail)
    return np.array2string(truncated.to_numpy(), threshold=threshold, separator=separator)
