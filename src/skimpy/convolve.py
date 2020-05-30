from skimpy.tensor import Tensor


def conv_2d(tensor: Tensor, kernel: Tensor, padding: int = 0, fill: int = 0) -> Tensor:
    pad_shape = (
        tensor.shape[0] + 2 * padding,
        tensor.shape[1] + 2 * padding,
    )
    pad = Tensor(shape=pad_shape, dtype=tensor.dtype, val=fill)
    pad[padding:-padding, padding:-padding] = tensor

    out_shape = (
        pad_shape[0] - kernel.shape[0] + 1,
        pad_shape[1] - kernel.shape[1] + 1,
    )
    out = Tensor(shape=out_shape, dtype=tensor.dtype, val=0)
    for y in range(kernel.shape[1]):
        for x in range(kernel.shape[0]):
            stop_x = pad_shape[0] - kernel.shape[0] + x + 1
            stop_y = pad_shape[1] - kernel.shape[1] + y + 1
            out += kernel[x, y] * pad[x:stop_x, y:stop_y]
    return out


def conv_3d(tensor: Tensor, kernel: Tensor, padding: int = 0, fill: int = 0) -> Tensor:
    if padding == 0:
        pad = tensor
        pad_shape = tensor.shape
    else:
        pad_shape = (
            tensor.shape[0] + 2 * padding,
            tensor.shape[1] + 2 * padding,
            tensor.shape[2] + 2 * padding,
        )
        pad = Tensor(shape=pad_shape, dtype=tensor.dtype, val=fill)
        pad[padding:-padding, padding:-padding, padding:-padding] = tensor

    out_shape = (
        pad_shape[0] - kernel.shape[0] + 1,
        pad_shape[1] - kernel.shape[1] + 1,
        pad_shape[2] - kernel.shape[2] + 1,
    )

    out = Tensor(shape=out_shape, dtype=tensor.dtype, val=0)
    for z in range(kernel.shape[2]):
        for y in range(kernel.shape[1]):
            for x in range(kernel.shape[0]):
                stop_x = pad_shape[0] - kernel.shape[0] + x + 1
                stop_y = pad_shape[1] - kernel.shape[1] + y + 1
                stop_z = pad_shape[2] - kernel.shape[2] + z + 1
                out += kernel[x, y, z] * pad[x:stop_x, y:stop_y, z:stop_z]
    return out
