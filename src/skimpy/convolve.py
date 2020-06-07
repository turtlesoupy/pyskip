from skimpy.util import broadcast_shape
from skimpy.tensor import Tensor


def conv_2d(tensor: Tensor, kernel: Tensor, padding=0, fill=0) -> Tensor:
    assert tensor.ndim == 2
    assert kernel.ndim == 2
    tw, th = tensor.shape
    kw, kh = kernel.shape
    pw, ph = broadcast_shape(2, padding)

    # Assign the input tensor into one that's padded with the fill value.
    sw, sh = tw + 2 * pw, th + 2 * ph
    s = Tensor(shape=(sw, sh), dtype=tensor.dtype, val=fill)
    s[pw:sw - pw, ph:sh - ph] = tensor

    # Run the kernel over the padded tensor.
    out = Tensor(shape=(sw - kw + 1, sh - kh + 1), dtype=tensor.dtype, val=0)
    for y in range(kh):
        for x in range(kw):
            stop_x = sw - kw + x + 1
            stop_y = sh - kh + y + 1
            out += kernel[x, y] * s[x:stop_x, y:stop_y]
    return out


def conv_3d(tensor: Tensor, kernel: Tensor, padding=0, fill=0) -> Tensor:
    assert tensor.ndim == 3
    assert kernel.ndim == 3
    tw, th, td = tensor.shape
    kw, kh, kd = kernel.shape
    pw, ph, pd = broadcast_shape(3, padding)

    # Assign the input tensor into one that's padded with the fill value.
    sw, sh, sd = tw + 2 * pw, th + 2 * ph, td + 2 * pd
    s = Tensor(shape=(sw, sh, sd), dtype=tensor.dtype, val=fill)
    s[pw:sw - pw, ph:sh - ph, pd:sd - pd] = tensor

    out = Tensor(shape=(sw - kw + 1, sh - kh + 1, sd - kd + 1), dtype=tensor.dtype, val=0)
    for z in range(kd):
        for y in range(kh):
            for x in range(kw):
                stop_x = sw - kw + x + 1
                stop_y = sh - kh + y + 1
                stop_z = sd - kd + z + 1
                out += kernel[x, y, z] * s[x:stop_x, y:stop_y, z:stop_z]
    return out
