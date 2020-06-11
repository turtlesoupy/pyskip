import skimpy
from skimpy import functional as F


def resize():
    pass


def blur(tensor: skimpy.Tensor):
    G_x = (1 / 64.0) * skimpy.Tensor.from_list([[[1, 6, 15, 20, 15, 6, 1]]]).to(float)
    G_y = (1 / 64.0) * skimpy.Tensor.from_list([[[1], [6], [15], [20], [15], [6], [1]]]).to(float)
    G_z = (1 / 64.0) * skimpy.Tensor.from_list([[[1]], [[6]], [[15]], [[20]], [[15]], [[6]], [[1]]]).to(float)
    tensor = F.conv_3d(tensor, G_x, padding=(3, 0, 0), fill=0.0)
    tensor = F.conv_3d(tensor, G_y, padding=(0, 3, 0), fill=0.0)
    tensor = F.conv_3d(tensor, G_z, padding=(0, 0, 3), fill=0.0)
    return tensor
