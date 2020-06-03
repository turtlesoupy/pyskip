import numpy as np
import skimpy
from _skimpy_3d_cpp_ext import generate_mesh, VoxelConfig, VoxelMesh, EmptyVoxel, ColorVoxel


class VoxelTensor:
    def __new__(cls, w, h, d, default=0):
        return skimpy.Tensor(shape=(w, h, d), val=default)
