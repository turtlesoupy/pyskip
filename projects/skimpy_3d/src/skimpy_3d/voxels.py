import random
import skimpy
from _skimpy_3d_cpp_ext import (
    generate_mesh,
    VoxelConfig,
    VoxelMesh,
    EmptyVoxel,
    ColorVoxel,
)


def random_color_config(n: int) -> VoxelConfig:
    config = VoxelConfig.from_dict({
        0: EmptyVoxel(),
    })
    for i in range(1, n):
        config[i] = ColorVoxel(
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )
    return config


def to_mesh(tensor: skimpy.Tensor, config: VoxelConfig) -> VoxelMesh:
    return generate_mesh(config, tensor._tensor)
