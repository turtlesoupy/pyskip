from dataclasses import dataclass
import numpy as np
import skimpy


@dataclass
class VoxelKind:
    empty: bool = False
    color: int


class Voxels:
    """Abstracts a 3D space composed of discrete voxel types.

    Each voxel is associated with "kind" that is stored in an index."""

def __init__(self, w, h, d, default=0):
    self.tensor = skimpy.Tensor(shape=(w, h, d), dtype=int, val=0)



@dataclass
class Mesh:
    positions: np.ndarray[float]
    colors: np.ndarray[float]

class MeshBuilder:

    def __init__(self, voxels, kinds):
        self.voxels = voxels
        self.kinds = kinds

    def to_mesh(self):
        # 1. Create a mask of "empty vs non-empty" voxels
        # 2. For every "surface" face bordering empty and non-empty:
        #   a. Emit 4 vertices
        #   b. Emit vertex color for each vertex based on non-empty voxel type
        #   c. Emit vertex indices for two triangles
