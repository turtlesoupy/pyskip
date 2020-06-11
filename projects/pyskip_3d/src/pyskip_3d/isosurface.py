import numpy as np
import pyskip
import pyskip.functional as F
import _pyskip_3d_cpp_ext as ext


def marching_cubes(tensor: pyskip.Tensor, surface_density: float = 0.5) -> ext.SurfaceMesh:
    avg_kernel = pyskip.Tensor.from_numpy(np.ones(shape=(2, 2, 2))).to(float) / 8
    density_lattice = F.conv_3d(tensor, avg_kernel, padding=1)
    return ext.marching_cubes(density_lattice._tensor, surface_density)
