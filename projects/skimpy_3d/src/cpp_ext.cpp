#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "voxels.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_skimpy_3d_cpp_ext, m) {
  using namespace skimpy_3d;

  m.doc() = "Wrapper skimpy library with 3D routines";
  m.attr("__version__") = "0.0.1";

  // Bind the VoxelConfig class.
  py::class_<VoxelConfig>(m, "VoxelConfig")
      .def(py::init<>())
      .def(
          "__setitem__",
          [](VoxelConfig& config, VoxelConfig::Key key, VoxelDef def) {
            config.defs.emplace(key, def);
          });
  py::class_<EmptyVoxel>(m, "EmptyVoxel").def(py::init<>());
  py::class_<ColorVoxel>(m, "ColorVoxel").def(py::init<int, int, int>());

  // Bind the VoxelMesh class.
  py::class_<VoxelMesh>(m, "VoxelMesh")
      .def(py::init<>())
      .def_readwrite("positions", &VoxelMesh::positions)
      .def_readwrite("normals", &VoxelMesh::normals)
      .def_readwrite("colors", &VoxelMesh::colors)
      .def_readwrite("triangles", &VoxelMesh::triangles);

  // Bind the mesh generation routine.
  m.def("generate_mesh", generate_mesh);
};
