#define CATCH_CONFIG_MAIN

#include <fmt/ranges.h>

#include <catch2/catch.hpp>
#include <skimpy/skimpy.hpp>
#include <voxels.hpp>

using namespace skimpy_3d;

TEST_CASE("Test generate mesh", "[voxels]") {
  auto tensor = skimpy::make_tensor<3>({3, 3, 3}, 0);
  tensor.set({0, 0, 0}, 1);
  tensor.set({1, 0, 0}, 2);

  VoxelConfig config;
  config.defs.emplace(0, EmptyVoxel());
  config.defs.emplace(1, ColorVoxel(255, 0, 0));
  config.defs.emplace(2, ColorVoxel(0, 0, 255));

  auto mesh = generate_mesh(config, tensor);
  fmt::print("{}\n", fmt::join(mesh.positions, ", "));
  fmt::print("{}\n", fmt::join(mesh.colors, ", "));
  fmt::print("{}\n", fmt::join(mesh.triangles, ", "));
}
