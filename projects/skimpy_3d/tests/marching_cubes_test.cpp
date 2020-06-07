#define CATCH_CONFIG_MAIN

#include <fmt/ranges.h>

#include <catch2/catch.hpp>
#include <marching_cubes.hpp>
#include <skimpy/skimpy.hpp>

using namespace skimpy_3d;

TEST_CASE("Test marching cubes", "[marching_cubes]") {
  auto lattice = skimpy::make_tensor<3, float>({4, 4, 4}, 0.0f);
  lattice.set({1, 1, 1}, 1.0f);
  lattice.set({2, 1, 1}, 1.0f);
  lattice.set({1, 2, 1}, 1.0f);
  lattice.set({2, 2, 1}, 1.0f);
  lattice.set({1, 1, 2}, 1.0f);
  lattice.set({2, 1, 2}, 1.0f);
  lattice.set({1, 2, 2}, 1.0f);
  lattice.set({2, 2, 2}, 1.0f);

  auto mesh = marching_cubes(lattice);

  for (int i = 0; i < mesh.vertex_count(); i += 1) {
    fmt::print(
        "{}, {}, {}\n",
        mesh.positions[3 * i],
        mesh.positions[3 * i + 1],
        mesh.positions[3 * i + 2]);
  }
}
