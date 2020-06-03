#pragma once

#include <cstdint>
#include <skimpy/skimpy.hpp>
#include <unordered_map>

#include "utils.hpp"

namespace skimpy_3d {

struct EmptyVoxel {};

struct ColorVoxel {
  int32_t rgb;

  ColorVoxel(int32_t r, int32_t g, int32_t b) : rgb((r << 16) | (g << 8) | b) {}
};

using VoxelDef = std::variant<EmptyVoxel, ColorVoxel>;

struct VoxelConfig {
  using Key = int32_t;
  std::unordered_map<Key, VoxelDef> defs;
};

using VoxelTensor = skimpy::Tensor<3, int32_t>;

struct VoxelMesh {
  std::vector<float> positions;
  std::vector<uint8_t> colors;
  std::vector<uint16_t> triangles;

  auto vertex_count() const {
    return positions.size() / 3;
  }

  auto triangle_count() const {
    return triangles.size() / 3;
  }
};

auto compute_adjacency_mask(const VoxelTensor& tensor) {
  auto w = tensor.shape()[0];
  auto h = tensor.shape()[1];
  auto d = tensor.shape()[2];
  auto sl = [&](int x_0, int x_1, int y_0, int y_1, int z_0, int z_1) {
    x_0 = x_0 < 0 ? w + x_0 : x_0;
    y_0 = y_0 < 0 ? h + y_0 : y_0;
    z_0 = z_0 < 0 ? d + z_0 : z_0;
    x_1 = x_1 <= 0 ? w + x_1 : x_1;
    y_1 = y_1 <= 0 ? h + y_1 : y_1;
    z_1 = z_1 <= 0 ? d + z_1 : z_1;
    return skimpy::TensorSlice<3>(
        {{x_0, x_1, 1}, {y_0, y_1, 1}, {z_0, z_1, 1}});
  };

  auto update = [](auto& tensor, auto slice, auto input) {
    auto merged = tensor.get(slice).array() | input;
    tensor.set(slice, skimpy::make_tensor<3>(slice.shape(), std::move(merged)));
  };

  auto empty = skimpy::make_tensor<3>(
      tensor.shape(), skimpy::splat<int>(tensor.array() == 0, 0xFFFFFFFF, 0));
  auto adjacencies = skimpy::make_tensor<3>(tensor.shape(), 0);

  constexpr auto x_neg_mask = static_cast<int>(DirMask::X_NEG);
  constexpr auto x_pos_mask = static_cast<int>(DirMask::X_POS);
  constexpr auto y_neg_mask = static_cast<int>(DirMask::Y_NEG);
  constexpr auto y_pos_mask = static_cast<int>(DirMask::Y_POS);
  constexpr auto z_neg_mask = static_cast<int>(DirMask::Z_NEG);
  constexpr auto z_pos_mask = static_cast<int>(DirMask::Z_POS);

  // Initialize the adjacency tensor boundaries.
  update(
      adjacencies,
      sl(0, 1, 0, h, 0, d),
      x_neg_mask & ~empty.get(sl(0, 1, 0, h, 0, d)).array());
  update(
      adjacencies,
      sl(-1, w, 0, h, 0, d),
      x_pos_mask & ~empty.get(sl(-1, w, 0, h, 0, d)).array());
  update(
      adjacencies,
      sl(0, w, 0, 1, 0, d),
      y_neg_mask & ~empty.get(sl(0, w, 0, 1, 0, d)).array());
  update(
      adjacencies,
      sl(0, w, -1, h, 0, d),
      y_pos_mask & ~empty.get(sl(0, w, -1, h, 0, d)).array());
  update(
      adjacencies,
      sl(0, w, 0, h, 0, 1),
      z_neg_mask & ~empty.get(sl(0, w, 0, h, 0, 1)).array());
  update(
      adjacencies,
      sl(0, w, 0, h, -1, d),
      z_pos_mask & ~empty.get(sl(0, w, 0, h, -1, d)).array());

  // Initialize the adjacency tensor interior values.
  auto shift = [&](int dx, int dy, int dz) {
    auto x_0 = std::max(0, dx), x_1 = std::min(w, w + dx);
    auto y_0 = std::max(0, dy), y_1 = std::min(h, h + dy);
    auto z_0 = std::max(0, dz), z_1 = std::min(d, d + dz);
    return sl(x_0, x_1, y_0, y_1, z_0, z_1);
  };
  auto edge = [&](int dx, int dy, int dz) {
    auto s_1 = shift(dx, dy, dz);
    auto s_2 = shift(-dx, -dy, -dz);
    return ~empty.get(s_1).array() & empty.get(s_2).array();
  };

  update(adjacencies, shift(1, 0, 0), x_neg_mask & edge(1, 0, 0));
  update(adjacencies, shift(-1, 0, 0), x_pos_mask & edge(-1, 0, 0));
  update(adjacencies, shift(0, 1, 0), y_neg_mask & edge(0, 1, 0));
  update(adjacencies, shift(0, -1, 0), y_pos_mask & edge(0, -1, 0));
  update(adjacencies, shift(0, 0, 1), z_neg_mask & edge(0, 0, 1));
  update(adjacencies, shift(0, 0, -1), z_pos_mask & edge(0, 0, -1));

  return adjacencies;
}

auto generate_mesh(const VoxelConfig& config, const VoxelTensor& tensor) {
  VoxelMesh mesh;

  // Adds 4 vertices and 2 triangles (for one cube face) to the output mesh.
  auto emit_face_geometry = [&](Vec3i origin, DirMask dir) {
    static auto positions = [] {
      using T = std::array<Vec3i, 4>;
      std::vector<T> ret;
      ret.push_back(T{{{0, 0, 0}, {0, 1, 0}, {0, 1, 1}, {0, 0, 1}}});  // X_NEG
      ret.push_back(T{{{1, 1, 0}, {1, 0, 0}, {1, 0, 1}, {1, 1, 1}}});  // X_POS
      ret.push_back(T{{{1, 0, 0}, {0, 0, 0}, {0, 0, 1}, {1, 0, 1}}});  // Y_NEG
      ret.push_back(T{{{0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}}});  // Y_POS
      ret.push_back(T{{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}}});  // Z_NEG
      ret.push_back(T{{{0, 1, 1}, {1, 1, 1}, {1, 0, 1}, {0, 0, 1}}});  // Z_POS
      return ret;
    }();

    // Emit vertex indices for the two new triangles.
    auto base = mesh.vertex_count();
    mesh.triangles.push_back(base);
    mesh.triangles.push_back(base + 2);
    mesh.triangles.push_back(base + 1);
    mesh.triangles.push_back(base + 3);
    mesh.triangles.push_back(base + 2);
    mesh.triangles.push_back(base);

    // Emit vertex positions for the 4 new vertices.
    auto dir_index = lg2(static_cast<int>(dir));
    for (const auto& pos_vec : positions.at(dir_index)) {
      Vec3i out = add(origin, pos_vec);
      mesh.positions.push_back(out[0]);
      mesh.positions.push_back(out[1]);
      mesh.positions.push_back(out[2]);
    }
  };

  // Adds 4 vertex color attributes (for one cube face) to the output mesh.
  auto emit_face_colors = [&](int32_t rgb) {
    for (int i = 0; i < 4; i += 1) {
      mesh.colors.push_back((rgb & 0xFF0000) >> 16);
      mesh.colors.push_back((rgb & 0x00FF00) >> 8);
      mesh.colors.push_back((rgb & 0x0000FF));
    }
  };

  // Create a tensor of the adjacency bitmask at each non-empty tensor cell.
  auto adjacencies = compute_adjacency_mask(tensor).eval();

  // Iterate over each boundary voxel and emit its mesh attributes.
  array_walk(
      [&](auto pos, int adj, int key) {
        auto w = tensor.shape()[0];
        auto h = tensor.shape()[1];
        auto d = tensor.shape()[2];
        Vec3i origin = {pos % w, (pos / w) % h, pos / (w * h)};

        // Emit different geometry based on the type of voxel.
        std::visit(
            Overloaded{
                [](const EmptyVoxel&) {
                  CHECK_UNREACHABLE("Adjacency mask included an empty voxel");
                },
                [&](const ColorVoxel& v) {
                  if (adj & static_cast<int>(DirMask::X_NEG)) {
                    emit_face_geometry(origin, DirMask::X_NEG);
                    emit_face_colors(v.rgb);
                  }
                  if (adj & static_cast<int>(DirMask::X_POS)) {
                    emit_face_geometry(origin, DirMask::X_POS);
                    emit_face_colors(v.rgb);
                  }
                  if (adj & static_cast<int>(DirMask::Y_NEG)) {
                    emit_face_geometry(origin, DirMask::Y_NEG);
                    emit_face_colors(v.rgb);
                  }
                  if (adj & static_cast<int>(DirMask::Y_POS)) {
                    emit_face_geometry(origin, DirMask::Y_POS);
                    emit_face_colors(v.rgb);
                  }
                  if (adj & static_cast<int>(DirMask::Z_NEG)) {
                    emit_face_geometry(origin, DirMask::Z_NEG);
                    emit_face_colors(v.rgb);
                  }
                  if (adj & static_cast<int>(DirMask::Z_POS)) {
                    emit_face_geometry(origin, DirMask::Z_POS);
                    emit_face_colors(v.rgb);
                  }
                },
            },
            config.defs.at(key));
      },
      adjacencies.array() != 0,
      adjacencies.array(),
      tensor.array());

  return mesh;
}

}  // namespace skimpy_3d
