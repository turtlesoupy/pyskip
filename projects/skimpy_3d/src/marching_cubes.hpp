#pragma once

#include <fmt/ranges.h>

#include <skimpy/skimpy.hpp>
#include <unordered_map>

#include "utils.hpp"

namespace skimpy_3d {

struct SurfaceMesh {
  std::vector<float> positions;
  std::vector<uint32_t> triangles;

  auto vertex_count() const {
    return positions.size() / 3;
  }

  auto triangle_count() const {
    return triangles.size() / 3;
  }
};

static constexpr std::array<std::array<Vec3i, 2>, 12> kEdgeKeys = {{
    {Vec3i{0, 0, 0}, Vec3i{0, 0, 1}},
    {Vec3i{0, 0, 0}, Vec3i{0, 1, 0}},
    {Vec3i{0, 0, 0}, Vec3i{1, 0, 0}},
    {Vec3i{0, 0, 1}, Vec3i{0, 1, 1}},
    {Vec3i{0, 0, 1}, Vec3i{1, 0, 1}},
    {Vec3i{0, 1, 0}, Vec3i{0, 1, 1}},
    {Vec3i{0, 1, 0}, Vec3i{1, 1, 0}},
    {Vec3i{1, 0, 0}, Vec3i{1, 0, 1}},
    {Vec3i{1, 0, 0}, Vec3i{1, 1, 0}},
    {Vec3i{0, 1, 1}, Vec3i{1, 1, 1}},
    {Vec3i{1, 0, 1}, Vec3i{1, 1, 1}},
    {Vec3i{1, 1, 0}, Vec3i{1, 1, 1}},
}};

static const std::array<std::vector<Vec3i>, 256> kTrianglesIndex = {{
    {},
    {{2, 1, 0}},
    {{2, 7, 8}},
    {{1, 0, 7}, {1, 7, 8}},
    {{1, 6, 5}},
    {{0, 2, 6}, {0, 6, 5}},
    {{2, 7, 8}, {1, 6, 5}},
    {{6, 5, 8}, {5, 7, 8}, {5, 0, 7}},
    {{8, 11, 6}},
    {{2, 1, 0}, {6, 8, 11}},
    {{2, 7, 11}, {2, 11, 6}},
    {{1, 0, 6}, {0, 11, 6}, {0, 7, 11}},
    {{1, 8, 11}, {1, 11, 5}},
    {{8, 11, 2}, {11, 0, 2}, {11, 5, 0}},
    {{2, 7, 1}, {7, 5, 1}, {7, 11, 5}},
    {{0, 7, 11}, {0, 11, 5}},
    {{0, 3, 4}},
    {{2, 1, 3}, {2, 3, 4}},
    {{2, 7, 8}, {0, 3, 4}},
    {{7, 8, 4}, {8, 3, 4}, {8, 1, 3}},
    {{1, 6, 5}, {0, 3, 4}},
    {{3, 4, 5}, {4, 6, 5}, {4, 2, 6}},
    {{0, 3, 4}, {2, 7, 8}, {1, 6, 5}},
    {{6, 7, 8}, {6, 4, 7}, {6, 5, 4}, {5, 3, 4}},
    {{6, 8, 11}, {0, 3, 4}},
    {{2, 1, 3}, {2, 3, 4}, {6, 8, 11}},
    {{6, 2, 7}, {6, 7, 11}, {0, 3, 4}},
    {{6, 1, 3}, {6, 3, 11}, {11, 3, 4}, {11, 4, 7}},
    {{1, 8, 11}, {1, 11, 5}, {0, 3, 4}},
    {{8, 4, 2}, {8, 11, 4}, {11, 3, 4}, {11, 5, 3}},
    {{2, 7, 1}, {7, 5, 1}, {7, 11, 5}, {0, 3, 4}},
    {{4, 7, 3}, {7, 5, 3}, {7, 11, 5}},
    {{7, 4, 10}},
    {{2, 1, 0}, {7, 4, 10}},
    {{8, 2, 4}, {8, 4, 10}},
    {{4, 10, 0}, {10, 1, 0}, {10, 8, 1}},
    {{1, 6, 5}, {7, 4, 10}},
    {{2, 6, 5}, {2, 5, 0}, {7, 4, 10}},
    {{2, 4, 10}, {2, 10, 8}, {1, 6, 5}},
    {{6, 10, 8}, {6, 5, 10}, {5, 4, 10}, {5, 0, 4}},
    {{6, 8, 11}, {7, 4, 10}},
    {{2, 1, 0}, {6, 8, 11}, {7, 4, 10}},
    {{11, 6, 10}, {6, 4, 10}, {6, 2, 4}},
    {{1, 11, 6}, {1, 10, 11}, {1, 0, 10}, {0, 4, 10}},
    {{1, 8, 11}, {1, 11, 5}, {7, 4, 10}},
    {{8, 11, 2}, {11, 0, 2}, {11, 5, 0}, {7, 4, 10}},
    {{1, 2, 4}, {1, 4, 5}, {5, 4, 10}, {5, 10, 11}},
    {{10, 11, 4}, {11, 0, 4}, {11, 5, 0}},
    {{7, 0, 3}, {7, 3, 10}},
    {{2, 1, 7}, {1, 10, 7}, {1, 3, 10}},
    {{0, 3, 2}, {3, 8, 2}, {3, 10, 8}},
    {{8, 1, 3}, {8, 3, 10}},
    {{7, 0, 3}, {7, 3, 10}, {1, 6, 5}},
    {{5, 3, 10}, {5, 10, 6}, {6, 10, 7}, {6, 7, 2}},
    {{0, 3, 2}, {3, 8, 2}, {3, 10, 8}, {1, 6, 5}},
    {{5, 3, 6}, {3, 8, 6}, {3, 10, 8}},
    {{7, 0, 3}, {7, 3, 10}, {6, 8, 11}},
    {{2, 1, 7}, {1, 10, 7}, {1, 3, 10}, {6, 8, 11}},
    {{0, 6, 2}, {0, 3, 6}, {3, 11, 6}, {3, 10, 11}},
    {{6, 1, 11}, {1, 10, 11}, {1, 3, 10}},
    {{7, 0, 3}, {7, 3, 10}, {1, 8, 11}, {1, 11, 5}},
    {{2, 8, 11}, {2, 11, 5}, {2, 5, 3}, {2, 3, 10}, {2, 10, 7}},
    {{2, 0, 3}, {2, 3, 10}, {2, 10, 11}, {2, 11, 5}, {2, 5, 1}},
    {{5, 3, 10}, {5, 10, 11}},
    {{5, 9, 3}},
    {{2, 1, 0}, {5, 9, 3}},
    {{2, 7, 8}, {5, 9, 3}},
    {{1, 0, 7}, {1, 7, 8}, {5, 9, 3}},
    {{1, 6, 9}, {1, 9, 3}},
    {{0, 2, 3}, {2, 9, 3}, {2, 6, 9}},
    {{1, 6, 9}, {1, 9, 3}, {2, 7, 8}},
    {{3, 0, 7}, {3, 7, 9}, {9, 7, 8}, {9, 8, 6}},
    {{8, 11, 6}, {5, 9, 3}},
    {{2, 1, 0}, {6, 8, 11}, {5, 9, 3}},
    {{2, 7, 11}, {2, 11, 6}, {5, 9, 3}},
    {{1, 0, 6}, {0, 11, 6}, {0, 7, 11}, {5, 9, 3}},
    {{9, 3, 11}, {3, 8, 11}, {3, 1, 8}},
    {{2, 3, 0}, {2, 9, 3}, {2, 8, 9}, {8, 11, 9}},
    {{2, 3, 1}, {2, 7, 3}, {7, 9, 3}, {7, 11, 9}},
    {{3, 0, 9}, {0, 11, 9}, {0, 7, 11}},
    {{0, 5, 9}, {0, 9, 4}},
    {{5, 9, 1}, {9, 2, 1}, {9, 4, 2}},
    {{0, 5, 9}, {0, 9, 4}, {2, 7, 8}},
    {{5, 8, 1}, {5, 9, 8}, {9, 7, 8}, {9, 4, 7}},
    {{1, 6, 0}, {6, 4, 0}, {6, 9, 4}},
    {{2, 6, 9}, {2, 9, 4}},
    {{1, 6, 0}, {6, 4, 0}, {6, 9, 4}, {2, 7, 8}},
    {{8, 6, 7}, {6, 4, 7}, {6, 9, 4}},
    {{0, 5, 9}, {0, 9, 4}, {6, 8, 11}},
    {{5, 9, 1}, {9, 2, 1}, {9, 4, 2}, {6, 8, 11}},
    {{0, 5, 9}, {0, 9, 4}, {6, 2, 7}, {6, 7, 11}},
    {{1, 5, 9}, {1, 9, 4}, {1, 4, 7}, {1, 7, 11}, {1, 11, 6}},
    {{0, 1, 8}, {0, 8, 4}, {4, 8, 11}, {4, 11, 9}},
    {{11, 9, 8}, {9, 2, 8}, {9, 4, 2}},
    {{1, 2, 7}, {1, 7, 11}, {1, 11, 9}, {1, 9, 4}, {1, 4, 0}},
    {{7, 11, 9}, {7, 9, 4}},
    {{7, 4, 10}, {5, 9, 3}},
    {{2, 1, 0}, {7, 4, 10}, {5, 9, 3}},
    {{2, 4, 10}, {2, 10, 8}, {5, 9, 3}},
    {{4, 10, 0}, {10, 1, 0}, {10, 8, 1}, {5, 9, 3}},
    {{6, 9, 3}, {6, 3, 1}, {7, 4, 10}},
    {{0, 2, 3}, {2, 9, 3}, {2, 6, 9}, {7, 4, 10}},
    {{6, 9, 3}, {6, 3, 1}, {2, 4, 10}, {2, 10, 8}},
    {{0, 4, 10}, {0, 10, 8}, {0, 8, 6}, {0, 6, 9}, {0, 9, 3}},
    {{6, 8, 11}, {7, 4, 10}, {5, 9, 3}},
    {{2, 1, 0}, {6, 8, 11}, {7, 4, 10}, {5, 9, 3}},
    {{11, 6, 10}, {6, 4, 10}, {6, 2, 4}, {5, 9, 3}},
    {{6, 10, 11}, {6, 4, 10}, {6, 1, 4}, {1, 0, 4}, {5, 9, 3}},
    {{9, 3, 11}, {3, 8, 11}, {3, 1, 8}, {7, 4, 10}},
    {{2, 3, 0}, {2, 9, 3}, {2, 8, 9}, {8, 11, 9}, {7, 4, 10}},
    {{11, 9, 3}, {11, 3, 1}, {11, 1, 2}, {11, 2, 4}, {11, 4, 10}},
    {{0, 4, 10}, {0, 10, 11}, {0, 11, 9}, {0, 9, 3}},
    {{10, 7, 9}, {7, 5, 9}, {7, 0, 5}},
    {{9, 1, 5}, {9, 2, 1}, {9, 10, 2}, {10, 7, 2}},
    {{9, 10, 8}, {9, 8, 5}, {5, 8, 2}, {5, 2, 0}},
    {{9, 10, 5}, {10, 1, 5}, {10, 8, 1}},
    {{1, 7, 0}, {1, 6, 7}, {6, 10, 7}, {6, 9, 10}},
    {{7, 2, 10}, {2, 9, 10}, {2, 6, 9}},
    {{0, 1, 6}, {0, 6, 9}, {0, 9, 10}, {0, 10, 8}, {0, 8, 2}},
    {{8, 6, 9}, {8, 9, 10}},
    {{10, 7, 9}, {7, 5, 9}, {7, 0, 5}, {6, 8, 11}},
    {{10, 5, 9}, {10, 1, 5}, {10, 7, 1}, {7, 2, 1}, {6, 8, 11}},
    {{10, 11, 6}, {10, 6, 2}, {10, 2, 0}, {10, 0, 5}, {10, 5, 9}},
    {{1, 5, 9}, {1, 9, 10}, {1, 10, 11}, {1, 11, 6}},
    {{9, 10, 7}, {9, 7, 0}, {9, 0, 1}, {9, 1, 8}, {9, 8, 11}},
    {{2, 8, 11}, {2, 11, 9}, {2, 9, 10}, {2, 10, 7}},
    {{2, 0, 1}, {11, 9, 10}},
    {{11, 9, 10}},
    {{11, 10, 9}},
    {{2, 1, 0}, {11, 10, 9}},
    {{2, 7, 8}, {11, 10, 9}},
    {{1, 0, 7}, {1, 7, 8}, {11, 10, 9}},
    {{1, 6, 5}, {11, 10, 9}},
    {{2, 6, 5}, {2, 5, 0}, {11, 10, 9}},
    {{2, 7, 8}, {1, 6, 5}, {11, 10, 9}},
    {{6, 5, 8}, {5, 7, 8}, {5, 0, 7}, {11, 10, 9}},
    {{8, 10, 9}, {8, 9, 6}},
    {{6, 8, 10}, {6, 10, 9}, {2, 1, 0}},
    {{10, 9, 7}, {9, 2, 7}, {9, 6, 2}},
    {{0, 6, 1}, {0, 7, 6}, {7, 9, 6}, {7, 10, 9}},
    {{5, 1, 9}, {1, 10, 9}, {1, 8, 10}},
    {{10, 9, 5}, {10, 5, 8}, {8, 5, 0}, {8, 0, 2}},
    {{5, 10, 9}, {5, 7, 10}, {5, 1, 7}, {1, 2, 7}},
    {{9, 5, 10}, {5, 7, 10}, {5, 0, 7}},
    {{0, 3, 4}, {11, 10, 9}},
    {{2, 1, 3}, {2, 3, 4}, {11, 10, 9}},
    {{2, 7, 8}, {0, 3, 4}, {11, 10, 9}},
    {{7, 8, 4}, {8, 3, 4}, {8, 1, 3}, {11, 10, 9}},
    {{1, 6, 5}, {0, 3, 4}, {11, 10, 9}},
    {{3, 4, 5}, {4, 6, 5}, {4, 2, 6}, {11, 10, 9}},
    {{2, 7, 8}, {1, 6, 5}, {0, 3, 4}, {11, 10, 9}},
    {{8, 4, 7}, {8, 3, 4}, {8, 6, 3}, {6, 5, 3}, {11, 10, 9}},
    {{6, 8, 10}, {6, 10, 9}, {0, 3, 4}},
    {{2, 1, 3}, {2, 3, 4}, {6, 8, 10}, {6, 10, 9}},
    {{10, 9, 7}, {9, 2, 7}, {9, 6, 2}, {0, 3, 4}},
    {{7, 10, 9}, {7, 9, 6}, {7, 6, 1}, {7, 1, 3}, {7, 3, 4}},
    {{5, 1, 9}, {1, 10, 9}, {1, 8, 10}, {0, 3, 4}},
    {{5, 3, 4}, {5, 4, 2}, {5, 2, 8}, {5, 8, 10}, {5, 10, 9}},
    {{1, 9, 5}, {1, 10, 9}, {1, 2, 10}, {2, 7, 10}, {0, 3, 4}},
    {{5, 3, 4}, {5, 4, 7}, {5, 7, 10}, {5, 10, 9}},
    {{7, 4, 9}, {7, 9, 11}},
    {{7, 4, 9}, {7, 9, 11}, {2, 1, 0}},
    {{8, 2, 11}, {2, 9, 11}, {2, 4, 9}},
    {{1, 0, 4}, {1, 4, 8}, {8, 4, 9}, {8, 9, 11}},
    {{7, 4, 9}, {7, 9, 11}, {1, 6, 5}},
    {{2, 6, 5}, {2, 5, 0}, {11, 7, 4}, {11, 4, 9}},
    {{8, 2, 11}, {2, 9, 11}, {2, 4, 9}, {1, 6, 5}},
    {{8, 6, 5}, {8, 5, 0}, {8, 0, 4}, {8, 4, 9}, {8, 9, 11}},
    {{7, 4, 8}, {4, 6, 8}, {4, 9, 6}},
    {{7, 4, 8}, {4, 6, 8}, {4, 9, 6}, {2, 1, 0}},
    {{2, 4, 9}, {2, 9, 6}},
    {{0, 4, 1}, {4, 6, 1}, {4, 9, 6}},
    {{1, 9, 5}, {1, 8, 9}, {8, 4, 9}, {8, 7, 4}},
    {{8, 7, 4}, {8, 4, 9}, {8, 9, 5}, {8, 5, 0}, {8, 0, 2}},
    {{1, 2, 5}, {2, 9, 5}, {2, 4, 9}},
    {{0, 4, 9}, {0, 9, 5}},
    {{9, 11, 3}, {11, 0, 3}, {11, 7, 0}},
    {{1, 7, 2}, {1, 3, 7}, {3, 11, 7}, {3, 9, 11}},
    {{0, 8, 2}, {0, 11, 8}, {0, 3, 11}, {3, 9, 11}},
    {{11, 8, 9}, {8, 3, 9}, {8, 1, 3}},
    {{9, 11, 3}, {11, 0, 3}, {11, 7, 0}, {1, 6, 5}},
    {{3, 9, 11}, {3, 11, 7}, {3, 7, 2}, {3, 2, 6}, {3, 6, 5}},
    {{9, 0, 3}, {9, 2, 0}, {9, 11, 2}, {11, 8, 2}, {1, 6, 5}},
    {{8, 6, 5}, {8, 5, 3}, {8, 3, 9}, {8, 9, 11}},
    {{0, 3, 9}, {0, 9, 7}, {7, 9, 6}, {7, 6, 8}},
    {{7, 2, 1}, {7, 1, 3}, {7, 3, 9}, {7, 9, 6}, {7, 6, 8}},
    {{3, 9, 0}, {9, 2, 0}, {9, 6, 2}},
    {{1, 3, 9}, {1, 9, 6}},
    {{9, 5, 1}, {9, 1, 8}, {9, 8, 7}, {9, 7, 0}, {9, 0, 3}},
    {{2, 8, 7}, {5, 3, 9}},
    {{2, 0, 3}, {2, 3, 9}, {2, 9, 5}, {2, 5, 1}},
    {{5, 3, 9}},
    {{5, 11, 10}, {5, 10, 3}},
    {{11, 10, 3}, {11, 3, 5}, {2, 1, 0}},
    {{11, 10, 3}, {11, 3, 5}, {2, 7, 8}},
    {{8, 1, 0}, {8, 0, 7}, {5, 11, 10}, {5, 10, 3}},
    {{11, 10, 6}, {10, 1, 6}, {10, 3, 1}},
    {{2, 3, 0}, {2, 6, 3}, {6, 10, 3}, {6, 11, 10}},
    {{11, 10, 6}, {10, 1, 6}, {10, 3, 1}, {2, 7, 8}},
    {{6, 11, 10}, {6, 10, 3}, {6, 3, 0}, {6, 0, 7}, {6, 7, 8}},
    {{6, 8, 5}, {8, 3, 5}, {8, 10, 3}},
    {{6, 8, 5}, {8, 3, 5}, {8, 10, 3}, {2, 1, 0}},
    {{3, 5, 6}, {3, 6, 10}, {10, 6, 2}, {10, 2, 7}},
    {{6, 1, 0}, {6, 0, 7}, {6, 7, 10}, {6, 10, 3}, {6, 3, 5}},
    {{8, 10, 3}, {8, 3, 1}},
    {{2, 8, 0}, {8, 3, 0}, {8, 10, 3}},
    {{7, 10, 2}, {10, 1, 2}, {10, 3, 1}},
    {{7, 10, 3}, {7, 3, 0}},
    {{4, 0, 10}, {0, 11, 10}, {0, 5, 11}},
    {{2, 1, 5}, {2, 5, 4}, {4, 5, 11}, {4, 11, 10}},
    {{4, 0, 10}, {0, 11, 10}, {0, 5, 11}, {2, 7, 8}},
    {{4, 7, 8}, {4, 8, 1}, {4, 1, 5}, {4, 5, 11}, {4, 11, 10}},
    {{6, 0, 1}, {6, 4, 0}, {6, 11, 4}, {11, 10, 4}},
    {{10, 4, 11}, {4, 6, 11}, {4, 2, 6}},
    {{4, 11, 10}, {4, 6, 11}, {4, 0, 6}, {0, 1, 6}, {8, 2, 7}},
    {{6, 11, 10}, {6, 10, 4}, {6, 4, 7}, {6, 7, 8}},
    {{8, 5, 6}, {8, 10, 5}, {10, 0, 5}, {10, 4, 0}},
    {{5, 6, 8}, {5, 8, 10}, {5, 10, 4}, {5, 4, 2}, {5, 2, 1}},
    {{10, 4, 0}, {10, 0, 5}, {10, 5, 6}, {10, 6, 2}, {10, 2, 7}},
    {{1, 5, 6}, {7, 10, 4}},
    {{0, 1, 4}, {1, 10, 4}, {1, 8, 10}},
    {{8, 10, 4}, {8, 4, 2}},
    {{1, 2, 7}, {1, 7, 10}, {1, 10, 4}, {1, 4, 0}},
    {{7, 10, 4}},
    {{3, 5, 4}, {5, 7, 4}, {5, 11, 7}},
    {{3, 5, 4}, {5, 7, 4}, {5, 11, 7}, {2, 1, 0}},
    {{2, 11, 8}, {2, 4, 11}, {4, 5, 11}, {4, 3, 5}},
    {{4, 3, 5}, {4, 5, 11}, {4, 11, 8}, {4, 8, 1}, {4, 1, 0}},
    {{1, 6, 11}, {1, 11, 3}, {3, 11, 7}, {3, 7, 4}},
    {{3, 0, 2}, {3, 2, 6}, {3, 6, 11}, {3, 11, 7}, {3, 7, 4}},
    {{11, 8, 2}, {11, 2, 4}, {11, 4, 3}, {11, 3, 1}, {11, 1, 6}},
    {{6, 11, 8}, {0, 4, 3}},
    {{8, 5, 6}, {8, 3, 5}, {8, 7, 3}, {7, 4, 3}},
    {{3, 7, 4}, {3, 8, 7}, {3, 5, 8}, {5, 6, 8}, {2, 1, 0}},
    {{5, 6, 3}, {6, 4, 3}, {6, 2, 4}},
    {{6, 1, 0}, {6, 0, 4}, {6, 4, 3}, {6, 3, 5}},
    {{4, 3, 7}, {3, 8, 7}, {3, 1, 8}},
    {{8, 7, 4}, {8, 4, 3}, {8, 3, 0}, {8, 0, 2}},
    {{2, 4, 3}, {2, 3, 1}},
    {{0, 4, 3}},
    {{0, 5, 11}, {0, 11, 7}},
    {{1, 5, 2}, {5, 7, 2}, {5, 11, 7}},
    {{2, 0, 8}, {0, 11, 8}, {0, 5, 11}},
    {{1, 5, 11}, {1, 11, 8}},
    {{6, 11, 1}, {11, 0, 1}, {11, 7, 0}},
    {{2, 6, 11}, {2, 11, 7}},
    {{0, 1, 6}, {0, 6, 11}, {0, 11, 8}, {0, 8, 2}},
    {{8, 6, 11}},
    {{8, 7, 6}, {7, 5, 6}, {7, 0, 5}},
    {{5, 6, 8}, {5, 8, 7}, {5, 7, 2}, {5, 2, 1}},
    {{0, 5, 6}, {0, 6, 2}},
    {{1, 5, 6}},
    {{1, 8, 7}, {1, 7, 0}},
    {{2, 8, 7}},
    {{2, 0, 1}},
    {},
}};

auto marching_cubes(
    const skimpy::Tensor<3, float>& lattice, float surface_density = 0.5) {
  SurfaceMesh mesh;

  using VertexKey = std::tuple<int, int>;
  struct VertexKeyHash {
    size_t operator()(VertexKey key) const {
      uint64_t a = static_cast<uint64_t>(std::get<0>(key));
      uint64_t b = static_cast<uint64_t>(std::get<1>(key));
      return std::hash<uint64_t>()((a << 32) | b);
    }
  };
  std::unordered_map<VertexKey, int, VertexKeyHash> v_index;

  auto lw = lattice.shape()[0];
  auto lh = lattice.shape()[1];
  auto ld = lattice.shape()[2];

  // Helper routine to generate a slice shifted by the given deltas.
  auto shift = [&](int dx, int dy, int dz) {
    auto x0 = std::max(0, dx), x1 = std::min(lw, lw + dx);
    auto y0 = std::max(0, dy), y1 = std::min(lh, lh + dy);
    auto z0 = std::max(0, dz), z1 = std::min(ld, ld + dz);
    return skimpy::TensorSlice<3>({{x0, x1, 1}, {y0, y1, 1}, {z0, z1, 1}});
  };

  // Helper routine to map a vertex coordinate to its one-dimensional index.
  auto xyz_to_pos = [&](Vec3i xyz) {
    return xyz[0] + xyz[1] * lw + xyz[2] * lw * lh;
  };

  auto pos_to_xyz = [](Vec3i whd, int pos) {
    auto [w, h, d] = whd;
    return Vec3i{pos % w, (pos / w) % h, pos / (w * h)};
  };

  {
    auto x0 = lattice.get(shift(-1, 0, 0)).array();
    auto x1 = lattice.get(shift(1, 0, 0)).array();
    auto y0 = lattice.get(shift(0, -1, 0)).array();
    auto y1 = lattice.get(shift(0, 1, 0)).array();
    auto z0 = lattice.get(shift(0, 0, -1)).array();
    auto z1 = lattice.get(shift(0, 0, 1)).array();

    // Emit vertices along edges in the x direction.
    array_walk(
        [&](auto pos, float mid) {
          auto xyz = pos_to_xyz({lw - 1, lh, ld}, pos);
          auto s = xyz_to_pos(xyz), e = xyz_to_pos(add(xyz, {1, 0, 0}));
          v_index[std::tuple(s, e)] = mesh.vertex_count();
          mesh.positions.push_back(xyz[0] + mid);
          mesh.positions.push_back(xyz[1]);
          mesh.positions.push_back(xyz[2]);
        },
        (x0 < surface_density) != (x1 < surface_density),
        (surface_density - x0) / (x1 - x0));
    array_walk(
        [&](auto pos, float mid) {
          auto xyz = pos_to_xyz({lw, lh - 1, ld}, pos);
          auto s = xyz_to_pos(xyz), e = xyz_to_pos(add(xyz, {0, 1, 0}));
          v_index[std::tuple(s, e)] = mesh.vertex_count();
          mesh.positions.push_back(xyz[0]);
          mesh.positions.push_back(xyz[1] + mid);
          mesh.positions.push_back(xyz[2]);
        },
        (y0 < surface_density) != (y1 < surface_density),
        (surface_density - y0) / (y1 - y0));
    array_walk(
        [&](auto pos, float mid) {
          auto xyz = pos_to_xyz({lw, lh, ld - 1}, pos);
          auto s = xyz_to_pos(xyz), e = xyz_to_pos(add(xyz, {0, 0, 1}));
          v_index[std::tuple(s, e)] = mesh.vertex_count();
          mesh.positions.push_back(xyz[0]);
          mesh.positions.push_back(xyz[1]);
          mesh.positions.push_back(xyz[2] + mid);
        },
        (z0 < surface_density) != (z1 < surface_density),
        (surface_density - z0) / (z1 - z0));
  }

  auto make_vertex_key = [&](Vec3i origin, int index) {
    auto [s, e] = kEdgeKeys.at(index);
    auto s_pos = xyz_to_pos(add(origin, s));
    auto e_pos = xyz_to_pos(add(origin, e));
    if (s_pos < e_pos) {
      return std::tuple(s_pos, e_pos);
    } else {
      return std::tuple(e_pos, s_pos);
    }
  };

  {
    auto l_000 = lattice.get(shift(-1, -1, -1)).array();
    auto l_100 = lattice.get(shift(1, -1, -1)).array();
    auto l_010 = lattice.get(shift(-1, 1, -1)).array();
    auto l_110 = lattice.get(shift(1, 1, -1)).array();
    auto l_001 = lattice.get(shift(-1, -1, 1)).array();
    auto l_101 = lattice.get(shift(1, -1, 1)).array();
    auto l_011 = lattice.get(shift(-1, 1, 1)).array();
    auto l_111 = lattice.get(shift(1, 1, 1)).array();

    auto mask = skimpy::splat<int>(l_000 >= surface_density, 0b00000001, 0) |
                skimpy::splat<int>(l_100 >= surface_density, 0b00000010, 0) |
                skimpy::splat<int>(l_010 >= surface_density, 0b00000100, 0) |
                skimpy::splat<int>(l_110 >= surface_density, 0b00001000, 0) |
                skimpy::splat<int>(l_001 >= surface_density, 0b00010000, 0) |
                skimpy::splat<int>(l_101 >= surface_density, 0b00100000, 0) |
                skimpy::splat<int>(l_011 >= surface_density, 0b01000000, 0) |
                skimpy::splat<int>(l_111 >= surface_density, 0b10000000, 0);

    // Emit triangles for each cell that intersects with the surface.
    array_walk(
        [&](auto pos, int mask) {
          Vec3i xyz = pos_to_xyz({lw - 1, lh - 1, ld - 1}, pos);

          // Emit triangles based on mask type.
          for (const auto& triangle : kTrianglesIndex.at(mask)) {
            auto [i, j, k] = triangle;
            auto key_i = make_vertex_key(xyz, i);
            auto key_j = make_vertex_key(xyz, j);
            auto key_k = make_vertex_key(xyz, k);

            auto print_stuff = [&] {
              auto [i, j, k] = triangle;

              fmt::print("x={}, y={}, z={}\n", xyz[0], xyz[1], xyz[2]);
              fmt::print("i={}, j={}, k={}\n", i, j, k);
              fmt::print("mask={}\n", mask);
              fmt::print("key_i={}\n", key_i);
              fmt::print("key_j={}\n", key_j);
              fmt::print("key_k={}\n", key_k);

              auto v000 = lattice.get({xyz[0], xyz[1], xyz[2]});
              auto v100 = lattice.get({xyz[0] + 1, xyz[1], xyz[2]});
              auto v010 = lattice.get({xyz[0], xyz[1] + 1, xyz[2]});
              auto v110 = lattice.get({xyz[0] + 1, xyz[1] + 1, xyz[2]});
              auto v001 = lattice.get({xyz[0], xyz[1], xyz[2] + 1});
              auto v101 = lattice.get({xyz[0] + 1, xyz[1], xyz[2] + 1});
              auto v011 = lattice.get({xyz[0], xyz[1] + 1, xyz[2] + 1});
              auto v111 = lattice.get({xyz[0] + 1, xyz[1] + 1, xyz[2] + 1});

              fmt::print("v000={}\n", v000);
              fmt::print("v100={}\n", v100);
              fmt::print("v010={}\n", v010);
              fmt::print("v110={}\n", v110);
              fmt::print("v001={}\n", v001);
              fmt::print("v101={}\n", v101);
              fmt::print("v011={}\n", v011);
              fmt::print("v111={}\n", v111);
            };

            if (!v_index.count(key_i)) {
              fmt::print("C1\n");
              print_stuff();
            }
            if (!v_index.count(key_j)) {
              fmt::print("C2\n");
              print_stuff();
            }
            if (!v_index.count(key_k)) {
              fmt::print("C3\n");
              print_stuff();
            }
            mesh.triangles.push_back(v_index.at(key_i));
            mesh.triangles.push_back(v_index.at(key_j));
            mesh.triangles.push_back(v_index.at(key_k));
          }
        },
        mask != 0b11111111 && mask != 0b00000000,
        mask);
  }

  return mesh;
}

}  // namespace skimpy_3d
