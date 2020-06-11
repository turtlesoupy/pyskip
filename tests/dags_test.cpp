#define CATCH_CONFIG_MAIN

#include <fmt/format.h>

#include <catch2/catch.hpp>
#include <pyskip/detail/dags.hpp>

using namespace pyskip::detail::dags;

TEST_CASE("Test rooted graph creation and allocation", "[dags]") {
  Graph<2, int> graph;

  auto root = graph.emplace(0);

  // Generate some kind of DAG (node i points to 2 * i + 1 and 2 * i + 2).
  dfs(root, [&](auto node, auto& q) {
    auto i = node->data;
    if (i >= 10) {
      return;
    }
    if (!node->deps[0]) {
      q.push_back(node->deps[0] = graph.emplace(2 * i + 1));
    }
    if (!node->deps[1]) {
      q.push_back(node->deps[1] = graph.emplace(2 * i + 2));
    }
  });

  // Validate the DAG.
  REQUIRE(graph.size() == 21);
  dfs(root, [](auto node, auto& q) {
    if (node->data >= 10) {
      REQUIRE(!node->deps[0]);
      REQUIRE(!node->deps[1]);
    } else {
      REQUIRE(node->deps[0]);
      REQUIRE(node->deps[0]->data == 2 * node->data + 1);
      REQUIRE(node->deps[1]);
      REQUIRE(node->deps[1]->data == 2 * node->data + 2);
      q.push_back(node->deps[0]);
      q.push_back(node->deps[1]);
    }
  });

  // Mutate a bunch of nodes in the DAG.
  dfs(root, [&](auto node, auto& q) {
    if (node->deps[0]) {
      node->deps[0] = nullptr;
    }
    if (node->deps[1]) {
      q.push_back(node->deps[1]);
    }
  });

  // Validate the DAG.
  REQUIRE(graph.size() == 4);
  REQUIRE(root->data == 0);
  REQUIRE(!root->deps[0]);
  REQUIRE(root->deps[1]->data == 2);
  REQUIRE(!root->deps[1]->deps[0]);
  REQUIRE(root->deps[1]->deps[1]->data == 6);
  REQUIRE(!root->deps[1]->deps[1]->deps[0]);
  REQUIRE(root->deps[1]->deps[1]->deps[1]->data == 14);
  REQUIRE(!root->deps[1]->deps[1]->deps[1]->deps[0]);
  REQUIRE(!root->deps[1]->deps[1]->deps[1]->deps[1]);

  // Try re-assiging some branches.
  root->deps[0] = graph.emplace(1);
  root->deps[0]->deps[0] = root->deps[1];
  root->deps[0]->deps[1] = root->deps[1];
  root->deps[1] = nullptr;
  REQUIRE(graph.size() == 5);
  REQUIRE(root->deps[0]->deps[0]->deps[1]);
  REQUIRE(root->deps[0]->deps[1]->data == 2);
  REQUIRE(root->deps[0]->deps[1]->deps[1]->data == 6);
  REQUIRE(root->deps[0]->deps[1]->deps[1]->deps[1]->data == 14);
  REQUIRE(!root->deps[1]);
}
