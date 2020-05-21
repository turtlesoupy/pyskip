#pragma once

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "errors.hpp"
#include "util.hpp"

namespace skimpy::detail::dags {

template <int k, typename T>
struct SharedNode {
  static constexpr int deps_count = k;
  using Ptr = std::shared_ptr<SharedNode<k, T>>;

  Ptr deps[k];
  T data;

  template <typename... Args>
  SharedNode(Args&&... args) : data(std::forward<Args>(args)...) {}

  template <typename... Args>
  static auto make_ptr(Args&&... args) {
    return std::make_shared<SharedNode<k, T>>(std::forward<Args>(args)...);
  }
};

template <typename Node, typename Fn>
inline void dfs(Node&& root, Fn&& fn) {
  std::vector<std::decay_t<Node>> queue;
  std::vector<std::decay_t<Node>> stack{std::forward<Node>(root)};
  while (stack.size()) {
    auto top = std::move(stack.back());
    stack.pop_back();
    fn(std::move(top), queue);
    for (auto it = queue.rbegin(); it != queue.rend(); ++it) {
      stack.push_back(std::move(*it));
    }
    queue.clear();
  }
}

template <int k, typename T>
class Graph;

template <int k, typename T>
class GraphNode;

template <int k, typename T>
class GraphHandle {
 public:
  GraphHandle() : graph_(nullptr), index_(-1) {}
  GraphHandle(std::nullptr_t) : GraphHandle() {}

  GraphHandle(const GraphHandle& other) : GraphHandle() {
    *this = other;
  }
  GraphHandle(GraphHandle&& other) : GraphHandle() {
    *this = std::move(other);
  }

  ~GraphHandle() {
    free();
  }

  GraphHandle<k, T>& operator=(const GraphHandle& other) {
    if (this != &other) {
      free();
      graph_ = other.graph_;
      index_ = other.index_;
      bump();
    }
    return *this;
  }
  GraphHandle<k, T>& operator=(GraphHandle&& other) {
    if (this != &other) {
      free();
      std::swap(graph_, other.graph_);
      std::swap(index_, other.index_);
    }
    return *this;
  }

  operator bool() const {
    return index_ >= 0;
  }

  GraphNode<k, T>& operator*() const {
    CHECK_STATE(index_ >= 0);
    return graph_->nodes_(index_);
  }

  GraphNode<k, T>* operator->() const {
    CHECK_STATE(index_ >= 0);
    return &graph_->nodes_.at(index_);
  }

  size_t hash() const {
    return util::hash_combine(graph_, index_);
  }

  bool operator==(const GraphHandle& other) const {
    return graph_ == other.graph_ && index_ == other.index_;
  }

 private:
  GraphHandle(Graph<k, T>* graph, int index) : graph_(graph), index_(index) {
    bump();
  }

  void bump() {
    if (index_ >= 0) {
      CHECK_STATE(graph_);
      ++graph_->refs_[index_];
    }
  }

  void free() {
    if (index_ >= 0 && --graph_->refs_[index_] == 0) {
      graph_->free(index_);
    }
    graph_ = nullptr;
    index_ = -1;
  }

  Graph<k, T>* graph_;
  int index_;

  friend class Graph<k, T>;
};

template <int k, typename T>
struct GraphNode {
  static constexpr int deps_count = k;
  using Ptr = GraphHandle<k, T>;

  Ptr deps[k];
  T data;

  void clear() {
    for (int i = 0; i < k; i += 1) {
      deps[i] = nullptr;
    }
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    data = T(std::forward<Args>(args)...);
    clear();
  }
};

template <int k, typename T>
class Graph {
 public:
  using Node = GraphNode<k, T>;
  using Handle = GraphHandle<k, T>;

  Graph() = default;

  Graph(const Graph& graph) = delete;
  Graph(Graph&& graph) = delete;

  Graph& operator=(const Graph& graph) = delete;
  Graph& operator=(Graph&& graph) = delete;

  auto size() const {
    return nodes_.size() - free_.size();
  }

  void reserve(size_t n) {
    nodes_.reserve(n);
    refs_.reserve(n);
  }

  template <typename... Args>
  auto emplace(Args&&... args) {
    auto index = nodes_.size();
    if (free_.size()) {
      nodes_.at(index = free_.back()).emplace(std::forward<Args>(args)...);
      free_.pop_back();
      CHECK_STATE(refs_.at(index) == 0);
    } else {
      nodes_.emplace_back().emplace(std::forward<Args>(args)...);
      refs_.emplace_back(0);
    }
    CHECK_STATE(index < nodes_.size());
    return Handle(this, index);
  }

 private:
  void free(int index) {
    free_.push_back(index);
    nodes_[index].clear();
  }

  std::vector<int> refs_;
  std::vector<int> free_;
  std::vector<Node> nodes_;  // must be destroyed first due to node refs

  friend class GraphHandle<k, T>;
};

}  // namespace skimpy::detail::dags

namespace std {
template <int k, typename T>
struct hash<skimpy::detail::dags::GraphHandle<k, T>> {
  size_t operator()(
      const skimpy::detail::dags::GraphHandle<k, T>& handle) const {
    return handle.hash();
  }
};
}  // namespace std
