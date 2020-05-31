#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <thread>

#include "errors.hpp"

namespace skimpy::detail::threads {

template <typename Value>
class MPMCQueue {
 public:
  MPMCQueue() : closed_(false) {}

  bool is_open() {
    std::lock_guard lock(mutex_);
    return !closed_;
  }

  bool is_empty() {
    std::lock_guard lock(mutex_);
    return queue_.empty();
  }

  auto size() {
    std::lock_guard lock(mutex_);
    return queue_.size();
  }

  void close() {
    {
      std::lock_guard lock(mutex_);
      closed_ = true;
      queue_.clear();
    }
    cv_.notify_all();
  }

  void push(Value value) {
    {
      std::lock_guard lock(mutex_);
      CHECK_STATE(!closed_);
      queue_.push_back(std::move(value));
    }
    cv_.notify_one();
  }

  std::optional<Value> pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    std::optional<Value> ret;
    while (!closed_) {
      if (!queue_.empty()) {
        ret = std::move(queue_.front());
        queue_.pop_front();
        break;
      } else {
        cv_.wait(lock);
      }
    }
    return ret;
  }

 private:
  std::mutex mutex_;
  std::condition_variable cv_;
  std::deque<Value> queue_;
  bool closed_;
};

class QueueExecutor {
 public:
  explicit QueueExecutor(size_t thread_count) : finished_workers_(0) {
    CHECK_ARGUMENT(thread_count > 0);
    for (int i = 0; i < thread_count; i += 1) {
      workers_.emplace_back([&] {
        while (auto task = task_queue_.pop()) {
          (*task)();
        }
        finished_workers_ += 1;
      });
    }
  }

  ~QueueExecutor() {
    task_queue_.close();
    for (auto& worker : workers_) {
      worker.join();
    }
  }

  auto size() {
    return task_queue_.size();
  }

  bool is_done() {
    return workers_.size() == finished_workers_;
  }

  void close() {
    return task_queue_.close();
  }

  template <typename Function>
  auto schedule(Function&& fn) {
    CHECK_STATE(task_queue_.is_open());
    auto promise = std::make_shared<std::promise<decltype(fn())>>();
    auto ret = promise->get_future();
    task_queue_.push(make_task(std::forward<Function>(fn), std::move(promise)));
    return ret;
  }

 private:
  template <typename Function>
  auto make_task(Function&& fn, std::shared_ptr<std::promise<void>>&& promise) {
    return [fn = std::forward<Function>(fn),
            promise = std::move(promise)]() mutable {
      try {
        fn();
        promise->set_value();
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    };
  }

  template <
      typename Function,
      typename PromiseType,
      typename = std::enable_if_t<!std::is_void_v<PromiseType>>>
  auto make_task(
      Function&& fn, std::shared_ptr<std::promise<PromiseType>>&& promise) {
    return [fn = std::forward<Function>(fn),
            promise = std::move(promise)]() mutable {
      try {
        promise->set_value(fn());
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    };
  }

  std::vector<std::thread> workers_;
  MPMCQueue<std::function<void()>> task_queue_;
  std::atomic<int> finished_workers_;
};

template <typename FnRange>
inline void run_in_parallel(const FnRange& fns) {
  static auto executor = [] {
    auto n = std::thread::hardware_concurrency();
    return std::make_unique<threads::QueueExecutor>(n);
  }();

  std::vector<std::future<void>> futures;
  for (const auto& fn : fns) {
    futures.push_back(executor->schedule(fn));
  }
  for (auto& future : futures) {
    future.get();
  }
}
};  // namespace skimpy::detail::threads
