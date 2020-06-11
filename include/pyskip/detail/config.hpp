#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>

// TODO(taylorgordon): Move this out of "detail" since it's external facing.
namespace pyskip::detail::config {

using ConfigTypes = std::variant<std::string, int64_t, double, bool>;

class GlobalConfig {
 public:
  static inline GlobalConfig& get() {
    static GlobalConfig instance;
    return instance;
  }

  template <typename Val>
  Val get_config_val(const std::string& key, Val fallback) {
    const std::lock_guard<std::mutex> lock(lock_);
    const auto val = config_map_.find(key);
    if (val == config_map_.end()) {
      return fallback;
    } else if (!std::holds_alternative<Val>(val->second)) {
      return fallback;
    } else {
      return std::get<Val>(val->second);
    }
  }

  template <typename Val>
  void set_config_val(const std::string& key, Val val) {
    const std::lock_guard<std::mutex> lock(lock_);
    config_map_[key] = val;
  }

  void clear_config_val(const std::string& key) {
    const std::lock_guard<std::mutex> lock(lock_);
    config_map_.erase(key);
  }

  std::unordered_map<std::string, ConfigTypes> get_config_map() {
    const std::lock_guard<std::mutex> lock(lock_);
    return config_map_;
  }

  void set_config_map(std::unordered_map<std::string, ConfigTypes> map) {
    const std::lock_guard<std::mutex> lock(lock_);
    config_map_ = std::move(map);
  }

 private:
  GlobalConfig() {}
  std::unordered_map<std::string, ConfigTypes> config_map_;
  std::mutex lock_;
};

template <typename Val>
inline auto get_or(const std::string& key, Val fallback) {
  return GlobalConfig::get().get_config_val(key, fallback);
}

template <typename Val>
inline void set(const std::string& key, Val val) {
  GlobalConfig::get().set_config_val(key, val);
}

inline void clear(const std::string& key) {
  GlobalConfig::get().clear_config_val(key);
}

}  // namespace pyskip::detail::config
