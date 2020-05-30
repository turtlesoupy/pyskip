#pragma once

#include <string>
#include <unordered_map>
#include <mutex>

namespace skimpy {

using ConfigTypes = std::variant<std::string, long, double, bool>;

class GlobalConfig {
 public:
  static inline GlobalConfig& get() {
    static GlobalConfig instance;
    return instance;
  }

  template <typename Val>
  Val getConfigVal(const std::string &key, Val fallback) {
      const std::lock_guard<std::mutex> lock(lock_);
      const auto val = config_map_.find(key);
      if (val == config_map_.end()) {
          return fallback;
      } else {
          return std::get<Val>(val->second);
      }
  }

  template <typename Val>
  void setConfigVal(const std::string &key, Val val) {
      const std::lock_guard<std::mutex> lock(lock_);
      config_map_[key] = val;
  }

  void clearConfigVal(const std::string &key) {
      const std::lock_guard<std::mutex> lock(lock_);
      config_map_.erase(key);
  }

  std::unordered_map<std::string, ConfigTypes> getConfigMap() {
    const std::lock_guard<std::mutex> lock(lock_);
    return config_map_;
  }

  void setConfigMap(std::unordered_map<std::string, ConfigTypes> &map) {
    const std::lock_guard<std::mutex> lock(lock_);
    config_map_ = map;
  }

 private:
  GlobalConfig() {}
  std::unordered_map<std::string, ConfigTypes> config_map_;
  std::mutex lock_;
};

}  // namespace skimpy