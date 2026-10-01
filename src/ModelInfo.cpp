// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/ModelInfo.hpp"

#include <yaml-cpp/yaml.h>

#include <stdexcept>

namespace infvino
{

namespace
{
template<typename T>
T getOr(const YAML::Node & n, const char * key, const T & fallback)
{
  return (n && n[key]) ? n[key].as<T>() : fallback;
}
} // namespace

ModelInfo ModelInfo::fromYaml(const std::string & yaml_path, const std::string & key)
{
  const YAML::Node root = YAML::LoadFile(yaml_path);
  YAML::Node node;
  if (!key.empty())
  {
    if (!root["models"] || !root["models"][key])
      throw std::runtime_error("ModelInfo: key '" + key + "' not found in " + yaml_path);
    node = root["models"][key];
  }
  else
  {
    const std::string def = root["default"] ? root["default"].as<std::string>() : "";
    if (def.empty() || !root["models"] || !root["models"][def])
      throw std::runtime_error("ModelInfo: no default model in " + yaml_path);
    node = root["models"][def];
  }

  ModelInfo info;
  info.path = node["path"] ? node["path"].as<std::string>() : info.path;
  info.plan = node["plan"] ? node["plan"].as<std::string>() : info.plan;
  info.kernel_dir = getOr<std::string>(node, "kernel_dir", info.kernel_dir);
  info.task = node["task"] ? taskFromString(node["task"].as<std::string>()) : info.task;
  info.num_classes = getOr<int>(node, "num_classes", info.num_classes);
  info.conf = getOr<float>(node, "conf", info.conf);
  info.iou = getOr<float>(node, "iou", info.iou);
  info.max_det = getOr<int>(node, "max_det", info.max_det);
  info.end2end = getOr<bool>(node, "end2end", info.end2end);
  info.device = getOr<std::string>(node, "device", info.device);
  info.allow_cpu_fallback = getOr<bool>(node, "allow_cpu_fallback", info.allow_cpu_fallback);
  info.letterbox = getOr<bool>(node, "letterbox", info.letterbox);
  info.to_rgb = getOr<bool>(node, "to_rgb", info.to_rgb);
  info.normalize = getOr<bool>(node, "normalize", info.normalize);
  if (node["mean"]) info.mean = node["mean"].as<std::vector<float>>();
  if (node["std"]) info.std = node["std"].as<std::vector<float>>();

  if (node["imgsz"])
  {
    const auto v = node["imgsz"].as<std::vector<int>>();
    if (v.size() == 1) info.input_size = {v[0], v[0]};
    else if (v.size() >= 2) info.input_size = {v[0], v[1]};
  }
  if (node["kpt_shape"]) info.kpt_shape = node["kpt_shape"].as<std::vector<int>>();
  if (node["names"])
  {
    try
    {
      info.names = node["names"].as<std::vector<std::string>>();
    }
    catch (const std::exception &)
    {
      info.names.clear(); // names 可能是 {0: 'x'} 形式，MVP 先忽略
    }
  }
  return info;
}

} // namespace infvino
