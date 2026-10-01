// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/Device.hpp"

#include <algorithm>
#include <cctype>

namespace infvino
{

DeviceInfo describeDevice(const gk::DeviceInfo & used, const std::string & requested)
{
  DeviceInfo info;
  info.requested = requested.empty() ? std::string("AUTO") : requested;
  info.resolved = used.name;
  info.full_name = used.name + " [" + used.vendor + " " + used.opencl_version + "]";
  info.opencl_version = used.opencl_version;
  info.eu = static_cast<unsigned>(used.eu);
  info.clock_mhz = static_cast<unsigned>(used.clock_mhz);

  for (const auto & d : gk::ClRuntime::enumerate()) info.available.push_back(d.name);

  // ClRuntime 优先选 GPU 平台；只有退到非 GPU 设备时才算 fallback。
  info.fell_back_to_cpu = !used.is_gpu;
  return info;
}

}  // namespace infvino
