// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 设备信息（OpenCL）。替代原 OpenVINO 的设备探测：不再硬编码 GPU，
// 由 gk::ClRuntime 自动选择（优先 GPU 平台，无 GPU 时退到任一可用设备）。
#ifndef INFVINO__DEVICE_HPP_
#define INFVINO__DEVICE_HPP_

#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"

namespace infvino
{

struct DeviceInfo
{
  std::vector<std::string> available;         /**< 枚举到的所有 OpenCL 设备名 */
  std::string              requested;         /**< 用户请求 (AUTO/GPU/CPU) */
  std::string              resolved;          /**< 实际使用设备名 */
  std::string              full_name;         /**< 设备全名 */
  bool                     fell_back_to_cpu = false; /**< 是否退到非 GPU 设备 */
  std::string              opencl_version;
  unsigned                 eu         = 0;    /**< compute units */
  unsigned                 clock_mhz  = 0;
};

/** @brief 由已选中的运行设备 + 用户请求构造 DeviceInfo。 */
DeviceInfo describeDevice(const gk::DeviceInfo & used, const std::string & requested);

}  // namespace infvino

#endif  // INFVINO__DEVICE_HPP_
