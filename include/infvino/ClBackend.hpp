// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// ClBackend —— 基于自研 OpenCL kernel（gk::PlanModel）的推理后端。
// 不再依赖 OpenVINO。输入为预处理后的 f32 NCHW blob，输出为 f32 张量。
#ifndef INFVINO__CL_BACKEND_HPP_
#define INFVINO__CL_BACKEND_HPP_

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "infvino/Device.hpp"
#include "infvino/ModelInfo.hpp"
#include "infvino/PlanModel.hpp"
#include "infvino/Tensor.hpp"

namespace infvino
{

/**
 * @brief 自研 OpenCL 后端：加载 onnx2plan 生成的执行计划，跑完整网络。
 *
 * - 计划 + 权重由 `scripts/onnx2plan.py` 离线生成（见 ModelInfo::plan）。
 * - 权重常驻设备；每次 infer 写入 fp16 输入、执行、读回 fp16 输出并转 f32。
 * - 内部用互斥锁串行化 run（设备缓冲区共享），因此多个 Session 可安全复用同一后端。
 */
class ClBackend
{
public:
  explicit ClBackend(const ModelInfo & info, bool profiling = false);

  const std::vector<int64_t> &              inputDims() const { return input_dims_; }
  const std::vector<std::vector<int64_t>> & outputShapes() const { return output_shapes_; }

  /** @brief 执行一次。input 为 f32 NCHW blob（numel = prod(inputDims())）。 */
  std::vector<Tensor> infer(const float * input_f32, size_t numel);

  const DeviceInfo & device() const { return device_; }
  double             lastInferMs() const { return last_infer_ms_; }

private:
  std::unique_ptr<gk::PlanModel>    model_;
  std::vector<int64_t>              input_dims_;
  std::vector<std::vector<int64_t>> output_shapes_;
  DeviceInfo                        device_;

  std::vector<uint16_t> in_f16_;
  std::vector<uint16_t> out_f16_;
  double                last_infer_ms_{0.0};
  std::mutex            mutex_;
};

}  // namespace infvino

#endif  // INFVINO__CL_BACKEND_HPP_
