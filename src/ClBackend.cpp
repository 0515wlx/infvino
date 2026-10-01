// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/ClBackend.hpp"

#include <chrono>
#include <stdexcept>

#include "infvino/Half.hpp"

namespace infvino
{

ClBackend::ClBackend(const ModelInfo & info, bool profiling)
{
  if (info.plan.empty())
    throw std::runtime_error(
      "ClBackend: ModelInfo.plan is empty. Generate it first with "
      "scripts/onnx2plan.py --onnx <model.onnx> --out-dir models");

  const std::string kernel_dir = info.kernel_dir.empty() ? INFVINO_KERNEL_DIR : info.kernel_dir;
  model_ = std::make_unique<gk::PlanModel>(info.plan, kernel_dir, -1, 0, profiling);

  input_dims_ = model_->inputDims();
  output_shapes_.clear();
  for (size_t i = 0; i < model_->outputCount(); ++i)
    output_shapes_.push_back(model_->outputDims(i));

  device_ = describeDevice(model_->device(), info.device);
}

std::vector<Tensor> ClBackend::infer(const float * input_f32, size_t numel)
{
  std::lock_guard<std::mutex> lock(mutex_);

  const size_t expect = model_->inputNumel();
  if (numel != expect)
    throw std::runtime_error(
      "ClBackend: input numel mismatch: got " + std::to_string(numel) + ", expected " +
      std::to_string(expect));

  in_f16_.resize(numel);
  for (size_t i = 0; i < numel; ++i) in_f16_[i] = gk::f32_to_f16(input_f32[i]);

  const auto t0 = std::chrono::steady_clock::now();
  model_->setInput(in_f16_.data());
  model_->run();
  last_infer_ms_ =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::vector<Tensor> outputs;
  outputs.reserve(model_->outputCount());
  for (size_t i = 0; i < model_->outputCount(); ++i)
  {
    const size_t n = model_->outputNumel(i);
    out_f16_.resize(n);
    model_->readOutput(i, out_f16_.data());

    Tensor t;
    t.shape = model_->outputDims(i);
    t.data.resize(n);
    for (size_t j = 0; j < n; ++j) t.data[j] = gk::f16_to_f32(out_f16_[j]);
    outputs.push_back(std::move(t));
  }
  return outputs;
}

}  // namespace infvino
