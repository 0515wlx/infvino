// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/InferenceEngine.hpp"

#include <chrono>
#include <stdexcept>

namespace infvino
{

namespace
{
using Clock = std::chrono::steady_clock;
inline double ms_since(const Clock::time_point & t0)
{
  return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
}  // namespace

InferenceEngine::Session::Session(
  std::shared_ptr<ClBackend> backend, std::shared_ptr<Preprocessor> pre,
  std::shared_ptr<Decoder> decoder, ModelInfo info)
: backend_(std::move(backend)),
  pre_(std::move(pre)),
  decoder_(std::move(decoder)),
  info_(std::move(info))
{
}

InferResult InferenceEngine::Session::infer(const cv::Mat & bgr)
{
  const auto t_start = Clock::now();

  LetterboxInfo lb;
  const auto   t_pre = Clock::now();
  const cv::Mat blob = (*pre_)(bgr, &lb);
  const double  pre_ms = ms_since(t_pre);

  const auto t_infer = Clock::now();
  const size_t numel = static_cast<size_t>(blob.total()) * blob.channels();
  std::vector<Tensor> outputs = backend_->infer(blob.ptr<float>(0), numel);
  const double infer_ms = ms_since(t_infer);

  const auto t_post = Clock::now();
  InferResult result = decoder_->decode(outputs, info_, lb);
  const double post_ms = ms_since(t_post);

  result.input2image = cv::Matx33f::eye();
  result.input2image(0, 0) = 1.f / lb.scale;
  result.input2image(1, 1) = 1.f / lb.scale;
  result.input2image(0, 2) = -static_cast<float>(lb.pad_left) / lb.scale;
  result.input2image(1, 2) = -static_cast<float>(lb.pad_top) / lb.scale;

  result.preprocess_ms  = pre_ms;
  result.infer_ms       = infer_ms;
  result.postprocess_ms = post_ms;
  result.total_ms       = ms_since(t_start);
  return result;
}

InferenceEngine::InferenceEngine(const ModelInfo & info) : info_(info)
{
  backend_ = std::make_shared<ClBackend>(info_);

  std::string err;
  decoder_ = Decoder::create(info_, backend_->outputShapes(), &err);
  if (!decoder_)
    throw std::runtime_error("InferenceEngine: decoder not available: " + err);

  pre_ = std::make_shared<Preprocessor>(PreprocessConfig{
    info_.input_size, info_.letterbox, info_.to_rgb, info_.normalize, info_.mean, info_.std});
}

InferResult InferenceEngine::infer(const cv::Mat & bgr)
{
  if (!default_session_) default_session_ = createSession();
  return default_session_->infer(bgr);
}

std::unique_ptr<InferenceEngine::Session> InferenceEngine::createSession()
{
  return std::unique_ptr<Session>(new Session(backend_, pre_, decoder_, info_));
}

}  // namespace infvino
