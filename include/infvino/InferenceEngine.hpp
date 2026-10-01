// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#ifndef INFVINO__INFERENCE_ENGINE_HPP_
#define INFVINO__INFERENCE_ENGINE_HPP_

#include <memory>
#include <mutex>

#include "infvino/ClBackend.hpp"
#include "infvino/Decoder.hpp"
#include "infvino/Preprocess.hpp"
#include "infvino/Tensor.hpp"
#include "infvino/Types.hpp"

namespace infvino
{

/**
 * @brief 推理门面。纯 C++（OpenCV + OpenCL），不依赖 ROS / OpenVINO。
 *
 * 用法：
 * @code
 *   infvino::ModelInfo info = infvino::ModelInfo::fromYaml("config/models.yaml", "yolov8n-pose");
 *   infvino::InferenceEngine engine(info);
 *   cv::Mat bgr = cv::imread("x.jpg");
 *   auto result = engine.infer(bgr);              // 便捷（内部串行）
 *   auto session = engine.createSession();        // 每线程/每相机一个
 * @endcode
 */
class InferenceEngine
{
public:
  explicit InferenceEngine(const ModelInfo & info);

  /**
   * @brief 独立推理会话。共享后端（权重常驻），每次推理串行访问设备缓冲区。
   */
  class Session
  {
  public:
    InferResult infer(const cv::Mat & bgr);

  private:
    friend class InferenceEngine;
    Session(
      std::shared_ptr<ClBackend> backend, std::shared_ptr<Preprocessor> pre,
      std::shared_ptr<Decoder> decoder, ModelInfo info);

    std::shared_ptr<ClBackend>    backend_;
    std::shared_ptr<Preprocessor> pre_;
    std::shared_ptr<Decoder>      decoder_;
    ModelInfo                     info_;
  };

  /** @brief 便捷推理（内部单个 session）。 */
  InferResult infer(const cv::Mat & bgr);

  std::unique_ptr<Session> createSession();

  const DeviceInfo & device() const { return backend_->device(); }
  const ModelInfo &  info() const { return info_; }

private:
  ModelInfo                  info_;
  std::shared_ptr<ClBackend> backend_;
  std::shared_ptr<Preprocessor> pre_;
  std::shared_ptr<Decoder>   decoder_;
  std::unique_ptr<Session>   default_session_;
};

}  // namespace infvino

#endif  // INFVINO__INFERENCE_ENGINE_HPP_
