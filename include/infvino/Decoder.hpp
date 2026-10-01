// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#ifndef INFVINO__DECODER_HPP_
#define INFVINO__DECODER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "infvino/ModelInfo.hpp"
#include "infvino/Preprocess.hpp"
#include "infvino/Tensor.hpp"
#include "infvino/Types.hpp"

namespace infvino
{

/**
 * @brief 输出解码抽象。不同任务/不同 ultralytics 版本各自实现，
 *        上层（InferenceEngine）不感知输出布局差异。
 */
class Decoder
{
public:
  virtual ~Decoder() = default;

  /**
   * @brief 解码为统一的 InferResult（坐标已反映射回原图）。
   * @param outputs 模型输出张量（按模型 output 顺序，f32 主机内存）
   * @param info    模型元信息
   * @param lb      letterbox 几何参数
   */
  virtual InferResult decode(
    const std::vector<Tensor> & outputs, const ModelInfo & info,
    const LetterboxInfo & lb) const = 0;

  /**
   * @brief 工厂：根据任务创建对应 Decoder。
   *        支持 yolov8 ~ 后续系列（detect/pose/classify 已落地，seg/obb/end2end 预留）。
   */
  static std::unique_ptr<Decoder> create(
    const ModelInfo & info, const std::vector<std::vector<int64_t>> & output_shapes,
    std::string * error = nullptr);
};

}  // namespace infvino

#endif  // INFVINO__DECODER_HPP_
