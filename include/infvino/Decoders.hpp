// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Detect / Pose / End2End / Classify 解码器实现。
// 其余任务（seg/obb）在 Decoder::create 中返回明确错误，后续按同样模式补齐。

#ifndef INFVINO__DECODERS_HPP_
#define INFVINO__DECODERS_HPP_

#include "infvino/Decoder.hpp"

namespace infvino
{

/**
 * @brief anchor-free 检测头。ultralytics v8/v11/v26 detect 输出 [1, 4+nc, N]。
 *        同时兼容转置布局 [1, N, 4+nc]。
 */
class DetectionDecoder : public Decoder
{
public:
  explicit DetectionDecoder(const ModelInfo & info) : info_(info) {}
  InferResult decode(
    const std::vector<Tensor> & outputs, const ModelInfo & info,
    const LetterboxInfo & lb) const override;

private:
  ModelInfo info_;
};

/**
 * @brief 关键点检测头。ultralytics -pose 输出 [1, 4+1+3K, N]（K=kpt_shape[0]）。
 */
class PoseDecoder : public Decoder
{
public:
  explicit PoseDecoder(const ModelInfo & info) : info_(info) {}
  InferResult decode(
    const std::vector<Tensor> & outputs, const ModelInfo & info,
    const LetterboxInfo & lb) const override;

private:
  ModelInfo info_;
};

/**
 * @brief 已内置 NMS 的端到端输出 [1, N, 6]，6=[x1,y1,x2,y2,conf,cls]。
 */
class End2EndDecoder : public Decoder
{
public:
  InferResult decode(
    const std::vector<Tensor> & outputs, const ModelInfo & info,
    const LetterboxInfo & lb) const override;
};

/**
 * @brief 图像分类头，输出 [1, nc]（logits）。做 softmax 后返回 top-k。
 *        用于 mobilenetv3-small 等分类模型。
 */
class ClassifyDecoder : public Decoder
{
public:
  explicit ClassifyDecoder(int topk = 5) : topk_(topk) {}
  InferResult decode(
    const std::vector<Tensor> & outputs, const ModelInfo & info,
    const LetterboxInfo & lb) const override;

private:
  int topk_;
};

}  // namespace infvino

#endif  // INFVINO__DECODERS_HPP_
