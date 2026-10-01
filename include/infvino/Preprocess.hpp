// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#ifndef INFVINO__PREPROCESS_HPP_
#define INFVINO__PREPROCESS_HPP_

#include <opencv2/core.hpp>

#include <utility>
#include <vector>

namespace infvino
{

/**
 * @brief letterbox 的几何参数，用于把网络输出反映射回原图。
 */
struct LetterboxInfo
{
  float    scale    = 1.f; /**< 缩放比 = min(in_w/src_w, in_h/src_h) */
  int      pad_left = 0;
  int      pad_top  = 0;
  cv::Size input_size{0, 0};

  /** @brief 网络坐标 (x,y) -> 原图坐标 */
  inline cv::Point2f toImage(float x, float y) const
  {
    return {(x - static_cast<float>(pad_left)) / scale,
            (y - static_cast<float>(pad_top)) / scale};
  }
  inline cv::Rect2f rectToImage(float cx, float cy, float w, float h) const
  {
    const float x = (cx - static_cast<float>(pad_left)) / scale;
    const float y = (cy - static_cast<float>(pad_top)) / scale;
    return {x - w / scale / 2.f, y - h / scale / 2.f, w / scale, h / scale};
  }
};

/**
 * @brief 预处理配置。默认与 ultralytics 一致；分类模型（如 mobilenet）覆盖 mean/std。
 */
struct PreprocessConfig
{
  cv::Size           input_size{640, 640};
  bool               letterbox = true;  /**< true: letterbox；false: 直接 resize */
  bool               to_rgb    = true;  /**< BGR -> RGB */
  bool               normalize = true;  /**< /255 */
  std::vector<float> mean;              /**< 每通道均值(RGB)，在 /255 之后减去 */
  std::vector<float> std;               /**< 每通道标准差(RGB)，减去均值后再除 */
};

/**
 * @brief ultralytics 风格预处理：letterbox/resize + BGR->RGB + /255 (+ mean/std) + NCHW blob。
 */
class Preprocessor
{
public:
  explicit Preprocessor(PreprocessConfig cfg) : cfg_(std::move(cfg)) {}
  Preprocessor(cv::Size input_size, bool to_rgb = true, bool normalize = true)
  : cfg_{input_size, true, to_rgb, normalize, {}, {}} {}

  /**
   * @brief 生成 NCHW float32 blob（cv::Mat 连续内存，可直接喂 Tensor）。
   */
  cv::Mat operator()(const cv::Mat & bgr, LetterboxInfo * info = nullptr) const;

  cv::Size inputSize() const { return cfg_.input_size; }

private:
  PreprocessConfig cfg_;
};

} // namespace infvino

#endif // INFVINO__PREPROCESS_HPP_
