// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/Preprocess.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace infvino
{

cv::Mat Preprocessor::operator()(const cv::Mat & bgr, LetterboxInfo * info) const
{
  const cv::Size in = cfg_.input_size;
  cv::Mat        canvas;

  if (cfg_.letterbox)
  {
    const float scale =
      std::min(static_cast<float>(in.width) / bgr.cols, static_cast<float>(in.height) / bgr.rows);
    const int resized_w = static_cast<int>(std::round(bgr.cols * scale));
    const int resized_h = static_cast<int>(std::round(bgr.rows * scale));
    const int pad_w = in.width - resized_w;
    const int pad_h = in.height - resized_h;
    const int pad_left = pad_w / 2;
    const int pad_top = pad_h / 2;

    if (info)
    {
      info->scale = scale;
      info->pad_left = pad_left;
      info->pad_top = pad_top;
      info->input_size = in;
    }

    cv::Mat resized;
    cv::resize(bgr, resized, {resized_w, resized_h}, 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(
      resized, canvas, pad_top, pad_h - pad_top, pad_left, pad_w - pad_left,
      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  }
  else
  {
    cv::resize(bgr, canvas, in, 0, 0, cv::INTER_LINEAR);
    if (info)
    {
      info->scale = 1.f;
      info->pad_left = 0;
      info->pad_top = 0;
      info->input_size = in;
    }
  }

  // BGR->RGB, /255, NCHW
  cv::Mat blob = cv::dnn::blobFromImage(
    canvas, cfg_.normalize ? (1.0 / 255.0) : 1.0, in, cv::Scalar(0, 0, 0),
    /*swapRB=*/cfg_.to_rgb, /*crop=*/false, CV_32F);

  // 每通道 (pixel/255 - mean) / std —— 在 blobFromImage 之后手写，保证与参考实现一致
  const bool has_mean = !cfg_.mean.empty();
  const bool has_std = !cfg_.std.empty();
  if (has_mean || has_std)
  {
    const int channels = blob.size[1];
    for (int c = 0; c < channels; ++c)
    {
      float m = (has_mean && c < static_cast<int>(cfg_.mean.size())) ? cfg_.mean[c] : 0.f;
      float s = (has_std && c < static_cast<int>(cfg_.std.size())) ? cfg_.std[c] : 1.f;
      if (s == 0.f) s = 1.f;
      float * plane = blob.ptr<float>(0, c);
      const size_t n = static_cast<size_t>(blob.size[2]) * blob.size[3];
      for (size_t i = 0; i < n; ++i) plane[i] = (plane[i] - m) / s;
    }
  }
  return blob;
}

} // namespace infvino
