// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#ifndef INFVINO__NMS_HPP_
#define INFVINO__NMS_HPP_

#include <opencv2/core.hpp>

#include <vector>

namespace infvino
{

/** @brief IoU (x,y,w,h) */
float iou(const cv::Rect2f & a, const cv::Rect2f & b);

/** @brief 单类 NMS，返回保留下来的下标，按 score 降序。 */
std::vector<int> nms(
  const std::vector<cv::Rect2f> & boxes, const std::vector<float> & scores, float iou_threshold);

} // namespace infvino

#endif // INFVINO__NMS_HPP_
