// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/Nms.hpp"

#include <algorithm>

namespace infvino
{

float iou(const cv::Rect2f & a, const cv::Rect2f & b)
{
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.width, b.x + b.width);
  const float y2 = std::min(a.y + a.height, b.y + b.height);

  const float inter_w = std::max(0.f, x2 - x1);
  const float inter_h = std::max(0.f, y2 - y1);
  const float inter = inter_w * inter_h;

  const float area_a = std::max(0.f, a.width) * std::max(0.f, a.height);
  const float area_b = std::max(0.f, b.width) * std::max(0.f, b.height);
  const float uni = area_a + area_b - inter;

  return uni > 1e-9f ? inter / uni : 0.f;
}

std::vector<int> nms(
  const std::vector<cv::Rect2f> & boxes, const std::vector<float> & scores, float iou_threshold)
{
  std::vector<int> order(boxes.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = static_cast<int>(i);

  std::stable_sort(
    order.begin(), order.end(),
    [&scores](int a, int b) { return scores[a] > scores[b]; });

  std::vector<int> keep;
  keep.reserve(order.size());
  std::vector<char> removed(boxes.size(), 0);

  for (size_t i = 0; i < order.size(); ++i)
  {
    const int idx = order[i];
    if (removed[idx]) continue;
    keep.push_back(idx);
    for (size_t j = i + 1; j < order.size(); ++j)
    {
      const int other = order[j];
      if (removed[other]) continue;
      if (iou(boxes[idx], boxes[other]) > iou_threshold) removed[other] = 1;
    }
  }
  return keep;
}

} // namespace infvino
