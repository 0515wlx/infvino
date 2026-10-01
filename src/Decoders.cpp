// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/Decoders.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include "infvino/Nms.hpp"

namespace infvino
{

namespace
{

/**
 * @brief 判断 anchor-free 输出是 [C, N] 还是 [N, C]。
 * @param shape 输出形状 (size==3)
 * @param feature 期望的特征维 (4+nc 或 4+1+3K)
 * @return true 表示 [1, C, N]
 */
bool isChannelMajor(const std::vector<int64_t> & shape, int feature)
{
  const int64_t d1 = shape[1];
  const int64_t d2 = shape[2];
  if (feature > 0 && d1 == feature) return true;
  if (feature > 0 && d2 == feature) return false;
  return d1 < d2;  // 特征维通常远小于 anchor 维
}

std::vector<Detection> nmsPerClass(
  std::vector<Detection> dets, float iou_thr, int max_det)
{
  if (dets.empty()) return dets;

  std::vector<Detection> out;
  // 按类别分组做 NMS
  std::vector<int> class_ids;
  for (const auto & d : dets) class_ids.push_back(d.class_id);
  std::sort(class_ids.begin(), class_ids.end());
  class_ids.erase(std::unique(class_ids.begin(), class_ids.end()), class_ids.end());

  for (int cid : class_ids)
  {
    std::vector<cv::Rect2f> boxes;
    std::vector<float>      scores;
    std::vector<int>        index;
    for (size_t i = 0; i < dets.size(); ++i)
    {
      if (dets[i].class_id != cid) continue;
      boxes.push_back({dets[i].box.x, dets[i].box.y, dets[i].box.width, dets[i].box.height});
      scores.push_back(dets[i].score);
      index.push_back(static_cast<int>(i));
    }
    for (int k : nms(boxes, scores, iou_thr)) out.push_back(dets[index[k]]);
  }

  std::sort(
    out.begin(), out.end(),
    [](const Detection & a, const Detection & b) { return a.score > b.score; });
  if (static_cast<int>(out.size()) > max_det) out.resize(max_det);
  return out;
}

}  // namespace

InferResult DetectionDecoder::decode(
  const std::vector<Tensor> & outputs, const ModelInfo & info, const LetterboxInfo & lb) const
{
  InferResult result;
  if (outputs.empty() || outputs[0].shape.size() != 3)
    throw std::runtime_error("DetectionDecoder: unexpected output shape");

  const std::vector<int64_t> & shape = outputs[0].shape;
  const int64_t                d1 = shape[1];
  const int64_t                d2 = shape[2];

  const int expected_feature = 4 + std::max(1, info.num_classes);
  bool      c_major = isChannelMajor(shape, expected_feature);
  int64_t   channels = c_major ? d1 : d2;
  if (channels != expected_feature)
  {
    // num_classes 与模型不符时，以“较小的一维”作为特征维兜底
    c_major = d1 < d2;
    channels = c_major ? d1 : d2;
  }
  const int64_t total_anchors = c_major ? d2 : d1;
  const int     nc = static_cast<int>(channels) - 4;

  const float * data = outputs[0].dataPtr();
  auto value = [&](int64_t c, int64_t a) -> float {
    return c_major ? data[c * total_anchors + a] : data[a * channels + c];
  };

  std::vector<Detection> dets;
  dets.reserve(128);
  for (int64_t a = 0; a < total_anchors; ++a)
  {
    float best = 0.f;
    int   best_c = -1;
    for (int c = 0; c < nc; ++c)
    {
      const float s = value(4 + c, a);
      if (s > best) { best = s; best_c = c; }
    }
    if (best_c < 0 || best < info.conf) continue;

    const float cx = value(0, a);
    const float cy = value(1, a);
    const float w = value(2, a);
    const float h = value(3, a);

    Detection d;
    d.box = lb.rectToImage(cx, cy, w, h);
    d.score = best;
    d.class_id = best_c;
    if (d.box.width <= 0.f || d.box.height <= 0.f) continue;
    dets.push_back(std::move(d));
  }

  result.detections = nmsPerClass(std::move(dets), info.iou, info.max_det);
  return result;
}

InferResult PoseDecoder::decode(
  const std::vector<Tensor> & outputs, const ModelInfo & info, const LetterboxInfo & lb) const
{
  InferResult result;
  if (outputs.empty() || outputs[0].shape.size() != 3)
    throw std::runtime_error("PoseDecoder: unexpected output shape");

  const std::vector<int64_t> & shape = outputs[0].shape;
  const int64_t                d1 = shape[1];
  const int64_t                d2 = shape[2];

  const int k = info.numKeypoints() > 0 ? info.numKeypoints() : 0;
  const int dims = info.kptDims() > 0 ? info.kptDims() : 3;
  int feature = 4 + 1 + k * dims;
  if (k == 0)
  {
    // 未声明 kpt_shape 时，用较小维兜底推断
    feature = static_cast<int>(std::min(d1, d2));
  }

  bool c_major = isChannelMajor(shape, feature);
  const int64_t channels = c_major ? d1 : d2;
  const int64_t total_anchors = c_major ? d2 : d1;
  const int nk = k > 0 ? k : (static_cast<int>(channels) - 5) / dims;

  const float * data = outputs[0].dataPtr();
  auto value = [&](int64_t c, int64_t a) -> float {
    return c_major ? data[c * total_anchors + a] : data[a * channels + c];
  };

  std::vector<Detection> dets;
  for (int64_t a = 0; a < total_anchors; ++a)
  {
    const float score = value(4, a);
    if (score < info.conf) continue;

    Detection d;
    d.box = lb.rectToImage(value(0, a), value(1, a), value(2, a), value(3, a));
    d.score = score;
    d.class_id = 0;
    d.keypoints.resize(std::max(0, nk));
    for (int kk = 0; kk < nk; ++kk)
    {
      const float kx = value(5 + kk * dims + 0, a);
      const float ky = value(5 + kk * dims + 1, a);
      d.keypoints[kk] = lb.toImage(kx, ky);
    }
    dets.push_back(std::move(d));
  }

  result.detections = nmsPerClass(std::move(dets), info.iou, info.max_det);
  return result;
}

InferResult End2EndDecoder::decode(
  const std::vector<Tensor> & outputs, const ModelInfo & info, const LetterboxInfo & lb) const
{
  InferResult result;
  if (outputs.empty()) return result;
  const std::vector<int64_t> & shape = outputs[0].shape;
  // 期望 [1, N, 6]；若为 [1, 6, N] 则转置
  bool row_major = shape.size() == 3 && shape[2] == 6;
  if (shape.size() == 3 && shape[1] == 6) row_major = false;
  if (!row_major && !(shape.size() == 3 && shape[1] == 6))
    throw std::runtime_error("End2EndDecoder: expected [1, N, 6] or [1, 6, N]");

  const int64_t n = row_major ? shape[1] : shape[2];
  const float * data = outputs[0].dataPtr();

  auto at = [&](int64_t i, int64_t c) -> float {
    return row_major ? data[i * 6 + c] : data[c * n + i];
  };

  for (int64_t i = 0; i < n; ++i)
  {
    const float conf = at(i, 4);
    if (conf < info.conf) continue;
    Detection d;
    const float x1 = (at(i, 0) - static_cast<float>(lb.pad_left)) / lb.scale;
    const float y1 = (at(i, 1) - static_cast<float>(lb.pad_top)) / lb.scale;
    const float x2 = (at(i, 2) - static_cast<float>(lb.pad_left)) / lb.scale;
    const float y2 = (at(i, 3) - static_cast<float>(lb.pad_top)) / lb.scale;
    d.box = {std::min(x1, x2), std::min(y1, y2), std::abs(x2 - x1), std::abs(y2 - y1)};
    d.score = conf;
    d.class_id = static_cast<int>(at(i, 5));
    result.detections.push_back(std::move(d));
  }
  if (static_cast<int>(result.detections.size()) > info.max_det)
    result.detections.resize(info.max_det);
  return result;
}

InferResult ClassifyDecoder::decode(
  const std::vector<Tensor> & outputs, const ModelInfo & info, const LetterboxInfo & /*lb*/) const
{
  InferResult result;
  if (outputs.empty()) return result;

  const float * data = outputs[0].dataPtr();
  const size_t  n = outputs[0].data.size();
  if (n == 0) return result;

  std::vector<float> prob(data, data + n);
  const float mx = *std::max_element(prob.begin(), prob.end());
  double sum = 0.0;
  for (auto & v : prob) { v = std::exp(v - mx); sum += v; }
  if (sum <= 0.0) sum = 1.0;
  for (auto & v : prob) v = static_cast<float>(v / sum);

  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  const int k = std::min<int>(topk_, static_cast<int>(n));
  std::partial_sort(
    idx.begin(), idx.begin() + k, idx.end(),
    [&prob](int a, int b) { return prob[a] > prob[b]; });

  result.topk.reserve(k);
  for (int i = 0; i < k; ++i) result.topk.emplace_back(idx[i], prob[idx[i]]);
  (void)info;
  return result;
}

std::unique_ptr<Decoder> Decoder::create(
  const ModelInfo & info, const std::vector<std::vector<int64_t>> & output_shapes, std::string * error)
{
  (void)output_shapes;
  const auto fail = [error](const std::string & msg) -> std::unique_ptr<Decoder> {
    if (error) *error = msg;
    return nullptr;
  };

  if (info.end2end) return std::make_unique<End2EndDecoder>();

  switch (info.task)
  {
    case Task::Detect: return std::make_unique<DetectionDecoder>(info);
    case Task::Pose:   return std::make_unique<PoseDecoder>(info);
    case Task::Seg:    return fail("Seg decoder not implemented yet");
    case Task::Obb:    return fail("Obb decoder not implemented yet");
    case Task::Classify: return std::make_unique<ClassifyDecoder>();
    default:           return fail("Unknown task; specify task in models.yaml");
  }
}

}  // namespace infvino
