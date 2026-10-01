// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// inference_engine 通用结果类型。
// 该头文件不依赖 ROS，只依赖 OpenCV，便于离线测试与复用。

#ifndef INFVINO__TYPES_HPP_
#define INFVINO__TYPES_HPP_

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace infvino
{

/**
 * @brief 任务类型。与 ultralytics 的 task 字段一致。
 */
enum class Task
{
  Detect,   /**< 目标检测   v8/v9/v10/v11/v26 ... */
  Seg,      /**< 实例分割   -seg */
  Pose,     /**< 关键点     -pose */
  Obb,      /**< 旋转框     -obb */
  Classify, /**< 分类       -cls */
  Unknown
};

const char * toString(Task task);
Task          taskFromString(const std::string & s);
Task          taskFromNumClasses(int nc); // 仅兜底用

/**
 * @brief 与后端无关的单条检测结果。坐标均为 **原图像素坐标**。
 */
struct Detection
{
  cv::Rect2f box;                     /**< 检测框 xywh -> 这里统一存 (x,y,w,h) 的原图坐标 */
  int        class_id = 0;            /**< 类别 id */
  float      score    = 0.f;          /**< 置信度 */
  std::vector<cv::Point2f> keypoints; /**< Pose: 原图坐标 (K 个); 其余任务为空 */
  float                    angle = 0.f; /**< Obb: 弧度 */
};

/**
 * @brief 一次推理的完整结果。
 */
struct InferResult
{
  std::vector<Detection> detections;
  std::vector<std::pair<int, float>> topk; /**< Classify: (class_id, prob)，概率降序 */
  cv::Matx33f            input2image{cv::Matx33f::eye()}; /**< 预处理逆变换（letterbox -> 原图） */

  double preprocess_ms  = 0.0;
  double infer_ms       = 0.0;
  double postprocess_ms = 0.0;
  double total_ms       = 0.0;

  bool empty() const { return detections.empty(); }
};

} // namespace infvino

#endif // INFVINO__TYPES_HPP_
