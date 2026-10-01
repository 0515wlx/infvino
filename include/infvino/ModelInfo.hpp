// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#ifndef INFVINO__MODEL_INFO_HPP_
#define INFVINO__MODEL_INFO_HPP_

#include <opencv2/core.hpp>

#include <string>
#include <vector>

#include "infvino/Types.hpp"

namespace infvino
{

/**
 * @brief 模型描述。来源优先级：
 *        1) models.yaml 显式声明（生产首选，最稳）
 *        2) 预留：ONNX metadata
 *        3) 预留：输出形状兜底推断
 */
struct ModelInfo
{
  std::string              path;                     /**< 模型路径 (.onnx，导出/校验用) */
  std::string              plan;                     /**< 执行计划路径 (.plan，GPU 后端必需) */
  std::string              kernel_dir;               /**< .cl 源码目录，空则用编译期默认 */
  Task                     task           = Task::Detect;
  cv::Size                 input_size     = {640, 640};
  int                      num_classes    = 1;
  std::vector<std::string> names;                    /**< 类别名（可空，仅调试/映射用） */
  std::vector<int>         kpt_shape;                /**< Pose: {K, dims}，如 {17,3} */
  float                    conf           = 0.25f;
  float                    iou            = 0.45f;
  int                      max_det        = 300;
  bool                     end2end        = false;   /**< 模型已内置 NMS (v10/v12 等导出) */
  std::string              device         = "AUTO";  /**< AUTO / GPU / CPU / GPU.0 ... */
  bool                     allow_cpu_fallback = true;

  // 预处理（默认与 ultralytics 一致；分类模型如 mobilenet 需覆盖）
  bool                     letterbox      = true;    /**< true: letterbox；false: 直接 resize */
  bool                     to_rgb         = true;
  bool                     normalize      = true;    /**< /255 */
  std::vector<float>       mean;                     /**< 每通道均值(RGB)，空则不减；在 /255 之后 */
  std::vector<float>       std;                      /**< 每通道标准差(RGB)，空则不除 */

  /** @brief 从 yaml 读取 model 配置。key 为空时读取顶层 default 指定的模型。 */
  static ModelInfo fromYaml(const std::string & yaml_path, const std::string & key = "");

  int numKeypoints() const { return kpt_shape.empty() ? 0 : kpt_shape[0]; }
  int kptDims() const { return kpt_shape.size() < 2 ? 0 : kpt_shape[1]; }
};

} // namespace infvino

#endif // INFVINO__MODEL_INFO_HPP_
