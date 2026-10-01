// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 与后端无关的张量：只含形状 + 主机侧 f32 数据。
// 自研 OpenCL 后端算完 fp16 后统一转成 f32 交给 Decoder，保持解码逻辑后端无关。
#ifndef INFVINO__TENSOR_HPP_
#define INFVINO__TENSOR_HPP_

#include <cstdint>
#include <numeric>
#include <vector>

namespace infvino
{

struct Tensor
{
  std::vector<int64_t> shape;
  std::vector<float>   data; /**< 行主序、连续 */

  Tensor() = default;
  Tensor(std::vector<int64_t> s, std::vector<float> d)
  : shape(std::move(s)), data(std::move(d))
  {
  }

  const std::vector<int64_t> & getShape() const { return shape; }
  size_t                       getSize() const { return data.size(); }
  const float *                dataPtr() const { return data.data(); }
  float *                      dataPtr() { return data.data(); }

  size_t numel() const
  {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d < 0 ? 0 : d);
    return n;
  }

  /** @brief 便捷构造：只有形状、数据填 0。 */
  static Tensor zeros(std::vector<int64_t> shape)
  {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d < 0 ? 0 : d);
    return Tensor(std::move(shape), std::vector<float>(n, 0.f));
  }
};

}  // namespace infvino

#endif  // INFVINO__TENSOR_HPP_
