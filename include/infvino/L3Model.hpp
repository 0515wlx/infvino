// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3Model —— L3（片上全局缓存）溢出的**全局、顺序相关**求值器 + 逐节点价格。
//
// 背景（R47/R49/R55）：L3 溢出**天然不可加**——节点 A 的在飞占用会逐出节点 B 的输入，
// 谁被逐出取决于整张图的执行顺序与容量。R49 §10 的负结果证明：把占用压力
// （`occupancyPressure`，量纲是**足迹字节**）直接当节点成本（× κ）会破坏可加性与量级
// 校准。R55 的双租户标定表明逐出是**近期重用（LRU/recency）加权**的。
//
// 本模块把这件事做成「**可加主问题 + 非可加精确求值/定价**」的单一真相源：
//   * `evaluateL3(nodes)`：按拓扑序跑一遍 LRU 模拟，返回总溢出（字节/ms）；
//   * `compute_prices=true` 时再用**有限差分**给出每个节点的**边际溢出耗时**
//     `ρ_i = (spill(occ_i) − spill(occ_i→0)) · κ`（单位 ms，≥0）。
// 于是**定价主问题**（mincut / 坐标下降）可以加一项 `ρ_i · occ_i(candidate)`——量纲是
// ms、且以实际溢出为上限，不会像 R49 那样压过隔离 ms；非线性只通过 `ρ_i` 的**重标定**
// （迭代）注入，主问题保持可加。
//
// 纯 host 逻辑、零 GPU，可离线单测（tests/test_l3_model.cpp）。
#ifndef INFVINO__L3_MODEL_HPP_
#define INFVINO__L3_MODEL_HPP_

#include <string>
#include <utility>
#include <vector>

#include "infvino/Tuning.hpp"

namespace infvino
{

/**
 * @brief 一个算子在 L3 模拟里的视图（纯值类型）。
 *
 * `occ_bytes` 是该节点的**在飞占用压力** R（`occupancyPressure`；小算子为其流式足迹）。
 * `reads`/`writes` 是它读/写的**激活**张量（name 唯一即可；权重/常量不计）。
 */
struct L3Access
{
  std::string                                 op;
  double                                      occ_bytes = 0.0;
  std::vector<std::pair<std::string, double>> reads;
  std::vector<std::pair<std::string, double>> writes;
};

/** @brief L3 模型参数（默认 = R47/R55 标定值）。 */
struct L3ModelConfig
{
  double l3_bytes = 3.75e6;                        ///< 标称 L3 容量（字节）
  double spill_per_byte_ms = kL3SpillPerByteMs;    ///< miss 折算 (1/BW_DRAM − 1/BW_L3)
  // R47 现有行为用 1 MB 锚点；R55 标定私有 tile 形态有效容量 ~2 MB（见
  // docs/round55-l3-coupling-calibration.md §2.3）。默认保持 1 MB 以免改动未经 A/B 的
  // `predictNet`；需要 R55 形态时显式置 2.0e6。
  double eff_cap_anchor = 1.0e6;                   ///< 有效容量锚点（R47=1MB，R55≈2MB）
  double occ_cap_slope = 0.5;                      ///< effCap = max(anchor, L3 − slope·press)
  bool   compute_prices = true;                    ///< 是否用有限差分算逐节点价 ρ_i
};

struct L3Result
{
  double              spill_bytes = 0.0;   ///< 总 miss 字节
  double              spill_ms = 0.0;      ///< spill_bytes × spill_per_byte_ms
  std::vector<double> node_price_ms;       ///< 每节点：其占用导致的溢出耗时（≥0）
};

/**
 * @brief 按拓扑序跑全局 LRU 溢出模拟。
 *
 * 每个节点先按其占用压力压缩有效容量、逐出 LRU 头，再读输入激活（未命中 → 记 miss）、
 * 写输出。跨节点的耦合（A 的占用逐出 B 的输入）由模拟本身一致处理。
 */
L3Result evaluateL3(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg = {});

}  // namespace infvino

#endif  // INFVINO__L3_MODEL_HPP_
