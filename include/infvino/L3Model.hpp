// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3Model —— L3（片上全局缓存）溢出的**全局、顺序相关**求值器 + 逐节点价格。
//
// 背景（R47/R49/R55/R59/R60）：L3 溢出**天然不可加**——节点 A 的在飞占用会逐出节点 B
// 的输入，谁被逐出取决于整张图的执行顺序与容量。R49 §10 的负结果证明：把占用压力
// （`occupancyPressure`，量纲是**足迹字节**）直接当节点成本（× κ）会破坏可加性与量级
// 校准。R55 的双租户标定表明逐出是**近期重用（LRU/recency）加权**的。
//
// R60 把这件事从「有限容量模拟 + 有限差分价」升级为**精确的复用距离/栈距离模型**：
//
//   关系式（stack-distance 定理）：  容量 C 下命中  ⇔  栈距离 d_t ≤ C
//   其中 d_t = Δ_local,t + I_other,t（自身字节 + 两次访问之间被触碰的**不同**数据字节）。
//   于是 spill = Σ_t bytes_t · 1[cold_t ∨ d_t > C]  由**复用距离直方图/CDF 完全决定**，
//   与任何容量常数解耦；多流/多算子的耦合表现为**干扰距离 I_other**（性质 4）。
//
//   同时保留有限调度下的**精确逐出归因**：某个张量是哪次访问把它从栈里挤出去的，
//   记为 `node_evict_bytes[node]`——这是「谁该为这次 miss 负责」的确定性因果账，
//   比「把节点占用归零」的有限差分更贴近实际可削减量（R55 §7.6 的过乐观根因）。
//
// 本模块把这件事做成「**可加主问题 + 非可加精确求值/定价**」的单一真相源：
//   * `evaluateL3(nodes)`：按拓扑序跑一遍 LRU 模拟，返回总溢出（字节/ms）+ 逐节点逐出价；
//   * `compute_prices=true` 时再用**有限差分**给出每个节点的**边际溢出耗时**
//     `ρ_i = (spill(occ_i) − spill(occ_i→0)) · κ`（单位 ms，≥0）。
//   * `profileL3Stack(nodes)`：无限栈的复用/栈距离画像（容量无关）+ 干扰分解 + 直方图；
//   * `spillAtCapacity(nodes, C)`：由复用距离 CDF 直接给出任意常数容量下的 spill。
//
// 逐出策略可用 `L3Policy` 切到 **NRU（1-bit not-recently-used + aging）**。这是**独立的
// 等价工程模型**（不是对任何具体硬件微架构的断言；参数由本项目自研标定工具实测，见
// docs/round60-* 与 THIRD_PARTY_NOTICES.md）。默认仍是严格 LRU，保证数值不变。
//
// 纯 host 逻辑、零 GPU，可离线单测（tests/test_l3_model.cpp）。
#ifndef INFVINO__L3_MODEL_HPP_
#define INFVINO__L3_MODEL_HPP_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "infvino/Tuning.hpp"

namespace infvino
{

/**
 * @brief R60: 替换策略的**等价工程模型**选择。
 *
 * * `LRU`：严格最近最少使用（基线，默认；与 R47–R59 行为逐位一致）。
 * * `NRU`：1-bit *not-recently-used* + aging 的**重用感知**变体。整条序列里会被重复访问
 *   的张量（reuse > 1）以「受保护」态填入；只被使用一次（流式）的张量以「evict-first」填入。
 *   需要逐出时优先牺牲未置位者，全部置位时统一清位（age）再退化为 LRU。这**行为等价于**
 *   公开资料所述的「1-bit LRU」式伪 LRU：频繁重用的热点抗流式污染——本项目**自行标定**
 *   的保留曲线（`config/l3_calibration.json`）给出 knee ≈ 11 MB × reuse^0.30，复现了
 *   R55 §3.5 实测的「热点到 ~8–12 MB 才被逐出」。
 *
 * ⚠️ 这是本项目**自行标定得到的等价工程模型**，不声称复刻任何厂商的具体微架构实现。
 */
enum class L3Policy
{
  LRU = 0,
  NRU = 1,
};

/** @brief R60: 由 `INFVINO_L3_POLICY` 解析默认替换策略（`nru`/`plru` → NRU，否则严格 LRU）。*/
L3Policy l3DefaultPolicy();

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

/** @brief L3 模型参数（默认 = R47/R55/R59 标定值）。 */
struct L3ModelConfig
{
  double l3_bytes = l3DefaultCapBytes();           ///< 跨算子 LRU 有效容量（R59 实测物理 L3；legacy 3.75e6）
  double spill_per_byte_ms = kL3SpillPerByteMs;    ///< miss 折算 (1/BW_DRAM − 1/BW_L3)
  // R59: 有效容量锚点 = R55 私有 tile 膝点 ~2MB（legacy 1MB）。见 docs/round59。
  double eff_cap_anchor = l3DefaultAnchorBytes();  ///< 有效容量锚点（R47=1MB，R55/R59≈2MB）
  double occ_cap_slope = 0.5;                      ///< effCap = max(anchor, L3 − slope·press)
  bool   compute_prices = true;                    ///< 是否用有限差分算逐节点价 ρ_i
  L3Policy policy = l3DefaultPolicy();             ///< R60: 替换策略等价模型（默认严格 LRU）
  // R71: 两级 miss 拆分（`l3TwoLevelSplit`）用；不影响 `evaluateL3` 的既有语义。
  double l3_private_bytes = l3PhysicalBytes();    ///< GPU 私有 L3 有效容量（3.75 MiB）
  double l3_bw_gbps = 145.0;                       ///< L3 命中带宽（copy 峰值）
  double llc_bw_gbps = 35.0;                       ///< 共享 LLC 段服务带宽（实测中值）
  double dram_bw_gbps = 20.0;                      ///< DRAM 平台带宽
};

struct L3Result
{
  double              spill_bytes = 0.0;   ///< 总 miss 字节
  double              spill_ms = 0.0;      ///< spill_bytes × spill_per_byte_ms
  std::vector<double> node_price_ms;       ///< 每节点：其占用导致的溢出耗时（≥0，有限差分）
  /**
   * @brief R60: **精确逐出归因价**（ms）。
   *
   * 模拟里，张量 t 在被逐出时记下「执行到哪个节点把它挤出去的」；之后 t 若再次被读
   * 且未命中，就把这次 miss 字节记到该节点头上。得到的是**确定性因果账**：
   * 节点 i 的价 = 它实际逐出、且后来真的 miss 掉的字节 × κ（≥0，且 Σ_i ≤ spill）。
   * 冷启动（首次访问）不计入归因（非干扰所致）。
   */
  std::vector<double> node_evict_ms;
};

/** @brief R60: 一次访问在**无限 LRU 栈**（容量无关）里的复用/栈距离画像。 */
struct L3StackTouch
{
  std::string name;          ///< 被访问的激活张量名
  double      bytes = 0.0;   ///< 本次访问的数据字节（Δ_local）
  bool        write = false; ///< true=写（生产者），false=读（消费者）
  size_t      node = 0;      ///< 发起本次访问的节点下标
  bool        cold = false;  ///< 该张量在整条序列里首次出现（compulsory miss）
  double      reuse_bytes = 0.0; ///< 两次访问之间被触碰的**不同**数据字节（干扰距离 I_other）
  double      stack_dist  = 0.0; ///< = reuse_bytes + bytes（命中阈值 d_t）
  bool        hit = false;       ///< 在**当前容量调度**下是否命中（有限模拟）
};

/** @brief R60: 全序列复用距离画像（无限栈）+ 逐节点精确逐出归因。 */
struct L3StackProfile
{
  std::vector<L3StackTouch> touches;      ///< 逐次访问（拓扑序）
  double                    total_bytes = 0.0;   ///< Σ bytes（所有访问）
  double                    spill_bytes = 0.0;   ///< 有限调度下的读 miss 字节
  std::vector<double>       node_evict_bytes;    ///< 每节点：其逐出导致、后来真 miss 的字节
};

/**
 * @brief 按拓扑序跑全局 L3 溢出模拟。
 *
 * 每个节点先按其占用压力压缩有效容量、逐出 LRU 头，再读输入激活（未命中 → 记 miss）、
 * 写输出。跨节点的耦合（A 的占用逐出 B 的输入）由模拟本身一致处理。
 * `L3Result::node_evict_ms` 给出精确逐出归因价。
 */
L3Result evaluateL3(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg = {});

/**
 * @brief R71: **两级缓存（GPU 私有 L3 → 共享 LLC → DRAM）miss 拆分**（诊断，不参与选择）。
 *
 * `evaluateL3` 把「读 miss」统一按 `(1/BW_DRAM − 1/BW_L3)` 折算——它不区分「只是从
 * GPU 私有 L3（3.75 MiB）掉到共享 LLC（8 MiB）」（每字节便宜 ~3×）与「两级都溢出、真打
 * DRAM」。R71 用**栈距离**把读 miss 拆成两档（与 `spillAtCapacity` 同源、容量无关）：
 *
 *   * `llc_served_bytes` = 栈距离 > GPU 私有 L3 但 ≤ 两级和（服务自 LLC）；
 *   * `dram_miss_bytes`  = 栈距离 > 两级和（或冷启动，服务自 DRAM）。
 *
 * 两档分别按各自相对 L3 命中的**增量代价**计价（LLC：`1/BW_LLC − 1/BW_L3`；
 * DRAM：`1/BW_DRAM − 1/BW_L3`）。这是纯诊断视图，`evaluateL3` 的既有语义与选择不变。
 */
struct L3TwoLevelSplit
{
  double l3_cap_bytes = 0.0;       ///< GPU 私有 L3 有效容量
  double llc_cap_bytes = 0.0;      ///< 跨算子有效容量（L3 + LLC）
  double l3_miss_bytes = 0.0;      ///< 超过 GPU L3 的读字节（= llc_served + dram）
  double llc_served_bytes = 0.0;   ///< 由共享 LLC 服务
  double dram_miss_bytes = 0.0;    ///< 直接打 DRAM
  double llc_miss_ms = 0.0;        ///< LLC 服务部分的增量耗时
  double dram_miss_ms = 0.0;       ///< DRAM 服务部分的增量耗时
  double total_ms = 0.0;           ///< = llc_miss_ms + dram_miss_ms
};
L3TwoLevelSplit l3TwoLevelSplit(const std::vector<L3Access> & nodes,
                                const L3ModelConfig & cfg = {});

/**
 * @brief R60: 无限栈复用/栈距离画像 + 精确逐出归因。
 *
 * 与 `evaluateL3` 同源地跑一遍有限调度（得到 `hit` / 逐出归因 / `spill_bytes`），
 * 另在同一次遍历里对**无限栈**记下每次访问的栈距离 `d_t` 与干扰距离 `I_other`
 * （容量无关的量，供 CDF / 容量扫描 / 干扰分解使用）。
 */
L3StackProfile profileL3Stack(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg = {});

/**
 * @brief R60: 常数容量 C 下的 LRU spill（读 miss 字节）。
 *
 * 由复用距离 CDF 直接给出：`Σ_{读且 cold 或 stack_dist > C} bytes`。
 * 对任意常数 C，与「容量固定为 C 的有限 LRU 模拟」**逐位一致**（单测保证），
 * 这把动态替换问题转化为**静态阈值**问题，可嵌入容量分配/背包模型。
 * `policy != LRU` 时退化为用 `evaluateL3` 的有限模拟（阈值定理只对严格 LRU 成立）。
 */
double spillAtCapacity(const std::vector<L3Access> & nodes, double capacity_bytes,
                       L3Policy policy = L3Policy::LRU);

}  // namespace infvino

#endif  // INFVINO__L3_MODEL_HPP_
