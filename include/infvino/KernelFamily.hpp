// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// KernelFamily —— 算子族的**声明式注册表**：把「一个 kernel 族」的所有元信息
// （支持的 op / 布局契约 / 约束 / 候选枚举 / 物理上限模型）集中在一处，供
// Autotuner（选族）、PlanModel（布局规划 + dispatch）共同消费。
//
// 设计动机见 docs/kernel-families.md：
//   * 价值单元是「族」（数据通路 × 布局契约 × 配置空间），不是「kernel」；
//   * 加一个族 = 往 registry 加一条声明，而不是改 candidatesXxx / dispatch /
//     布局规划 / expectedOps 四处；
//   * 每族给出**自己的物理上限**（familyCeiling），报告器据此判断「本族写满了没 /
//     该不该换族」。
#ifndef INFVINO__KERNEL_FAMILY_HPP_
#define INFVINO__KERNEL_FAMILY_HPP_

#include <functional>
#include <string>
#include <vector>

#include "infvino/Autotuner.hpp"  // Candidate
#include "infvino/ClRuntime.hpp"
#include "infvino/Tuning.hpp"

namespace infvino
{

/** @brief 张量的存储布局（当前只有两种；可扩展 bfyx/bsv 等）。 */
enum class Layout
{
  NCHW = 0,  // 普通 bfyx，通道最外
  FSV16 = 1, // b_fs_yx_fsv16，[C/16][H][W][16]
};

/** @brief 该族受限的瓶颈类别——决定用哪把「中间标准」尺子。 */
enum class Bottleneck
{
  Fma,         // packed FMA 发射
  Instruction, // 非 mad 指令（地址/边界）主导，如 depthwise
  Memory,      // 内存 roofline（带宽/足迹）
  Launch,      // dispatch 地板 + 网格饥饿
  Latency,     // 依赖链 / 占用
};

/** @brief 一个族的布局契约。 */
struct LayoutReq
{
  Layout in = Layout::NCHW;       // 输入必须是这个布局
  Layout out = Layout::NCHW;      // 实际写出的布局
  bool canOutFsv16 = false;       // 能否直接产出 FSV16（供持久 blocked 链）
  bool needsWeightRepack = true;  // 是否有 host 版权重预重排
};

/**
 * @brief 一个算子族。
 *
 * `candidates(sig)` 给出该族在 sig 上的所有可枚举配置（空 = 不适用），
 * `ceiling(sig, dev)` 给出该族的物理上限（ops/EU/cyc 口径，与 Candidate 实测同量纲）。
 */
struct KernelFamily
{
  std::string name;    // 族 id（也作为 TuningEntry.kernel 的族前缀）
  std::string op;      // 主 op（"conv3x3"/"gemm"/"conv1x1"/"depthwise"/...）
  std::string source;  // .cl 源名（用于 buildKernel）
  LayoutReq   layout;
  Bottleneck  bottleneck = Bottleneck::Fma;
  // 该族能正确解释的激活码集合（bit i = 支持规范码 i）。规范激活码（所有族统一）：
  // 1=SiLU 2=ReLU 3=HardSwish 4=HardSigmoid 5=Sigmoid（0=none）。conv3x3 只实现
  // {0,1,3}；depthwise {0..4}；其余 {0..5}。注册表据此拒绝 act 码超出本族语义的节点。
  int actMask = 0x3F;
  std::function<bool(const OpSignature &)> supports;
  std::function<std::vector<Candidate>(const OpSignature &)> candidates;
  // 软预测（"我们认为这层能到多少"）：含 amort / gridFactor / 占用等**经验 derate**。
  // 仅用于排序与相对比较，**不是物理上界**。
  std::function<double(const OpSignature &, const ClDeviceInfo &)> ceiling;
  // 硬上限（"物理上不可能超过"）：只放 ISA 指令发射配额 `32·mad_frac`（或 roofline
  // 的唯一字节下界）这类**不可越**的界。经验 derate（延迟/L3 复用/占用）只能作为
  // **告警/证据**，不得进这里——否则一次证伪（measured>hard）就同时污染了这些结论。
  // 未设置时回退为 `ceiling`（本来就是 roofline 的族）。
  std::function<double(const OpSignature &, const ClDeviceInfo &)> hardCeiling;
};

/** @brief 全部已注册的算子族（单一真相源）。 */
const std::vector<KernelFamily> & kernelFamilies();

/** @brief 按 op 过滤 + `supports` 过滤后，拼出所有候选（去重由调用方保证）。 */
std::vector<Candidate> candidatesFromRegistry(const OpSignature & sig);

/** @brief 由 kernel 名反查族（例如 "conv1x1_blk" -> 该族）；找不到返回 nullptr。 */
const KernelFamily * familyByName(const std::string & kernelName);

/** @brief 该 op 在 sig 上所有族的上界之最大值（跨族 best ceiling）。 */
double bestFamilyCeiling(const OpSignature & sig, const ClDeviceInfo & dev);

}  // namespace infvino

#endif  // INFVINO__KERNEL_FAMILY_HPP_
