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
  // R48 D4: 消费者读「激活张量」的输入槽号。conv3x3/depthwise = 0；conv1x1/gemm 的槽 0 是
  // **权重**、激活在槽 1。布局规划器据此判断「该张量的消费者是否都吃 FSV16」——此前硬编码
  // 槽 0，导致 conv1x1_blk 作为消费者时永不持久化（其输入被误判为非激活），每次都付 reorder。
  int inIndex = 0;
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
  // R69: 本族候选的**占用/访存几何**（候选侧 → L3 溢出模拟的接口，见 Tuning.hpp
  // `MemContract`）。入参 = 签名 + 命中候选的 options。未声明（null）时消费方
  // （`occupancyPressure`/`occupancyThreads`）回退到 Tuning 的 legacy op 判据，
  // 保证小算子与未知 kernel 行为不变。
  std::function<MemContract(const OpSignature &, const std::string & options)> mem;
  // 软预测（"我们认为这层能到多少"）：含 amort / gridFactor / 占用等**经验 derate**。
  // 仅用于排序与相对比较，**不是物理上界**。
  std::function<double(const OpSignature &, const ClDeviceInfo &)> ceiling;
  // 硬上限（"物理上不可能超过"）：只放 ISA 指令发射配额 `32·mad_frac`（或 roofline
  // 的唯一字节下界）这类**不可越**的界。经验 derate（延迟/L3 复用/占用）只能作为
  // **告警/证据**，不得进这里——否则一次证伪（measured>hard）就同时污染了这些结论。
  // 未设置时回退为 `ceiling`（本来就是 roofline 的族）。
  std::function<double(const OpSignature &, const ClDeviceInfo &)> hardCeiling;
};

/**
 * @brief R48 §3.1: 候选预算（抗 IGC JIT / GPU HANG 的候选规模上限）。
 *
 * 候选数增长会放大 IGC JIT 暴露量与 GPU HANG 风险（历史主因）。注册表在
 * `candidatesFromRegistry` 末尾做**确定性**截断：
 *   * 每族配额：单族超过 `kFamilyCandidateQuota` 时按等距采样保留（含首尾）；
 *   * 每签名上限：各族候选按**轮转**交错后取前 `kSigCandidateCap` 个，保证
 *     「跨瓶颈类别」的多样性不被强势族挤掉。
 * 环境变量 `INFVINO_FAMILY_QUOTA` / `INFVINO_SIG_CAP` 可覆盖（0 = 不限）；
 * `INFVINO_CAND_STATS=1` 打印每签名的候选数/截断。D1–D6 新候选必须在此预算内声明。
 */
constexpr int kFamilyCandidateQuota = 32;
constexpr int kSigCandidateCap = 64;

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
