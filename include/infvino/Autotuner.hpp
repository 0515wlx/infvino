// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Autotuner —— 候选配置枚举 + 在线 GPU 基准，输出 TuningEntry。
//
// 与 OpenVINO 的差异：infvino 的候选**带实测指标**（ops/EU/cyc 与 expected），
// 因此调优缓存本身就是「物理极限的数据库」。枚举规则尽量小：候选来自 docs/kernel.md
// 里已验证的 config 谱系（而不是无脑笛卡尔积），避免开发板上 IGC 反复 JIT。
#ifndef INFVINO__AUTOTUNER_HPP_
#define INFVINO__AUTOTUNER_HPP_

#include <functional>
#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"
#include "infvino/Tiles.hpp"
#include "infvino/Tuning.hpp"

namespace infvino
{

/** @brief 一个候选 kernel 配置（不变量：kernel 名 + 编译选项 + 可读 config 串）。 */
struct Candidate
{
  std::string kernel;   // kernel 函数名
  std::string source;   // .cl 源名（"conv_ov" 等）
  std::string options;  // 编译宏
  std::string config;   // 人类可读 / 可 bake 的 config 串
};

// ---------------------------------------------------------------------------
// 候选枚举（每个 op 一个函数；只枚举 docs/kernel.md 已验证的谱系）
// ---------------------------------------------------------------------------

/** @brief conv3x3 groups=1：OV osv32 的 OBW/OBH 谱系（R22/R23 扫描范围）。*/
std::vector<Candidate> candidatesConv3x3(const OpSignature & sig);
/** @brief gemm：R9/R12 的 tile 谱系（BM/BN/BK/TM/TN/DBUF/SG）。*/
std::vector<Candidate> candidatesGemm(const OpSignature & sig);
/** @brief conv1x1 N>1：gemm + EPI 的 tile 谱系。*/
std::vector<Candidate> candidatesConv1x1(const OpSignature & sig);
/** @brief depthwise：GEMM 式 lane=空间 vs 标量 coalesced（R16）。*/
std::vector<Candidate> candidatesDepthwise(const OpSignature & sig);
/**
 * @brief Round 28：小算子（launch/带宽受限）的候选变体。
 *
 * 覆盖 ew_binary / ew_binary_bcast / ew_unary / concat4 / copy_c / slice_axis /
 * maxpool / resize_nn / permute_0213 / bmm / gap。变体都是**数值等价**的
 * （逐元素表达式与归约顺序不变），只改「每 work-item 处理多少元素 / 网格维度」，
 * 因此调优只影响性能、不影响数值。
 */
std::vector<Candidate> candidatesSmall(const OpSignature & sig);

/**
 * @brief 在设备上对一个具体配置计时。
 *
 * @param enqueue  调用方准备好 kernel 参数后，返回入队 event 的闭包（供 timeMs）。
 * @param iters    计时迭代数（warmup 固定 3）。
 * @param ms       输出：中位耗时。
 * @return 成功返回 true；build/set-arg 失败（异常）返回 false。
 */
bool benchCandidate(
  ClRuntime & rt, const std::function<cl_event()> & enqueue, int iters, double * ms,
  double * spread = nullptr);

/**
 * @brief 通用调优：枚举 → build → 计时 → 取最优，写入 entry。
 *
 * @param sig       节点签名（用于 expected_ops）。
 * @param cands     候选列表。
 * @param makeEnqueue  由候选生成「一次入队」闭包（负责 build kernel + set args + gws）。
 *                     允许抛异常（该候选被跳过）。
 * @param flops     用于 ops/EU/cyc 计算的 FLOPs。
 * @param iters     计时迭代数。
 * @return 最优 entry；所有候选都失败时返回空 kernel 的 entry。
 */
TuningEntry autotuneOp(
  ClRuntime & rt, const OpSignature & sig, const std::vector<Candidate> & cands,
  const std::function<std::function<cl_event()>(const Candidate &)> & makeEnqueue,
  double flops, int iters);

}  // namespace infvino

#endif  // INFVINO__AUTOTUNER_HPP_
