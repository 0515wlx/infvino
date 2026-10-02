// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Tuning —— 自动调优体系的基础设施：
//   1. OpSignature   ：把 (op, shape, 属性, dtype) 归一化成一个稳定的、可缓存的键；
//   2. TuningCache   ：按设备 + op + signature 记录「最优 kernel/config + 实测/期望 ops」；
//   3. expected_ops  ：**中间标准** —— 由 roofline / 寄存器预算 / 网格占用推导出的
//                      「这个 op 在这个 shape、这台机器上应该能到多少 ops/EU/cyc」。
//
// 设计见 docs/autotuning.md。缓存只影响「选哪个 kernel/config」，不参与数值，
// 查不到时永远可以回退到内核里的启发式。
#ifndef INFVINO_GK__TUNING_HPP_
#define INFVINO_GK__TUNING_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"

namespace gk
{

/**
 * @brief 一个待调优节点的不变签名（key）。
 *
 * 纯值类型；`str()` 产生稳定字符串用于缓存。字段按「影响 kernel 选择的物理量」组织，
 * 任一字段变化都应导致不同的缓存条目。
 */
struct OpSignature
{
  std::string op;      // "conv3x3" / "gemm" / "conv1x1" / "depthwise" / "gap"
  int W = 0, H = 0;    // 输出空间（conv）或 N=1 的占位
  int Cin = 0, Cout = 0;
  int stride = 1, pad = 1, groups = 1, K = 3;
  int act = 0;         // 融合激活码
  int batch = 1;
  int M = 0, N = 0;    // gemm 专用（M=Cout, N=HW, K=Cin）
  std::string dtype = "f16";
  // 通用小算子（Round 28）：op 专属维度的有序列表，参与签名串。用于
  // ew/unary/copy/slice/pool/resize/permute/bmm/concat/softmax 等无专用字段的算子，
  // 使「加一个新小算子」不必再改 OpSignature 结构。
  std::vector<int> params;

  std::string str() const;
  static OpSignature conv3x3(int Wout, int Hout, int stride, int pad, int Cin, int Cout, int act);
  static OpSignature convGeneral(int Wout, int Hout, int stride, int pad, int Cin, int Cout,
                                 int groups, int K, int act);
  static OpSignature gemm(int M, int N, int K, int act);
  static OpSignature conv1x1(int Cout, int N, int Cin, int act, int res);
  static OpSignature depthwise(int Wout, int Hout, int stride, int pad, int Cin, int K, int act);
  static OpSignature gap(int C, int HW);
  /** @brief 通用小算子签名（Round 28）：op 名 + 有序维度列表。*/
  static OpSignature custom(const std::string & op, std::vector<int> params, int act = 0);
};

/** @brief 一个缓存条目：最优 kernel + config（+ 实测与期望指标，用于分析）。 */
struct TuningEntry
{
  std::string kernel;   // e.g. "conv3x3_ov" / "conv3x3_f16" / "gemm_f16"
  std::string config;   // e.g. "OBW=5,OBH=2,STRIDE=1,PAD=1,ACT=1"（人类可读 / 可 bake）
  std::string options;  // kernel 编译宏（运行时据此 build/cache）
  double ms = 0.0;
  double ops = 0.0;          // 实测 ops/EU/cyc
  double expected = 0.0;     // 中间标准：期望 ops/EU/cyc
  double ratio = 0.0;        // ops / expected（1.0 = 达到期望）
  int    iters = 0;
  std::string device_id;
  std::string source;        // "tuned" / "heuristic"
};

/**
 * @brief 离线调优缓存（config/tuning.json）。
 *
 * 线程安全由调用方保证（PlanModel 单线程使用）。加载失败/文件缺失 → 空缓存，
 * 一切 lookup 未命中，安全回退。保存时 merge 已加载条目。
 */
class TuningCache
{
public:
  TuningCache() = default;

  /** @brief 从文件加载（不存在则空缓存）。*/
  static TuningCache load(const std::string & path);
  /** @brief 从默认路径加载：$INFVINO_TUNING_CACHE，否则 <repo>/config/tuning.json；
   *  环境变量 INFVINO_TUNING=off 或路径 "none" 时返回空缓存。*/
  static TuningCache loadDefault();
  /** @brief 写回文件（目录不存在则报错）。*/
  bool save(const std::string & path) const;

  bool enabled() const { return enabled_; }
  void setEnabled(bool e) { enabled_ = e; }
  const std::string & deviceId() const { return device_id_; }
  void setDeviceId(const std::string & id) { device_id_ = id; }

  /** @brief 查表；未命中或设备不匹配返回 nullptr。*/
  const TuningEntry * lookup(const OpSignature & sig) const;

  /** @brief 写入/更新一条（覆盖同 key）。*/
  void put(const OpSignature & sig, const TuningEntry & e);

  const std::map<std::string, TuningEntry> & entries() const { return entries_; }
  std::map<std::string, TuningEntry> &       entries() { return entries_; }
  size_t size() const { return entries_.size(); }

  /** @brief 由 DeviceInfo 生成设备键（优先 PCI id，其次 vendor:name:eu）。*/
  static std::string deviceKey(const DeviceInfo & d);

private:
  bool                              enabled_ = true;
  std::string                       device_id_;
  std::map<std::string, TuningEntry> entries_;
};

/**
 * @brief **中间标准**：期望 ops/EU/cyc。
 *
 * 这是 infvino 之前缺失的一层——在「物理极限（32/27/16/13/10）」与「实测」之间的
 * 可比较标尺。所有系数都标注了 docs/kernel.md 的出处，便于随实测修正。
 *
 * @param dev 设备信息（EU 数等，用于网格占用修正）。
 */
double expectedOps(const OpSignature & sig, const DeviceInfo & dev);

/** @brief 理论峰值 ops/EU/cyc（FP16 packed = 32）。*/
constexpr double kPeakOpsPerEuCycle = 32.0;
/** @brief 纯寄存器 FP16 FMA 的结构上限（R13/R18）。*/
constexpr double kRegisterFmaCeiling = 27.4;
/** @brief staging-free conv3x3 上限（R18）。*/
constexpr double kConvStagingFreeCeiling = 16.4;
/**
 * @brief conv3x3 OV 数据通路的**指令发射**上限（R24 ISA 实测）。
 *
 * 对移植的 `conv_ov.cl` 反汇编（OBW=8/OBH=2/s1）：主内循环 288 条 packed `mad` /
 * 453 条指令 = 63.6% mad（`sub_group_broadcast` 被 IGC 折进 mad 操作数，没有独立
 * 广播指令）。因此「指令配额」上限 = 32 × 0.636 ≈ 20.3，而不是 R20/R23 按
 * 「1 broadcast : 1 mad」推断的 16。实测 80×80 只有 13.8（≈68% 配额）、40×40
 * 只有 8.5（≈42%），缺口是**流水/延迟/占用**，不是指令数——这是 R24 要压榨的部分。
 */
constexpr double kConvOvIssueCeiling = 20.3;
/** @brief conv3x3 OV 主循环的 mad 指令占比（R24 ISA：288/453）。*/
constexpr double kConvOvMadFraction = 0.636;

}  // namespace gk

#endif  // INFVINO_GK__TUNING_HPP_
