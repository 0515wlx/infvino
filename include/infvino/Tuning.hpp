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
#ifndef INFVINO__TUNING_HPP_
#define INFVINO__TUNING_HPP_

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"

namespace infvino
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
  /** @brief R30c: fused `concat4 -> conv1x1` (Cin split into ca/cb/cc/cd). */
  static OpSignature conv1x1Cat4(int Cout, int N, int Cin, int ca, int cb, int cc, int cd,
                                 const int *off, int act, int res);
  static OpSignature depthwise(int Wout, int Hout, int stride, int pad, int Cin, int K, int act);
  static OpSignature gap(int C, int HW);
  /** @brief 通用小算子签名（Round 28）：op 名 + 有序维度列表。*/
  static OpSignature custom(const std::string & op, std::vector<int> params, int act = 0);
};

/**
 * @brief 调优缓存的 **kernel ABI 版本**。
 *
 * 缓存的 key 只含 op/shape；`options` 是编译宏（kernel 源码的「接口」）。若 kernel
 * 源码里某个 `-D` 宏的语义变化（重命名/行为改）而 key 不变，旧条目会被静默套用。
 * 每次做这类**破坏性 kernel 改动**时把这个值 +1：load 时若文件里的 abi 与当前不符，
 * 整份缓存视为未命中（回退启发式，可重新 retune），而不是错误地复用旧 options。
 */
constexpr int kTuningCacheAbi = 1;

/** @brief 一个缓存条目：最优 kernel + config（+ 实测与期望指标，用于分析）。 */
struct TuningEntry
{
  std::string kernel;   // e.g. "conv3x3_ov" / "conv3x3_f16" / "gemm_f16"
  std::string config;   // e.g. "OBW=5,OBH=2,STRIDE=1,PAD=1,ACT=1"（人类可读 / 可 bake）
  std::string options;  // kernel 编译宏（运行时据此 build/cache）
  double ms = 0.0;
  double ops = 0.0;          // 实测 ops/EU/cyc
  double expected = 0.0;     // 软预测（中间标准）：含经验 derate，仅用于排序/相对比较
  double ratio = 0.0;        // ops / expected（软）
  double hard_ceiling = 0.0; // 硬上限（ISA 配额/roofline 下界）：物理上不可能超过
  double hard_ratio = 0.0;   // ops / hard_ceiling（"离物理极限"）
  int    iters = 0;
  std::string device_id;
  std::string source;        // "tuned" / "heuristic"
  // R48 §3.2: 数值契约（从胜出候选带上来）。exact=true 表示仍逐位一致；否则 tol 为该
  // 候选允许的 mean_rel 上限，model_check 据此**按候选**放宽端到端验收口径。
  bool   exact = true;
  double tol = 0.0;
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
  int  abi() const { return abi_; }
  /** @brief 生成该缓存时 kernel 源码的组合指纹（R45 P1#7 数值契约守卫；空=未标注）。*/
  const std::string & sourceHash() const { return source_hash_; }
  void setSourceHash(const std::string & h) { source_hash_ = h; }

  /** @brief 查表；未命中或设备不匹配返回 nullptr。*/
  const TuningEntry * lookup(const OpSignature & sig) const;

  /** @brief 写入/更新一条（覆盖同 key）。*/
  void put(const OpSignature & sig, const TuningEntry & e);

  const std::map<std::string, TuningEntry> & entries() const { return entries_; }
  std::map<std::string, TuningEntry> &       entries() { return entries_; }
  size_t size() const { return entries_.size(); }

  /** @brief 由 ClDeviceInfo 生成设备键（优先 PCI id，其次 vendor:name:eu）。*/
  static std::string deviceKey(const ClDeviceInfo & d);

private:
  bool                              enabled_ = true;
  int                               abi_ = 0;   // 0 = 文件未标注（向后兼容，接受）
  std::string                       device_id_;
  std::string                       source_hash_;  // R45: kernel 源码指纹（数值/编译契约）
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
double expectedOps(const OpSignature & sig, const ClDeviceInfo & dev);

/** @brief R47: 按足迹插值的 copy(read+write) 带宽（GB/s）。锁频实测曲线；
 *  用于内存 roofline 与小算子流式成本估计。 */
double copyBwGbps(double footprint);
/** @brief R47: 小算子每 dispatch 的 launch 地板（µs）。 */
extern const double kSmallLaunchUs;

/** @brief R47/R49: 一个候选在整网里的**瞬时占用压力**（并发线程 × 每 WG 足迹字节）。
 *  线性、可加、设备无关（只依赖候选 options 与 shape）。是 L3/占用会计的单一真相源：
 *  `globalRetune` 的 L3 模拟与 `resolveLayoutMinCut` 的节点代价共用。*/
double occupancyPressure(const TuningEntry & e, const OpSignature & sig);
/** @brief R47 标定：L3 miss 折算 (1/BW_DRAM − 1/BW_L3)，单位 ms/byte。*/
extern const double kL3SpillPerByteMs;

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

}  // namespace infvino

#endif  // INFVINO__TUNING_HPP_
