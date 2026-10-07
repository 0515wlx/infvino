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
 * R66: 可选传入候选 `TuningEntry`。带上时，内存 roofline 用**二维有效带宽**
 * `effectiveBwGbps(bytes, occupancyThreads(e,s))`（R55：并发/MLP 因子 + 足迹容量因子），
 * 使低并发候选的内存墙更低；不带时退回 `copyBwGbps(footprint)`（旧行为）。
 *
 * @param dev 设备信息（EU 数等，用于网格占用修正）。
 * @param e   候选（提供 options 以算并发线程数）；nullptr = 旧行为。
 */
double expectedOps(const OpSignature & sig, const ClDeviceInfo & dev,
                   const TuningEntry * e = nullptr);

/** @brief R47: 按足迹插值的 copy(read+write) 带宽（GB/s）。锁频实测曲线；
 *  用于内存 roofline 与小算子流式成本估计。 */
double copyBwGbps(double footprint);
/** @brief R47: 小算子每 dispatch 的 launch 地板（µs）。 */
extern const double kSmallLaunchUs;

/**
 * @brief R69: 候选的**占用/访存几何契约**（候选侧 ↔ L3 溢出模拟的接口）。
 *
 * 整网 L3 溢出模拟（`L3Model`）对每个节点需要一个「在飞占用压力」R。R 由候选的**执行
 * 几何**决定：`concurrent` 个并发内存请求源，每个触达 `perWgBytes` 字节 →
 * `R = concurrent × perWgBytes`（= `occupancyPressure`）。
 *
 * 语义（与 R47/R55 标定口径一致）：
 *   * `concurrent` = 该候选的总 work-item 数（声明**原始值**；消费方 `occupancyStats`
 *     按 8192 饱和夹取）；
 *   * `perWgBytes` = 一个 work-**group** 触达的**输入激活**字节（权重/常量视作常驻不计、
 *     输出不计，与旧的 `occupancyPressure` 口径一致）；
 *   * `valid=false` → 该候选不计占用（launch/带宽受限的小算子，其外溢并入流式成本）。
 *
 * 此前这套几何散在 `Tuning.cpp::occupancyStats` 的 op 判据里（R48 §10.6-B 的「加族只加
 * 声明未落地」缺口）——新族/新候选不会自动进入 spill 模型，且 conv3x3 的四个 kernel、
 * split-K、GEMV、非 blk depthwise 都落进错误的判据。R69 把它声明化进 `KernelFamily`，
 * 注册表成为「候选 / 布局 / 上限 / **访存几何**」的完整单一真相源。
 */
struct MemContract
{
  bool   valid = false;
  double concurrent = 0.0;   ///< 原始总 work-item 数（消费方按 8192 夹取）
  double perWgBytes = 0.0;   ///< 每个 work-group 触达的输入激活字节
};

/** @brief R69: 从 kernel 编译选项串里取 `-D<key>=<int>`（找不到返回 `dflt`）。
 *  `key` 含完整 token（如 `"-DOBW="`）；供族声明的访存几何回调解析自己的 knob。*/
int tuningOptionInt(const std::string & options, const std::string & key, int dflt);

/** @brief R47/R49: 一个候选在整网里的**瞬时占用压力**（并发线程 × 每 WG 足迹字节）。
 *  R69: 改为**注册表优先**——按候选 kernel 名反查族，读族声明的 `MemContract`；
 *  未声明（小算子/未知）时回退到 legacy op 判据。`INFVINO_LEGACY_OCCUPANCY=1` 强制回退
 *  （A/B 消融）。线性、可加、设备无关（只依赖候选 options 与 shape）。是 L3/占用会计的
 *  单一真相源：`globalRetune` 的 L3 模拟与 `resolveLayoutMinCut` 的节点代价共用。*/
double occupancyPressure(const TuningEntry & e, const OpSignature & sig);
/** @brief R55: 某候选的**并发线程数** C = min(总线程数, 8192)（与 occupancyPressure 同源；
 *  供二维内存 roofline 的并发/MLP 因子使用）。小算子/未知 op 返回 0。*/
double occupancyThreads(const TuningEntry & e, const OpSignature & sig);
/** @brief R55 (`l3couple` 标定)：并发/MLP 因子（(0,1]，1=饱和）。有效行吞吐随并发线程数
 *  上升，约 **1024 线程**饱和；小网格欠占用 → 因子 <1（解释「小算子 prologue 受限」）。
 *  这给 `copyBwGbps` 补上此前缺失的**并发维**。*/
double l3MlpFactor(double threads);
/** @brief R55 (`l3couple` 标定)：L3 容量因子 g(R)（(0,1]，1=全命中）。R = 在飞驻留足迹
 *  字节（= occupancyPressure）。私有 tile 形态下平台延伸到 **~2 MB**，之后平滑下降，
 *  DRAM 平台 ~0.04——而非标称 3.75 MB。*/
double l3CapacityFactor(double residentBytes);
/** @brief R55: 二维有效带宽 = copyBwGbps(bytes) × l3MlpFactor(threads)（逐节点、可加）。
 *  `bytes` = 该节点的单遍工作集足迹（决定自有效容量断崖）；`threads` = 并发度（MLP）。
 *  跨算子的容量污染不在此项，由 `L3Model` 的全局 LRU 模拟/定价处理。*/
double effectiveBwGbps(double bytes, double threads);
/**
 * @brief R67: 一个算子的**单遍内存字节**（输入+输出+权重/激活），用于内存代价估计。
 * 与 `expectedOps` 的足迹口径一致；未知 op 返回 0。
 */
double singlePassBytes(const OpSignature & sig);
/**
 * @brief R67: 候选的**容量感知内存惩罚**（ms，逐节点、可加、≥0）。
 *
 * 只补「有效容量不足导致的**额外**内存时间」：
 *   `penalty = (1/g(R) − 1) · bytes / (BW_mlp(C)·1e9) · 1e3`
 * 其中 `R = occupancyPressure(e,s)`（在飞足迹）、`C = occupancyThreads(e,s)`、
 * `g = l3CapacityFactor(R)`（R55 曲线）、`BW_mlp = l3MlpFactor(C)`、`bytes = singlePassBytes(s)`。
 * `R` 小（放得下）时 `g=1` → 惩罚 0。量纲 ms，直接加到**布局节点代价**上；默认关（见 R67）。
 */
double capacityMemPenaltyMs(const TuningEntry & e, const OpSignature & sig);
/** @brief R47/R55 标定：L3 miss 折算 (1/BW_DRAM − 1/BW_L3)，单位 ms/byte。*/
extern const double kL3SpillPerByteMs;

/**
 * @brief R62: L3 标定的**实测/公开资料锚点**（i5-1135G7，Iris Xe / Tiger Lake）。
 *
 * R59 曾把 L3 容量设为 `8.0e6`——那是 **CPU 的** L3（`lscpu` / sysfs index3），不是 GPU 的。
 * R62 用**独立的几何测量**纠正：
 *   * 步进冲突探针（`kernel_bench --op l3conflict`）显示冲突周期 = **512 行（32 KB）**；
 *   * 容量扫描（stride 1）膝点 ≈ **4 MB**，8 MB 后骤降；
 *   * 公开 PRM（TGL Vol.7）bank = 480 KB = 120 way × 64 set × 64 B，8 bank ⇒
 *     `8 × 480 KiB = 3840 KiB = 3.75 MiB = 512 × 120 × 64 = 3,932,160 B`。
 *   三者一致 ⇒ **GPU L3 = 3.75 MiB**。
 *
 *   * `l3PhysicalBytes()`  = 3,932,160（GPU **私有** L3 Data Cache = 8 bank × 480 KiB）；
 *   * `l3LlcBytes()`       = 8,388,608（CPU+iGPU **共享 LLC**，即 CPU sysfs index3 的 8 MiB；
 *     TGL-U 4 核；TGL-H 8 核为 24 MiB）；
 *   * `l3WarmCapBytes()`   = `l3PhysicalBytes() + l3LlcBytes()` ≈ 12.3 MB（**跨算子热重用**的
 *     有效容量：GPU L3 逐出后仍可命中共享 LLC；R63 单缓冲别名实测阈值 ≈12 MB，吻合）；
 *   * `l3CpuL3Bytes()`     = 同 `l3LlcBytes()`（历史名，保留）；
 *   * `l3PrivateCapBytes()`= 2.0e6（R55 §2.3：单 kernel 私有 tile 的膝点）；
 *   * `l3DefaultCapBytes()`= 跨算子热重用容量 = `l3WarmCapBytes()`（`L3Model` 模拟重用张量）；
 *     `INFVINO_L3_GEOM=1` 用**仅 GPU 私有 L3** 3.75 MiB 做 A/B；
 *   * `l3DefaultAnchorBytes()` = 2.0e6（`INFVINO_L3_GEOM=1` 时 1.0e6）；
 *   * `l3DramBwGbps()`=20.0、`l3SramBwGbps()`=145.0（R55 copy 实测平台/峰值）。
 *
 * ⚠️ 两级层次：GPU 私有 L3（3.75 MiB）→ 共享 LLC（8 MiB）→ DRAM。单级 LRU 的「有效容量」
 * 因此是 **两者之和**（≈12 MB），而不是任一单级。见 docs/round63。
 */
double l3PhysicalBytes();
double l3CpuL3Bytes();
double l3LlcBytes();
double l3WarmCapBytes();
double l3PrivateCapBytes();
double l3DefaultCapBytes();
double l3DefaultAnchorBytes();
double l3DramBwGbps();
double l3SramBwGbps();

// ============================================================================
// R71: **绑定墙（binding wall）归因模型** —— 诊断/报告层，不参与任何选择。
//
// 此前 `expectedOps` 的「计算类」上限只有一把尺子（ISA 指令配额 × 网格），
// 「内存类」只有一条 `copyBwGbps` 曲线。R71 把两个开发中反复出现的物理维度显式化：
//
//   1. **带宽受限**：数据足迹按 **GPU 私有 L3（3.75 MiB）→ 共享 LLC（8 MiB）→ DRAM**
//      三段拆分，各段用自己的带宽（`memTierTime`）。这使「撞 GPU-L3 容量」与
//      「已经溢出到 CPU 共享 LLC」区分开——二者的每字节代价差 ~3×。
//   2. **计算受限**：FPU 跑不满不是「指令配额」单独能解释的，另有
//      (a) **寄存器文件大小**：128 GRF/线程里只有 ~40 给数据（IGC overhead ~85–95），
//          累加器链数 `ACC` 被顶到 ~16–19，若 `7·ACC < FMA 延迟 L≈30` 就掉进延迟受限；
//      (b) **SLM 带宽 / 容量耦合**：operand 从 SLM 流入的速率、以及 SLM 足迹
//          （每 WG ≤64 KiB、每 lane ≤73 B 否则掉 SIMD 宽度）。
//
// 所有系数都来自本项目实测/公开资料，见 docs/round71-*、docs/register-model.md、
// docs/xe-lp-isa.md。整层是**纯 host、零 GPU**，可离线单测。
// ============================================================================

/** @brief R71: 数据/算子撞的**那一堵墙**。*/
enum class WallKind
{
  FmaIssue,      ///< ISA 指令配额（mad 占比）——纯发射上限
  RegIlp,        ///< 寄存器文件大小 → ACC 上限 → 延迟受限（7·ACC < L）
  SlmBandwidth,  ///< SLM 读带宽喂不满 FMA
  SlmCapacity,   ///< SLM 足迹（>64 KiB/WG 或 >73 B/lane 掉 SIMD 宽度）
  GpuL3,         ///< 带宽：命中 GPU 私有 L3（3.75 MiB）
  SharedLlc,     ///< 带宽：GPU-L3 溢出，但命中 CPU 共享 LLC（8 MiB）
  Dram,          ///< 带宽：两级都溢出 → DRAM
  Launch,        ///< dispatch/网格地板或有界归约
  Unknown,
};
const char * wallKindName(WallKind w);

/** @brief R71: 寄存器/发射/SLM 侧的物理常数（全部有实测出处）。*/
constexpr double kEfResidentThreads = 7.0;     ///< 每 EU 驻留线程数（R27 实测）
constexpr double kFmaLatencyCyc = 30.0;        ///< FP16 FMA 延迟 L（R10/R18）
constexpr double kSlmBytesPerWg = 65536.0;     ///< 每 WG SLM 上限（clinfo）
constexpr double kSlmBytesPerLaneCliff = 73.0; ///< >73 B/lane 掉 SIMD 宽度（Intel guide）
constexpr double kSlmReadBwGbps = 350.0;       ///< SLM 流式读带宽（R71 实测 300–500 中值）

/** @brief R71: 一个算子的**单遍足迹按三级缓存拆分**（诊断；见 `memTierTime`）。*/
struct MemTierTime
{
  double footprint_bytes = 0.0;
  double l3_bytes = 0.0, llc_bytes = 0.0, dram_bytes = 0.0;   ///< 字节拆分
  double l3_ms = 0.0, llc_ms = 0.0, dram_ms = 0.0;            ///< 各段耗时（用各段带宽）
  double total_ms = 0.0;                                      ///< = l3+llc+dram（串行下界）
  WallKind tier = WallKind::GpuL3;                            ///< 主导段
};
/** @brief 把一个算子单遍足迹按 GPU-L3/LLC/DRAM 三段拆分（每段用各自实测带宽）。*/
MemTierTime memTierTime(const OpSignature & sig);

/** @brief R71: 计算侧的多堵墙（ISA 配额 / 寄存器 ILP / SLM 带宽 / SLM 容量）。*/
struct ComputeWall
{
  bool   fma_family = false;   ///< 是否 FMA 主导（conv3x3/gemm/conv1x1_blk/gemv）
  bool   instr_quota = false;  ///< 指令配额主导（depthwise：地址/边界指令）
  double acc = 0.0;            ///< 每线程独立累加链数
  double isa_ops = 0.0;        ///< 指令配额上限（ops/EU/cyc）
  double rf_ilp = 1.0;         ///< min(1, T_res·ACC/L)
  double rf_ops = 0.0;         ///< isa_ops × rf_ilp
  double slm_bw_ops = 0.0;     ///< SLM 读带宽上限（0 = 不适用/未知）
  double slm_bytes_per_wg = 0.0;
  double slm_bytes_per_lane = 0.0;
  bool   slm_capacity_ok = true;  ///< 足迹同时满足 ≤64 KiB/WG 与 ≤73 B/lane
  WallKind binding = WallKind::Unknown;
};
ComputeWall computeWall(const OpSignature & sig, const TuningEntry & e, const ClDeviceInfo & dev);

/** @brief R71: 一个节点的**绑定墙判决**（诊断；`measured_ms` 来自实测缓存）。*/
struct WallVerdict
{
  WallKind    binding = WallKind::Unknown;
  MemTierTime mem;
  ComputeWall comp;
  double      measured_ops = 0.0;   ///< 实测 ops/EU/cyc
  double      fma_ops = 0.0;        ///< 计算侧硬上限（含寄存器/SLM derate）
  double      mem_ops = 0.0;        ///< 内存侧硬上限（三段足迹 roofline）
  double      hard_ops = 0.0;       ///< min(fma, mem)
  double      ratio_to_wall = 0.0;  ///< measured / hard（1 = 已贴住该墙）
};
WallVerdict attributeWall(const OpSignature & sig, const TuningEntry & e, const ClDeviceInfo & dev,
                          double measured_ms);

/**
 * @brief R60: reorder（bfyx↔fsv16 等布局搬运）的**精确可加**成本模型。
 *
 * 把实测的单趟 `#reorder` 拆成两个**逐张量、可加**的项：
 *   `reorder_ms = launch_floor + (read_bytes + write_bytes) / BW(state)`
 * 其中 `launch_floor` 是一次 dispatch 的固定开销，`BW(state)` 取决于输入是否已在 L3：
 *   * 输入已在 L3（`input_resident=true`）→ 用 `l3SramBwGbps()` 档；
 *   * 输入需从 DRAM 取（默认）→ 用 `kReorderStreamBwGbps` 档（≈ 实测单趟 GB/s）。
 *
 * 这样「多趟 reorder」的合计不再是「张量数 × 单趟冷读价」，而是按各自状态求和——
 * 可加性由此恢复（`kernel_bench --op reorder_seq` 实测验证，见 docs/round60-*）。
 * 常数由本机 `kernel_bench --op reorder` 自行标定（非任何厂商内部文档）。
 */
extern const double kReorderLaunchMs;      ///< 一次 reorder dispatch 的 launch floor（ms）
extern const double kReorderStreamBwGbps;  ///< 冷输入流式搬运带宽（GB/s，实测）
/**
 * @brief R60: 顺序 dispatch 之间的**逐次间隔**（ms），由 `kernel_bench --op reorder_seq` 实测。
 *
 * 事件计时的单趟 `#reorder.ms` 不含「前一个 kernel 结束→下一个开始」的硬件间隔；把 N 趟
 * 相加会系统性少算 `N × kReorderDispatchGapMs`。锁频下实测：全热（共享输入+输出）N=16 的
 * 每次成本比单趟高 ~12.4 µs，即该间隔。它给出「多趟 reorder 为什么不可加」的第一项。
 */
extern const double kReorderDispatchGapMs;
double reorderCostMs(double read_bytes, double write_bytes, bool input_resident);

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
