// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// PlanModel —— 自研 OpenCL kernel 的整网执行器（库化自 tools/kernel_run）。
//
// 读取由 scripts/onnx2plan.py 生成的“算子计划”(plan)，用自研 kernel 执行完整模型。
// 计划为文本，每行一条（# 注释）：
//   input  <name> <d0> <d1> ...              # 运行时输入（fp16，NCHW 行主序）
//   init   <name> <file.bin> <d0> <d1> ...   # 权重（fp16，路径相对 plan 文件所在目录）
//   tensor <name> <d0> <d1> ...              # 中间激活
//   node   <op> <in_csv|-> <out_csv> [k=v ...]
//   output <name>                            # 输出（可多行）
// 约定：所有张量为 fp16、行主序；reshape/flatten 为视图（零拷贝别名）。
//
// PlanModel 不做图优化、不做类型转换：输入/输出均为 fp16，预处理与解码在更上层完成。
#ifndef INFVINO__PLAN_MODEL_HPP_
#define INFVINO__PLAN_MODEL_HPP_

#include <CL/cl.h>

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "infvino/ClRuntime.hpp"
#include "infvino/Tuning.hpp"
#include "infvino/L3Model.hpp"

namespace infvino
{

/**
 * @brief 执行一份算子计划的整网模型（单线程使用；内部持有上下文/队列）。
 *
 * 典型用法：
 * @code
 *   infvino::PlanModel model("models/yolov8n-pose.plan");
 *   model.setInput(fp16_input);       // inputNumel() 个 fp16
 *   model.run();
 *   std::vector<uint16_t> out(model.outputNumel(0));
 *   model.readOutput(0, out.data());
 * @endcode
 */
class PlanModel
{
public:
  /**
   * @param plan_path     计划文件路径。
   * @param kernel_dir     .cl 源码目录（默认编译期注入的源码树 kernels/）。
   * @param platform       OpenCL 平台序号，-1 = 自动选择含 GPU 的平台。
   * @param device_index   平台内设备序号。
   * @param profiling      是否开启 CL_QUEUE_PROFILING_ENABLE（算子计时/基准需要）。
   */
  explicit PlanModel(
    const std::string & plan_path, const std::string & kernel_dir = INFVINO_KERNEL_DIR,
    int platform = -1, int device_index = 0, bool profiling = false);
  ~PlanModel();

  PlanModel(const PlanModel &) = delete;
  PlanModel & operator=(const PlanModel &) = delete;

  const std::string &        planPath() const { return plan_path_; }
  const std::string &        inputName() const { return input_name_; }
  const std::vector<int64_t> & inputDims() const { return input_dims_; }
  size_t                     inputNumel() const;

  size_t                          outputCount() const { return outputs_.size(); }
  const std::string &             outputName(size_t i) const { return outputs_.at(i); }
  const std::vector<int64_t> &    outputDims(size_t i) const;
  size_t                          outputNumel(size_t i) const;

  /** @brief 用 fp16 主机缓冲（inputNumel() 个元素）填充图输入。 */
  void setInput(const void * fp16_host);
  /** @brief 前向一次（单次，不含 warmup）。 */
  void run();
  /** @brief 把第 i 个输出拷贝到 fp16 主机缓冲（outputNumel(i) 个元素）。 */
  void readOutput(size_t i, void * fp16_host);

  /** @brief P0 诊断：按名字读取任意已声明张量（fp16 主机缓冲，numel 个元素）。
   *  返回 false 表示无此张量。用于逐层误差定位。 */
  bool readTensor(const std::string & name, void * fp16_host) const;
  /** @brief P0 诊断：任意张量的元素数；0 表示无此张量。 */
  size_t tensorNumel(const std::string & name) const;

  const ClRuntime &   runtime() const { return rt_; }
  const ClDeviceInfo &  device() const { return rt_.info(); }

  /** @brief 调优缓存（只读；由构造时按 INFVINO_TUNING_CACHE / config/tuning.json 加载）。*/
  const TuningCache & tuning() const { return tuning_; }

  /** @brief P0: 激活缓冲池统计（requested = 朴素总量；allocated = 真正 clCreateBuffer）。*/
  size_t poolRequestedBytes() const { return act_pool_.requestedBytes(); }
  size_t poolAllocatedBytes() const { return act_pool_.allocatedBytes(); }
  size_t poolBufferCount() const { return act_pool_.bufferCount(); }

  /**
   * @brief 对计划里的节点做离线自动调优（枚举候选 + GPU 计时），返回结果并按需合并进
   *        `tuning_`。仅在 profiling=true 的 ClRuntime 上有意义。
   *
   * @param ops       只调优这些 op（空 = 全部支持调优的 op）。
   * @param onlySubstr 只调优 profile tag 含此子串的节点（用于分批，空 = 不筛）。
   * @param limit     最多调优多少个节点（0 = 不限；用于安全分批）。
   * @param iters     每个候选的计时迭代数。
   * @param merge     是否把结果写进 tuning_。
   * @param verbose   打印每个节点的候选扫描明细。
   * @return 本次调优的条目（key = OpSignature::str()）。
   */
  std::map<std::string, TuningEntry> autotune(
    const std::vector<std::string> & ops = {}, const std::string & onlySubstr = "",
    int limit = 0, int iters = 30, bool merge = true, bool verbose = false,
    bool retune = false);

  /** @brief 列出计划里可调优的唯一签名（不触碰 GPU，用于分批/审计）。 */
  std::vector<std::string> tuningTargets(const std::vector<std::string> & ops = {}) const;
  /** @brief 计划里可调优节点的唯一签名对象（与 tuningTargets 同源，便于重算中间标准）。 */
  std::vector<OpSignature> tuningSignatures(const std::vector<std::string> & ops = {}) const;
  /** @brief 用当前 `expectedOps` 重算缓存命中项的 expected/ratio（**零 GPU**）。
   *  用于中间标准公式更新后，让既有实测数据立即按新标尺排序。返回更新的条数。 */
  int refreshExpected(const std::vector<std::string> & ops = {});
  /** @brief 把 current tuning_ 写回文件。 */
  bool saveTuning(const std::string & path) const { return tuning_.save(path); }

  /**
   * @brief R45 P0#4: **per-plan 选择覆盖**（解决共享签名缓存的跨模型稀释）。
   *
   * 全局最优常是**位置相关**的（同一 shape 在 y8/y11 的邻居/持久化不同），而 `tuning.json`
   * 是签名级、跨模型共享的——一份「选谁」无法两全。本机制把**计划级**的 per-node 选择存成
   * 独立工件：key = 节点输出张量名（唯一），运行时由 `choiceEntry` 最高优先消费。
   * `config/tuning.json` 退化为「可移植的隔离默认」，per-plan 工件承载「本图的全局最优」。
   *
   * 加载：`INFVINO_PLAN_TUNING`，否则 `<plan>.tuning.json`；`INFVINO_TUNING=off` 一并禁用。
   * 保存：`globalRetune()` 选出后由调用方 `savePlanOverrides(path)` 落盘。
   */
  bool   savePlanOverrides(const std::string & path) const { return plan_overrides_.save(path); }
  size_t planOverrideCount() const { return plan_overrides_.size(); }

  /**
   * @brief R44: **整网 busy 坐标下降回验** —— 修正 autotune 目标函数的根本缺陷。
   *
   * 隔离 bench 的 `min(ms)` 只是**局部代理**：候选在冷/空 cache 下的名次，不等于它在真实
   * 流水线里的名次（R43 §5.1 实测 9 个 conv1x1 配置「隔离更快、整网更慢」，因为候选互相
   * 争 L3/DRAM 与在飞占用）。本函数把目标函数从「单节点隔离 min」换成**整网 busy**
   * （`profile().busy_ms`，逐 kernel event 时间之和）：
   *
   *   1. 取隔离扫描保留的每个签名短名单（top-K）；
   *   2. 逐签名坐标下降：把候选赋给该签名的节点 → 重规划布局 → 重录 dispatch →
   *      跑整网 reps 次取 busy 最小值，保留使整网 busy 最小者；若相对现状改善 >
   *      `minGain`（0.5%）才接受；
   *   3. 多轮直到无签名改变；最终结果写回 tuning_（由调用方落盘）。
   *
   * 这直接修复「局部最优 ≠ 全局最优」：只有真正降低端到端 busy 的候选才会被选中。
   *
   * @param ops    只回验这些 op（空 = 所有有短名单的 op）。
   * @param iters  每个候选的整网测量重复次数（0 = 默认 3；取每次的最小 busy）。
   * @param topK   每个签名参与回验的候选上限（按隔离 ms 取前 K，默认 3）。
   * @param rounds 坐标下降轮数上限（默认 3）。
   * @param limit  最多回验的签名数（0 = 不限；安全分批用）。
   * @param margin 隔离 margin 剪枝：剪掉 `iso_ms > best_iso*(1+margin)` 的候选（默认 **0** =
   *               不剪）。⚠️ 隔离名次并不预测整网名次（R43：隔离更慢的 blk 可能整网更快），
   *               激进的 margin 会把「隔离慢、流水线快」的候选剪掉、重蹈局部最优。只有在
   *               候选极多且明确接受此风险时才设 >0。
   * @param budget 整网测量次数总预算（0 = 不限）；耗尽即停（抗组合爆炸 / GPU 风险）。
   * @return 实际改变选择的签名数；**`-1` = 未执行（NO-OP）**——无 profiling / 无候选短名单 /
   *         短名单与本 plan 不匹配。调用方（`kernel_autotune --global`）据此区分「跑了但没选出
   *         更好配置（0）」与「根本没跑（-1）」，避免 R44 #12 的静默空操作。
   *
   * 需要 profiling=true（否则测不到逐 kernel event 时间）与可运行的整网输入。
   */
  int globalRetune(const std::vector<std::string> & ops = {}, int iters = 0, int topK = 3,
                   int rounds = 3, int limit = 0, double margin = 0.0, int budget = 0);

  /**
   * @brief P3 在线调优：对缓存里尚未命中的签名，按顺序在线 benchmark 至多 `budget`
   *        个并 merge 进 tuning_（需要 profiling=true 的 runtime 才能计时）。
   *        与离线 `kernel_autotune` 共享同一套候选/中间标准；失败静默跳过。
   * @return 本次实际调优的签名数。
   */
  int onlineTuneMissing(int budget, int iters, const std::vector<std::string> & ops);

  /** @brief 最近一次 run() 的墙钟耗时（ms）。 */
  double lastRunMs() const { return last_run_ms_; }
  /** @brief 算子耗时表（仅 profiling=true 时填充）：op -> {ms, calls}。
   *  耗时在多次 run() 之间**累加**，便于对 warm 后的多次运行求平均；
   *  统计前先调用 clearProfile()。 */
  const std::map<std::string, std::pair<double, int>> & opProfile() const { return tprof_; }
  /** @brief P2: profiling 下 host 侧累计耗时（ms）：入队 clEnqueueNDRangeKernel 的调用耗时。
   *  与 opProfile 一样在多次 run 间累加，统计前用 clearProfile() 清空。
   *  用途：拆解 wall - busy，区分「入队/提交」「同步等待」「其余 host（含 setArg）」。 */
  double hostEnqueueMs() const { return prof_enqueue_ms_; }
  /** @brief P2: profiling 下 host 侧累计 clWaitForEvents 等待耗时（含 GPU 执行）。 */
  double hostWaitMs() const { return prof_wait_ms_; }

  /**
   * @brief Phase 2: 结构化剖面 —— 逐节点 GPU 时间 + 结构计数 + 内存 + host 分段。
   *
   * 这是 busy/net/e2e 分析框架（`scripts/analyze_budget.py`）的机器可读来源：
   * 结构计数（节点数/dispatch/融合/alias/布局）任何时候都有效；逐节点/逐算子时间
   * 需要 profiling=true，且在多次 run() 间**累加**（除以 iters 得每帧值）。
   * 统计前先调用 clearProfile()。
   */
  struct PlanProfile
  {
    struct NodeRow
    {
      int         index = 0;
      std::string op;        ///< plan 节点算子名（conv3x3 / conv1x1 / depthwise / ew_binary ...）
      std::string tag;       ///< 运行时 profile tag（含 shape / 数据通路）
      std::string signature; ///< 该节点的调优签名（与 tuning.json 的 key 一致，可精确 join）
      double      ms = 0.0;  ///< 累计 GPU 时间（含该节点的 reorder 等附属 dispatch）
      int         calls = 0; ///< 累计 dispatch 次数
    };
    std::vector<NodeRow> nodes;
    int    plan_nodes = 0, dispatched_nodes = 0, skipped_nodes = 0, dispatches = 0;
    int    reorder_calls = 0;
    double reorder_ms = 0.0;
    int    fusions_res = 0, fusions_concat = 0;
    int    fsv16_tensors = 0;
    size_t pool_requested = 0, pool_allocated = 0, pool_buffers = 0;
    double wall_ms = 0.0, busy_ms = 0.0, enqueue_ms = 0.0, sync_ms = 0.0;
  };
  PlanProfile profile() const;

  /** @brief 清空算子耗时表、逐节点耗时表与 host 分段计时（在统计循环前调用）。 */
  void clearProfile()
  {
    tprof_.clear();
    node_ms_.clear();
    node_calls_.clear();
    node_tag_.clear();
    prof_enqueue_ms_ = prof_wait_ms_ = 0.0;
  }

private:
  struct Tensor
  {
    std::vector<int64_t> dims;
    cl_mem               mem{nullptr};
    // P0-offset 实验：mem 可能是 arena 的子 buffer；base/off 记录它在底层 arena 里的
    // 位置，供 copy_c 的连续切片别名（clCreateSubBuffer 不能基于子 buffer 再切）。
    cl_mem               base{nullptr};
    int64_t              base_off{0};
    // R36 (P1-layout): 物理布局。false=普通 NCHW(bfyx)；true=阻塞式
    // b_fs_yx_fsv16 [C/16][H][W][16]。仅当生产者是 blocked conv、所有消费者也是
    // blocked conv、且 Cout%16==0 时才置位（见 planBlockedLayout）。同一 size 的
    // 元素数不变，因此不影响内存池分配。
    bool                 fsv16{false};
    // R52: 该张量底层缓冲实际分配的元素数（pool arena/sub-buffer 或独立分配）。
    // fsv16 需要 ceil(C/16)*16*H*W 个元素；cap_elems 用于「池/布局一致性探针」
    // （INFVINO_POOL_ALIAS_PROBE）验证「被标记 fsv16 的张量缓冲确实放得下补齐布局」。
    int64_t              cap_elems{0};
    int64_t              numel() const
    {
      int64_t n = 1;
      for (auto d : dims) n *= d;
      return n;
    }
  };
  struct Node
  {
    std::string                        op;
    std::vector<std::string>           ins, outs;
    std::map<std::string, std::string> attr;
  };

  void    parse();
  /** @brief P0: assign activation tensors to the shared buffer pool by liveness. */
  void    allocateActivations();
  /** @brief R36 (P1-layout): 自动布局规划 —— 由 autotune 选出的 conv kernel 驱动，
   *  把「生产者是 blocked conv 且所有消费者也是 blocked conv」的激活张量标记为
   *  fsv16，使 blocked 链内零 reorder。`INFVINO_NO_BLOCK_LAYOUT=1` 关闭。 */
  void    planBlockedLayout();
  /**
   * @brief R52 诊断/守卫：池 + 布局一致性探针。
   *
   * 遍历所有被标记 `fsv16` 的张量，验证：
   *   1. 其底层缓冲容量 (`cap_elems`) ≥ `ceil(C/16)*16*H*W`（放得下补齐布局）；
   *   2. 其生产者当前选中的 kernel 确实声明 `canOutFsv16`（否则「标记 fsv16 但
   *      生产者写 NCHW」会在整网静默产生错误激活）。
   *
   * `INFVINO_POOL_ALIAS_PROBE=1` 时打印所有违例；违例数量 >0 返回 false。
   * 作为硬守卫：`resolveLayoutChoices` 在落地布局后会调用它，任何违例都说明
   * 分配补齐集合（`mayBeFsv16`）与布局标记判据发生了漂移。
   */
  bool    poolAliasProbe(const char * where, bool verbose = false) const;
  /** @brief R52 硬守卫：张量的底层缓冲是否放得下 b_fs_yx_fsv16 补齐布局。 */
  bool    mayMarkFsv16(const std::string & name) const;
  /**
   * @brief R38: 联合 (族, 布局) 选择不动点。
   *
   * 单看 kernel ms 选 conv3x3_blk 会忽略「它强制给输入做 bfyx→fsv16 重排」的成本，
   * 而重排是否发生又取决于布局（生产者是否同链、消费者是否全是 blocked）——是自指。
   * autotune 会把每个 sig 的 **两个备选** 写进缓存：`sig#blk` / `sig#non`，以及该输入的
   * 一次重排实测 `sig#reorder`。本函数在**计划期**迭代：
   *   1) 按当前 per-node 选择跑 planBlockedLayout；
   *   2) 每个 conv3x3 节点取 `cost(blk)=#blk.ms + (输入是否已 fsv16 ? 0 : #reorder.ms)`、
   *      `cost(non)=#non.ms`，选小者；
   *   直到稳定。结果写 `node_choice_`，dispatch/布局/`convWillUseBlk` 一致消费。
   *
   * 缓存里没有 `#blk/#non` 时退化为一次 `planBlockedLayout()`（与 R36 行为一致）。
   * 因为不动点按**每个 plan 的图**跑，同一 sig 在 y8/y11 可得到不同选择——这解决了
   * 「sig-cache 跨模型共享、无法区分布局」的问题。
   */
  void    resolveLayoutChoices();
  /** @brief R38: per-node 覆盖（不动点结果），未命中则回退到签名缓存。 */
  const TuningEntry * choiceEntry(size_t ni, const OpSignature & sig) const;
  /** @brief R55: 构造 L3 全局模拟视图（拓扑序；占用 = occupancyPressure，小算子 = 其流式足迹）。
   *  供 `predictNet` 的全局溢出与 `resolveLayoutMinCut` 的逐节点定价共用（单一真相源）。*/
  std::vector<L3Access> buildL3Access() const;
  /** @brief R55: 布局选择的离线整网评分 = Σ 选中 kernel ms + Σ 未持久化 blk 输入的 reorder
   *  + L3 spill_ms（小算子为布局不变量，略）。用于 L3 定价提案的**拒绝门**（只防回归）。*/
  double layoutModelScore() const;
  /** @brief R59: 上面评分的**分项**（kernel / reorder / spill），用于标定与验证。*/
  struct LayoutScore
  {
    double kernel = 0.0, reorder = 0.0, spill = 0.0;
    double total() const { return kernel + reorder + spill; }
  };
  LayoutScore layoutModelBreakdown() const;
  /** @brief R38/R49: 一个节点在缓存里的两种布局备选 + 一趟 reorder 成本。*/
  struct LayoutAlt
  {
    TuningEntry blk, non, reorder;
    // R50: blocked 族在「输出 fsv16」时的实测成本（同一 kernel + OUT_FSV16=1）。
    // 布局契约 canOutFsv16=true 表示该族可直出 fsv16；此时其成本用本项而非 bfyx 输出
    // 的 `blk.ms`——二者对 depthwise_blk 相差 ~2×（lane 合并写 vs 跨通道散写）。
    TuningEntry blkFsv16;
    bool        has = false;
  };
  /**
   * @brief R57: min-cut 提案的结构（供「按连通分量局部化验收门」使用）。
   *
   * `varName`/`labels` = 每个变量的张量名与提案标签；`comps` = 分量（每个是参与节点在
   * `nodes` 里的下标列表，按共享张量连通）；`nodes` 是参与 min-cut 的节点，
   * `nodeVarA/B` 是它们在变量表里的输入/输出变量号（用于把分量映射回节点选择）。
   */
  struct MinCutProposal
  {
    std::vector<std::string>      varName;
    std::vector<int>              labels;
    std::vector<std::vector<int>> comps;
    std::vector<size_t>           nodes;
    std::vector<int>              nodeVarA, nodeVarB;
  };
  /**
   * @brief R49 试点：把布局决策建模成二元标注的**精确最小割**（LayoutSolver）。
   *
   * 只对 `op == "conv1x1"` 且缓存含 `#blk/#non/#reorder` 的节点生效（一族试点，
   * `INFVINO_LAYOUT_MINCUT=1` 开启）。变量 = 相关激活张量的布局（NCHW/FSV16）；
   * 节点代价表（输入布局 × 输出布局 → ms，含 reorder）分解成 unary + 吸引项；
   * 网络输入/输出、以及非本族消费者/生产者的张量钉死 NCHW。
   *
   * 成功返回 true 并写好 `node_choice_` 与各张量 `fsv16`；任何不适配（非本族 alt、
   * 表非 submodular、变量为空）都返回 false，调用方回退到既有不动点/启发式。
   * R57：非 submodular 用 relaxed 表（不再 return false）；越界变量局部钉死。
   * `prop` 非空时填出提案结构（连通分量），供调用方做**分量级验收**。
   */
  bool    resolveLayoutMinCut(const std::vector<LayoutAlt> & alt, MinCutProposal * prop = nullptr);
  /** @brief R49: mincut 是否已接管布局（为真时 planBlockedLayout 直接返回，保留标注）。*/
  bool    mincut_active_ = false;

  /** @brief R45 P0#4: per-plan 覆盖的节点键（用节点输出名，保证唯一且跨重生成稳定）。*/
  OpSignature planNodeKey(const std::string & out_name) const;
  /** @brief R36: 该 conv3x3 节点是否会被纳入 blocked 通路（与 dispatch 同判据）。 */
  bool    convWillUseBlk(const Node & n) const;
  /** @brief R30c: fuse `concat4 -> conv1x1` into a CAT4 gemm (skip concat materialisation). */
  void    fuseConcatConv1x1();
  /** @brief R33: fuse `conv -> ew_binary(add)` into the conv epilogue (RES), removing the
   *  standalone elementwise add launch. Only for producers whose kernel supports RES. */
  void    fuseResidualAdd();
  /** @brief R51 D5: fuse `x * scale[c]` (channel-broadcast Mul, e.g. SE) into the conv1x1
   *  that consumes it as its activation, so the Mul pass + its tensor disappear; the conv
   *  reads the unscaled value + a per-input-channel `scale` (kernel `-DMUL_SCALE=1`). */
  void    fuseChannelScaleMul();
  /** @brief R56 D5: fuse the YOLO SPPF block (`concat4(x, mp(x), mp²(x), mp³(x))` with a
   *  chained MaxPool5s1p2) into a single `sppf_concat4` kernel (3 launches + 3 buffers saved;
   *  bit-exact, max is associative). */
  void    fuseSppfConcat();
  void    buildKernels();
  void    releaseKernels();
  void    dispatch(const Node & n);
  Tensor & ref(const std::string & name);
  int     attrInt(const Node & n, const char * key, int def) const;
  cl_mem  ovWeight(const std::string & name, Tensor & w, int Cout, int Cin);
  /** @brief R25: repack weights to OpenVINO os_is_yx_isv16_osv16 for conv3x3_blk. */
  cl_mem  blkWeight(const std::string & name, Tensor & w, int Cout, int Cin);
  /** @brief blocked 1x1 (conv1x1_blk) 的 os_is_yx_isv16_osv16 权重重排 [Cout/16][Cin/16][isv16][osv16]。 */
  cl_mem  blk1x1Weight(const std::string & name, Tensor & w, int Cout, int Cin);
  /** @brief blocked depthwise (depthwise_blk) 的 [C/16][K][K][16] 权重重排。 */
  cl_mem  blkDwWeight(const std::string & name, Tensor & w, int C, int K);
  /** @brief Cin<=4 首层 conv：把权重从 [Cout][Cin*9] 重排成 [Cin*9][Cout]（通道连续，
   *  让 conv3x3_cin3 的跨通道 half2 读是连续/广播的）。 */
  cl_mem  cin3Weight(const std::string & name, Tensor & w, int Cout, int Cin);
  /** @brief R25: reorder a conv input bfyx -> b_fs_yx_fsv16 (cached scratch). */
  cl_mem  blkInput(const std::string & name, Tensor & x, int Cin, int H, int W);
  /** @brief R31: zero-padded depthwise input Xp[C][Hp][Wpad] (cached by tensor name).
   *
   *  Ensures the buffer exists and its zero border is written once (at first use);
   *  the caller then runs `depthwise_pad` to refresh the interior every frame.
   *  Sized for DW_TW=8 (the largest candidate) so any tuner TW reads in-bounds.
   *  @param HpOut/WpadOut 回传实际 padded 尺寸（传给 depthwise_vp）。
   */
  cl_mem  dwPadInput(const std::string & name, int Cin, int H, int W, int K, int S, int P,
                     int * HpOut, int * WpadOut);
  /** @brief Build (once) and cache a kernel keyed by source|name|options. */
  cl_kernel getKernel(const std::string & src, const std::string & name, const std::string & opts);

  // ---------------------------------------------------------------------------
  // P2: per-node dispatch cache.
  //
  // run() 的每个节点每帧都要做一遍「签名/查表/拼 kernel key/attrInt/拼 tag/setArg」
  // 的 host 工作（P2 实测 host ≈ 26 µs/node，~4× OpenVINO）。计划是静态的、内存池
  // 在 parse 期一次性分配，所以每个节点的 kernel、参数、网格在帧间**不变**。
  //
  // 策略：首帧按原逻辑执行并**录制**「克隆 kernel + 已设参数 + 网格 + tag」；之后
  // 每帧直接重放（replay），零签名/零字符串/零 setArg。克隆 kernel（clCreateKernel）
  // 让每个 cmd 拥有独立的参数状态，参数只在录制时设一次——数值语义与首帧完全一致。
  // ---------------------------------------------------------------------------
  struct PlanArg
  {
    cl_uint                index{0};
    size_t                 size{0};
    std::vector<unsigned char> bytes;
  };
  struct PlanCmd
  {
    cl_kernel   k{nullptr};
    cl_uint     dim{0};
    size_t      gws[3]{1, 1, 1};
    size_t      lws[3]{1, 1, 1};
    bool        useLws{false};
    std::string tag;
  };
  /** @brief P2: 记录参数（capture 期）或直接下传 OpenCL。 */
  void    setArg(cl_kernel k, cl_uint index, size_t size, const void * value);
  /** @brief P2: 入队一个 P2 受管的 dispatch（capture 期顺带克隆+录制）。 */
  cl_event enqueueCmd(cl_kernel k, cl_uint dim, const size_t * gws, const size_t * lws,
                      const char * tag, bool useLws);
  /** @brief P2: 重放第 ni 个节点录制好的 dispatch 列表。 */
  void    replayNode(size_t ni);
  /** @brief Phase2: 记录第 ni 个节点的 GPU 时间/tag（reorder 等附属 dispatch 也计入该节点）。 */
  void    noteNode(size_t ni, const std::string & tag, double ms);
  /** @brief 求一个节点的调优签名（与 `tuningTargets` 同源；不支持的 op 置 ok=false）。 */
  OpSignature nodeSignature(const Node & n, bool * ok = nullptr) const;
  /** @brief R51 D5: conv1x1 节点的签名（含 RES/MUL_SCALE 位；从 ins 槽 3/4 派生）。 */
  OpSignature conv1x1Sig(const Node & n, int Cout, int N, int Cin, int act) const;
  /** @brief P2: 克隆一个 kernel（独立参数状态，用于跨帧跳过 setArg）。 */
  cl_kernel cloneKernel(cl_kernel src);
  /** @brief P2: 让已录制的 dispatch 失效（调优/布局改变后必须重录）。 */
  void    invalidateCapture();
  /** @brief R32 原型：把整帧 dispatch 录制进 cl_khr_command_buffer（CUDA-graph 类比）。*/
  void buildCommandBuffer();
  /**
   * @brief Round 28: set up a small-op kernel — args + launch geometry.
   *
   * 供 run() 与 autotune() 共用：给定 op 节点、已 build 的 kernel 及其编译宏，
   * 统一设置参数并返回 dim/gws/lws（kernel 变体如 `_v`/`_ch`/3-D 网格的差异都在这里
   * 处理，避免两处各写一遍）。kernel 语义不变，这里只做接线。
   */
  void smallLaunch(const Node & n, cl_kernel k, const std::string & kernel,
                   const std::string & opts, cl_uint & dim, size_t * gws, size_t * lws,
                   bool & useLws);
  /** @brief Round 28: 带缓存的广播维度缓冲（`ew_binary_bcast` 用）。*/
  cl_mem bcastDims(const std::string & spec);
  /** @brief Round 28: 小算子的稳定签名（dispatch / autotune / tuningTargets 共用）。*/
  OpSignature smallSig(const Node & n) const;

  ClRuntime         rt_;
  std::string       plan_path_;
  std::string       plan_dir_;
  std::string       kernel_dir_;
  bool              profiling_{false};

  std::unordered_map<std::string, Tensor> T_;
  std::vector<cl_mem>                     owned_;
  // P0: activation buffer pool + per-tensor liveness (static plan ⇒ assigned once in
  // parse(), zero runtime cost).  Weights/input/output are kept out of the pool.
  ActPool                                act_pool_;
  std::vector<std::string>               act_names_;   // P0: activation tensors in the pool
  std::vector<cl_mem>                    alias_subs_;  // P0: sub-buffers for sliced aliases
  std::vector<char>                      node_skipped_;  // P0: per-node "alias, don't launch"
  // R51 (P1): 可能被持久化为 fsv16 的激活张量（4-D conv + 生产者可直写 + 消费者可直读）。
  // 这些张量的缓冲按补齐通道分配；也是「输出 fsv16 成本」可安全测量的集合。
  std::unordered_set<std::string>        fsv16_capable_;
  // R38: per-node resolved (family, layout) choice from the joint fixpoint. Empty kernel
  // = no override (fall back to the signature cache).
  std::vector<TuningEntry>               node_choice_;
  // R44: 隔离扫描保留的每签名候选短名单（供整网 busy 坐标下降回验）。autotune() 每次
  // 调用重建；globalRetune() 消费。值里带 OpSignature，便于精确写回 tuning_。
  struct CandidateSet
  {
    OpSignature              sig;
    std::vector<TuningEntry> cands;
  };
  std::map<std::string, CandidateSet>    cand_short_;
  // R45 P0#4: per-plan 选择覆盖（node 输出名 -> TuningEntry）。最高优先级，承载本图的
  // 位置相关全局最优；与跨模型共享的 tuning_ 分离。
  TuningCache                            plan_overrides_;
  // Round 22: cached OSV-swizzled conv3x3 weights for the OpenVINO kernel port
  // (keyed by the plan init name), plus their owning handles.
  std::unordered_map<std::string, cl_mem> ov_w_;
  std::vector<cl_mem>                     owned_ov_;
  // R25: cached blocked-conv (conv3x3_blk) weights + reordered inputs.
  std::unordered_map<std::string, cl_mem> blk_w_;
  std::unordered_map<std::string, cl_mem> blk_in_;
  std::vector<cl_mem>                     owned_blk_;
  // blocked 1x1 (conv1x1_blk) 权重重排缓存。
  std::unordered_map<std::string, cl_mem> blk1x1_w_;
  // blocked depthwise (depthwise_blk) 权重重排缓存。
  std::unordered_map<std::string, cl_mem> blk_dw_w_;
  // Cin<=4 首层 conv 的 [Cin*9][Cout] 权重重排（keyed by init name）。
  std::unordered_map<std::string, cl_mem> cin3_w_;
  std::vector<cl_mem>                     owned_cin3_;
  // R36: 同一帧内对同一张量只重排一次（多个 blocked 消费者共享 bfyx->fsv16 结果）。
  // capture 期填充；重放期 blkInput 不再被调用。每帧 run() 开头清空。
  std::unordered_set<std::string>         reordered_frame_;
  // R31: cached zero-padded depthwise inputs (keyed by input tensor name).
  std::unordered_map<std::string, cl_mem> dw_pad_;
  std::vector<cl_mem>                     owned_dwp_;
  // Round 28: cached broadcast-dim buffers for the small-op autotune/tuning path.
  std::unordered_map<std::string, cl_mem> small_buf_;
  std::vector<Node>                       nodes_;
  std::vector<std::string>                outputs_;   // 输出张量名（按声明顺序）
  std::string                             input_name_;
  std::vector<int64_t>                    input_dims_;

  cl_kernel kGemm_{nullptr}, kConvG_{nullptr}, kBin_{nullptr}, kBinB_{nullptr}, kUn_{nullptr},
    kCopy_{nullptr}, kSlice_{nullptr}, kConcat_{nullptr}, kPool_{nullptr}, kResize_{nullptr},
    kSoftmax_{nullptr}, kPerm_{nullptr}, kGap_{nullptr}, kBias_{nullptr}, kBmm_{nullptr};
  // R23: per-node specialized kernels (conv3x3/conv1x1/gemm/gap/depthwise) built
  // once and reused across run() calls — avoids clCreateKernel on every node of
  // every inference. Pure caching; numerics are unchanged.
  std::unordered_map<std::string, cl_kernel> kcache_;

  // P2: per-node dispatch cache (see cloneKernel / enqueueCmd above).
  std::vector<std::vector<PlanCmd>> node_cmds_;       // 每个节点录制好的 dispatch 列表
  std::vector<cl_kernel>            clone_kernels_;   // 克隆 kernel 的所有权（析构释放）
  std::vector<PlanCmd>              cap_cmds_;         // capture 期当前节点的临时列表
  std::unordered_map<cl_kernel, std::vector<PlanArg>> cap_args_;  // capture 期 kernel->参数
  bool                              captured_{false};  // 首帧录制完成
  bool                              capturing_{false}; // 正在录制
  bool                              launch_cache_{true};  // INFVINO_NO_LAUNCH_CACHE 可关
  size_t                            cap_node_{0};      // 正在录制的节点序号

  // R32 原型：cl_khr_command_buffer 整帧重放（INFVINO_CMDBUF=1 开启）。
  void * cmdbuf_{nullptr};                 // cl_command_buffer_khr
  void * cmdbuf_enqueue_{nullptr};         // clEnqueueCommandBufferKHR_fn
  void * cmdbuf_release_{nullptr};         // clReleaseCommandBufferKHR_fn
  bool   cmdbuf_enabled_{false};
  bool   cmdbuf_failed_{false};

  std::map<std::string, std::pair<double, int>> tprof_;   // op -> {ms, calls}
  // Phase2: 逐节点 GPU 时间（tag/ms/calls），与 tprof_ 一样跨 run 累加。
  std::vector<double>                           node_ms_;
  std::vector<int>                              node_calls_;
  std::vector<std::string>                      node_tag_;
  size_t                                        cur_node_{0};   // run() 当前节点序号
  int                                           fusions_res_{0};    // R33 残差融合次数
  int                                           fusions_concat_{0}; // R30c concat->conv1x1 次数
  int                                           fusions_scale_{0};  // R51 D5 Mul->conv1x1 prologue 次数
  int                                           fusions_sppf_{0};   // R56 D5 SPPF maxpool->concat 次数
  double                                        last_run_ms_{0.0};
  // P2: host-side segmentation (only filled when profiling_): cumulative time in
  // clEnqueueNDRangeKernel and in clWaitForEvents.
  double prof_enqueue_ms_{0.0};
  double prof_wait_ms_{0.0};

  // 自动调优缓存（docs/autotuning.md）。查不到 → 回退到 dispatch 里的内置启发式。
  TuningCache tuning_;
};

}  // namespace infvino

#endif  // INFVINO__PLAN_MODEL_HPP_
