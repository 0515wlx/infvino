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
#include <utility>
#include <vector>

#include "infvino/ClRuntime.hpp"
#include "infvino/Tuning.hpp"

namespace gk
{

/**
 * @brief 执行一份算子计划的整网模型（单线程使用；内部持有上下文/队列）。
 *
 * 典型用法：
 * @code
 *   gk::PlanModel model("models/yolov8n-pose.plan");
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
  const DeviceInfo &  device() const { return rt_.info(); }

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
  /** @brief 把 current tuning_ 写回文件。 */
  bool saveTuning(const std::string & path) const { return tuning_.save(path); }

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
  /** @brief 清空算子耗时表与 host 分段计时（在统计循环前调用）。 */
  void clearProfile() { tprof_.clear(); prof_enqueue_ms_ = prof_wait_ms_ = 0.0; }

private:
  struct Tensor
  {
    std::vector<int64_t> dims;
    cl_mem               mem{nullptr};
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
  /** @brief R30c: fuse `concat4 -> conv1x1` into a CAT4 gemm (skip concat materialisation). */
  void    fuseConcatConv1x1();
  void    buildKernels();
  void    releaseKernels();
  void    dispatch(const Node & n);
  Tensor & ref(const std::string & name);
  int     attrInt(const Node & n, const char * key, int def) const;
  cl_mem  ovWeight(const std::string & name, Tensor & w, int Cout, int Cin);
  /** @brief R25: repack weights to OpenVINO os_is_yx_isv16_osv16 for conv3x3_blk. */
  cl_mem  blkWeight(const std::string & name, Tensor & w, int Cout, int Cin);
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
  // Round 22: cached OSV-swizzled conv3x3 weights for the OpenVINO kernel port
  // (keyed by the plan init name), plus their owning handles.
  std::unordered_map<std::string, cl_mem> ov_w_;
  std::vector<cl_mem>                     owned_ov_;
  // R25: cached blocked-conv (conv3x3_blk) weights + reordered inputs.
  std::unordered_map<std::string, cl_mem> blk_w_;
  std::unordered_map<std::string, cl_mem> blk_in_;
  std::vector<cl_mem>                     owned_blk_;
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

  std::map<std::string, std::pair<double, int>> tprof_;   // op -> {ms, calls}
  double                                        last_run_ms_{0.0};
  // P2: host-side segmentation (only filled when profiling_): cumulative time in
  // clEnqueueNDRangeKernel and in clWaitForEvents.
  double prof_enqueue_ms_{0.0};
  double prof_wait_ms_{0.0};

  // 自动调优缓存（docs/autotuning.md）。查不到 → 回退到 dispatch 里的内置启发式。
  TuningCache tuning_;
};

}  // namespace gk

#endif  // INFVINO__PLAN_MODEL_HPP_
