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

  const ClRuntime &   runtime() const { return rt_; }
  const DeviceInfo &  device() const { return rt_.info(); }

  /** @brief 最近一次 run() 的墙钟耗时（ms）。 */
  double lastRunMs() const { return last_run_ms_; }
  /** @brief 算子耗时表（仅 profiling=true 时填充）：op -> {ms, calls}。 */
  const std::map<std::string, std::pair<double, int>> & opProfile() const { return tprof_; }

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
  void    buildKernels();
  void    releaseKernels();
  void    dispatch(const Node & n);
  Tensor & ref(const std::string & name);
  int     attrInt(const Node & n, const char * key, int def) const;

  ClRuntime         rt_;
  std::string       plan_path_;
  std::string       plan_dir_;
  std::string       kernel_dir_;
  bool              profiling_{false};

  std::unordered_map<std::string, Tensor> T_;
  std::vector<cl_mem>                     owned_;
  std::vector<Node>                       nodes_;
  std::vector<std::string>                outputs_;   // 输出张量名（按声明顺序）
  std::string                             input_name_;
  std::vector<int64_t>                    input_dims_;

  cl_kernel kGemm_{nullptr}, kConvG_{nullptr}, kBin_{nullptr}, kBinB_{nullptr}, kUn_{nullptr},
    kCopy_{nullptr}, kSlice_{nullptr}, kConcat_{nullptr}, kPool_{nullptr}, kResize_{nullptr},
    kSoftmax_{nullptr}, kPerm_{nullptr}, kGap_{nullptr}, kBias_{nullptr}, kBmm_{nullptr};

  std::map<std::string, std::pair<double, int>> tprof_;   // op -> {ms, calls}
  double                                        last_run_ms_{0.0};
};

}  // namespace gk

#endif  // INFVINO__PLAN_MODEL_HPP_
