// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 轻量 OpenCL 运行时封装：设备枚举 / 上下文 / 队列 / kernel 构建缓存 / 计时。
// 仅用于自研 kernel 的开发与基准，不参与生产推理。
#ifndef INFVINO_GK__CL_RUNTIME_HPP_
#define INFVINO_GK__CL_RUNTIME_HPP_

#include <CL/cl.h>

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// .cl 源码目录。正常由 CMake 注入（源码树目录）；独立编译时退回工作目录下 kernels/。
#ifndef INFVINO_KERNEL_DIR
#define INFVINO_KERNEL_DIR "kernels"
#endif

namespace gk
{

struct DeviceInfo
{
  std::string     name;
  std::string     vendor;
  std::string     driver_version;
  std::string     opencl_version;
  cl_uint         eu{0};                // compute units (EU)
  cl_uint         clock_mhz{0};
  cl_ulong        global_mem_bytes{0};
  cl_ulong        local_mem_bytes{0};
  size_t          max_work_group{0};
  size_t          subgroup_size{0};
  bool            is_gpu{false};        // device is CL_DEVICE_TYPE_GPU
  double          peak_fp16_gflops{0};  // eu * clock(MHz) * 16 ops

  std::string     describe() const;
};

class ClRuntime
{
public:
  /** @brief 枚举平台/设备（不建立上下文）。 */
  static std::vector<DeviceInfo> enumerate();

  /**
   * @param kernel_dir  运行时加载 .cl 的目录（默认编译期写死的源码 kernels/）.
   * @param platform    -1 = 自动选择含 GPU 的平台.
   * @param device      平台内设备序号（按 CL_DEVICE_TYPE_GPU 过滤后）。
   * @param profiling   是否开启 CL_QUEUE_PROFILING_ENABLE（基准需要）。
   */
  explicit ClRuntime(
    const std::string & kernel_dir = INFVINO_KERNEL_DIR,
    int platform = -1, int device = 0, bool profiling = true);
  ~ClRuntime();

  ClRuntime(const ClRuntime &) = delete;
  ClRuntime & operator=(const ClRuntime &) = delete;

  const DeviceInfo & info() const { return info_; }
  cl_device_id       device() const { return device_; }
  cl_context         context() const { return context_; }
  cl_command_queue   queue() const { return queue_; }
  const std::string & kernelDir() const { return kernel_dir_; }

  /**
   * @brief 从 kernels/<name>.cl 构建（并按 options 缓存）一个 kernel。
   * @throws std::runtime_error 构建失败时，消息包含 OpenCL build log。
   */
  cl_kernel buildKernel(
    const std::string & source_name, const std::string & kernel_name,
    const std::string & options = "");

  /** @brief 原始源码构建（用于内联测试）。 */
  cl_kernel buildFromSource(
    const std::string & source, const std::string & kernel_name,
    const std::string & options = "");

  cl_mem alloc(size_t bytes, cl_mem_flags flags = CL_MEM_READ_WRITE);
  void   write(cl_mem buf, size_t bytes, const void * host, bool blocking = true);
  void   read(cl_mem buf, size_t bytes, void * host, bool blocking = true);
  void   finish() const;

  /** @brief 计时：enqueue() 内部负责入队并返回其 event；返回中位耗时(ms)。 */
  double timeMs(
    const std::function<cl_event()> & enqueue, int warmup, int iters,
    double * min_ms = nullptr, double * p90_ms = nullptr);

  /** @brief 计算 ops/EU/cycle（相对理论极限 16 的百分比）。 */
  double opsPerEuCycle(double flops, double milliseconds) const
  {
    const double sec = milliseconds * 1e-3;
    return flops / (static_cast<double>(info_.eu) * info_.clock_mhz * 1e6 * sec);
  }

  static cl_event enqueueND(
    cl_command_queue q, cl_kernel k, cl_uint dim, const size_t * gws, const size_t * lws);

private:
  std::string      kernel_dir_;
  DeviceInfo       info_;
  cl_device_id     device_{nullptr};
  cl_context       context_{nullptr};
  cl_command_queue queue_{nullptr};
  std::unordered_map<std::string, cl_program> programs_;  // key: source|options
};

}  // namespace gk

#endif  // INFVINO_GK__CL_RUNTIME_HPP_
