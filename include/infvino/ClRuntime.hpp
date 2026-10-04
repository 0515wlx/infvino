// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 轻量 OpenCL 运行时封装：设备枚举 / 上下文 / 队列 / kernel 构建缓存 / 计时。
// 仅用于自研 kernel 的开发与基准，不参与生产推理。
#ifndef INFVINO__CL_RUNTIME_HPP_
#define INFVINO__CL_RUNTIME_HPP_

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

namespace infvino
{

/**
 * @brief 激活缓冲池（P0）：把「每个中间张量一块常驻 cl_mem」改成「按生存期复用」。
 *
 * 与 OpenVINO `memory_pool` 同思路：先按尺寸找一块够大的空闲 buffer，若它的占用集合
 * 与调用方给的「冲突集」（生存期重叠的张量 id）不相交则复用，否则新建。所有 buffer
 * 由池持有并在析构时释放；`owned` 是新建计数，用于度量复用率。
 *
 * 全 fp16、行主序，因此无需关心元素类型/对齐差异。
 *
 * 用法（静态计划，分配在 parse 期一次完成，run 期零开销）：
 *   ActPool pool;
 *   cl_mem m = pool.acquire(bytes, conflict_ids);  // conflict_ids = 已占用该 buffer 的组
 */
class ActPool
{
public:
  ActPool() = default;
  ~ActPool()
  {
    for (auto & b : bufs_)
      if (b.mem) clReleaseMemObject(b.mem);
  }
  ActPool(const ActPool &) = delete;
  ActPool & operator=(const ActPool &) = delete;

  /**
   * @param alloc       底层分配器（clCreateBuffer）。
   * @param bytes       需要的字节数。
   * @param conflict    本块一旦复用后会与哪些「张量 id」重叠；用于判断能否共享。
   * @param id          当前张量的 id（用于把它记入被复用 buffer 的占用集合）。
   */
  cl_mem acquire(const std::function<cl_mem(size_t)> & alloc, size_t bytes,
                 const std::vector<int> & conflict, int id)
  {
    // 找一块「够大、且占用集合与 conflict 不相交」的现成 buffer。
    Buffer * best = nullptr;
    for (auto & b : bufs_)
    {
      if (b.mem == nullptr || b.bytes < bytes) continue;
      if (intersects(b.users, conflict)) continue;
      if (best == nullptr || b.bytes < best->bytes) best = &b;  // 最省（尺寸最小的够用块）
    }
    if (best == nullptr)
    {
      Buffer nb;
      nb.bytes = bytes;
      nb.mem   = alloc(bytes);
      bufs_.push_back(nb);
      best = &bufs_.back();
      newlyAllocatedBytes_ += bytes;
      ++allocCount_;
    }
    best->users.push_back(id);
    std::sort(best->users.begin(), best->users.end());
    requestedBytes_ += bytes;
    return best->mem;
  }

  size_t requestedBytes() const { return requestedBytes_; }
  size_t allocatedBytes() const { return newlyAllocatedBytes_; }
  size_t bufferCount() const { return bufs_.size(); }
  size_t allocCount() const { return allocCount_; }

  /** @brief 调试：每个 buffer 共享了哪些张量 id。 */
  std::vector<std::string> debugSharing() const
  {
    std::vector<std::string> out;
    for (size_t i = 0; i < bufs_.size(); ++i)
    {
      std::string s = "buf" + std::to_string(i) + " bytes=" + std::to_string(bufs_[i].bytes) +
                      " users=";
      for (int u : bufs_[i].users) s += std::to_string(u) + ",";
      out.push_back(s);
    }
    return out;
  }

private:
  struct Buffer
  {
    cl_mem           mem{nullptr};
    size_t           bytes{0};
    std::vector<int> users;  // 共享过此 buffer 的张量 id（排序）
  };

  static bool intersects(const std::vector<int> & a, const std::vector<int> & b)
  {
    size_t i = 0, j = 0;
    while (i < a.size() && j < b.size())
    {
      if (a[i] == b[j]) return true;
      if (a[i] < b[j]) ++i; else ++j;
    }
    return false;
  }

  std::vector<Buffer> bufs_;
  size_t              requestedBytes_{0};
  size_t              newlyAllocatedBytes_{0};
  size_t              allocCount_{0};
};

struct ClDeviceInfo
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
  // PCI 标识（优先 CL_DEVICE_PCI_BUS_INFO_KHR 的 device id，退回 vendor id + name）。
  // 用于自动调优缓存的设备键：不同型号/不同 EU 数的 GPU 必须分开缓存。
  uint32_t        pci_device_id{0};     // 0 = 未知
  uint32_t        pci_vendor_id{0};     // 0 = 未知
  double          peak_fp16_gflops{0};  // eu * clock(MHz) * 32 FLOP (16 packed FP16 FMA)

  std::string     describe() const;
};

class ClRuntime
{
public:
  /** @brief 枚举平台/设备（不建立上下文）。 */
  static std::vector<ClDeviceInfo> enumerate();

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

  const ClDeviceInfo & info() const { return info_; }
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

  /** @brief 计算 ops/EU/cycle（相对理论极限 32 的百分比；FP16 = 16 packed FMA/EU/cyc）。 */
  double opsPerEuCycle(double flops, double milliseconds) const
  {
    const double sec = milliseconds * 1e-3;
    return flops / (static_cast<double>(info_.eu) * info_.clock_mhz * 1e6 * sec);
  }

  static cl_event enqueueND(
    cl_command_queue q, cl_kernel k, cl_uint dim, const size_t * gws, const size_t * lws);

private:
  std::string      kernel_dir_;
  ClDeviceInfo       info_;
  cl_device_id     device_{nullptr};
  cl_context       context_{nullptr};
  cl_command_queue queue_{nullptr};
  std::unordered_map<std::string, cl_program> programs_;  // key: source|options
  std::unordered_map<std::string, std::string> sources_;  // source_name -> .cl text
};

}  // namespace infvino

#endif  // INFVINO__CL_RUNTIME_HPP_
