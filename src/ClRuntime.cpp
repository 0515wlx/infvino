// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
#include "infvino/ClRuntime.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#ifndef INFVINO_KERNEL_DIR
#define INFVINO_KERNEL_DIR "kernels"
#endif

// Intel 子组查询枚举，避免依赖特定头文件版本。
#ifndef CL_DEVICE_SUB_GROUP_SIZES_INTEL
#define CL_DEVICE_SUB_GROUP_SIZES_INTEL 0x4108
#endif
// CL_DEVICE_PCI_BUS_INFO_KHR（cl_khr_pci_bus_info 扩展）；部分驱动不返回。
#ifndef CL_DEVICE_PCI_BUS_INFO_KHR
#define CL_DEVICE_PCI_BUS_INFO_KHR 0x410F
#endif
#ifndef CL_DEVICE_VENDOR_ID
#define CL_DEVICE_VENDOR_ID 0x1000
#endif

namespace infvino
{

namespace
{
// 稳定 64-bit FNV-1a（std::hash 不保证跨进程/跨版本一致，不能用于磁盘缓存键）。
uint64_t fnv1a64(const std::string & s)
{
  uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
  return h;
}

std::string hex64(uint64_t v)
{
  char b[17];
  std::snprintf(b, sizeof(b), "%016llx", static_cast<unsigned long long>(v));
  return b;
}

std::string getStr(cl_device_id d, cl_device_info what)
{
  size_t len = 0;
  clGetDeviceInfo(d, what, 0, nullptr, &len);
  std::string s(len ? len - 1 : 0, '\0');
  if (len) clGetDeviceInfo(d, what, len, &s[0], nullptr);
  return s;
}

std::string readFile(const std::string & path)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open kernel source: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::vector<cl_device_id> devicesOf(cl_platform_id p, cl_device_type type)
{
  cl_uint n = 0;
  if (clGetDeviceIDs(p, type, 0, nullptr, &n) != CL_SUCCESS || n == 0) return {};
  std::vector<cl_device_id> v(n);
  clGetDeviceIDs(p, type, n, v.data(), nullptr);
  return v;
}

ClDeviceInfo queryDevice(cl_device_id d)
{
  ClDeviceInfo i;
  i.name = getStr(d, CL_DEVICE_NAME);
  i.vendor = getStr(d, CL_DEVICE_VENDOR);
  i.driver_version = getStr(d, CL_DRIVER_VERSION);
  i.opencl_version = getStr(d, CL_DEVICE_VERSION);
  clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(i.eu), &i.eu, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(i.clock_mhz), &i.clock_mhz, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(i.global_mem_bytes), &i.global_mem_bytes, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(i.local_mem_bytes), &i.local_mem_bytes, nullptr);
  clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(i.max_work_group), &i.max_work_group, nullptr);
  {
    cl_uint align_bits = 0;
    if (clGetDeviceInfo(d, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(align_bits), &align_bits, nullptr) == CL_SUCCESS)
      i.mem_base_align = align_bits / 8;  // 字段以 bit 计
  }
  cl_device_type dtype = 0;
  if (clGetDeviceInfo(d, CL_DEVICE_TYPE, sizeof(dtype), &dtype, nullptr) == CL_SUCCESS)
    i.is_gpu = (dtype & CL_DEVICE_TYPE_GPU) != 0;
  size_t sg = 0;
  if (clGetDeviceInfo(d, CL_DEVICE_SUB_GROUP_SIZES_INTEL, sizeof(sg), &sg, nullptr) == CL_SUCCESS) {
    i.subgroup_size = sg;
  }
  cl_uint vid = 0;
  if (clGetDeviceInfo(d, CL_DEVICE_VENDOR_ID, sizeof(vid), &vid, nullptr) == CL_SUCCESS)
    i.pci_vendor_id = vid;
  // cl_khr_pci_bus_info: {cl_uint domain, bus, device, function; cl_uint vendor_id, device_id}
  struct PciBusInfo { cl_uint domain, bus, dev, func, vendor_id, device_id; } pci;
  if (clGetDeviceInfo(d, CL_DEVICE_PCI_BUS_INFO_KHR, sizeof(pci), &pci, nullptr) == CL_SUCCESS) {
    i.pci_device_id = pci.device_id;
    if (!i.pci_vendor_id) i.pci_vendor_id = pci.vendor_id;
  }
  i.peak_fp16_gflops = static_cast<double>(i.eu) * i.clock_mhz * 1e6 * 32.0 / 1e9;
  return i;
}
}  // namespace

std::string ClDeviceInfo::describe() const
{
  char buf[512];
  std::snprintf(
    buf, sizeof(buf), "%s [%s] EU=%u clk=%uMHz subgroup=%zu peak_fp16=%.1fGFLOP/s",
    name.c_str(), opencl_version.c_str(), eu, clock_mhz, subgroup_size,
    peak_fp16_gflops);
  return buf;
}

std::vector<ClDeviceInfo> ClRuntime::enumerate()
{
  std::vector<ClDeviceInfo> out;
  cl_uint np = 0;
  if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0) return out;
  std::vector<cl_platform_id> plats(np);
  clGetPlatformIDs(np, plats.data(), nullptr);

  for (cl_platform_id p : plats) {
    for (cl_device_id d : devicesOf(p, CL_DEVICE_TYPE_ALL)) {
      out.push_back(queryDevice(d));
    }
  }
  return out;
}

ClRuntime::ClRuntime(const std::string & kernel_dir, int platform, int device, bool profiling)
: kernel_dir_(kernel_dir)
{
  cl_uint np = 0;
  if (clGetPlatformIDs(0, nullptr, &np) != CL_SUCCESS || np == 0)
    throw std::runtime_error("ClRuntime: no OpenCL platforms");
  std::vector<cl_platform_id> plats(np);
  clGetPlatformIDs(np, plats.data(), nullptr);

  std::vector<cl_device_id> devs;
  if (platform >= 0) {
    devs = devicesOf(plats[static_cast<size_t>(platform)], CL_DEVICE_TYPE_ALL);
  } else {
    for (cl_platform_id p : plats) {
      devs = devicesOf(p, CL_DEVICE_TYPE_GPU);
      if (!devs.empty()) break;
    }
    if (devs.empty())
      for (cl_platform_id p : plats) {
        devs = devicesOf(p, CL_DEVICE_TYPE_ALL);
        if (!devs.empty()) break;
      }
  }
  if (devs.empty()) throw std::runtime_error("ClRuntime: no OpenCL devices");
  if (device < 0 || static_cast<size_t>(device) >= devs.size())
    throw std::runtime_error("ClRuntime: device index out of range");
  device_ = devs[static_cast<size_t>(device)];
  info_ = queryDevice(device_);

  cl_int err;
  context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("ClRuntime: clCreateContext failed");
  queue_ = clCreateCommandQueue(
    context_, device_, profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("ClRuntime: clCreateCommandQueue failed");

  // T2: on-disk program binary cache. INFVINO_PROGRAM_CACHE 覆盖目录；"none"/"" 关闭。
  {
    const char * env = std::getenv("INFVINO_PROGRAM_CACHE");
    if (env && (std::string(env) == "none" || std::string(env) == "0"))
      program_cache_dir_.clear();
    else if (env && std::string(env).size() > 0)
      program_cache_dir_ = env;
    else if (const char * home = std::getenv("HOME"))
      program_cache_dir_ = std::string(home) + "/.cache/infvino/programs";
    else
      program_cache_dir_ = "/tmp/infvino_programs";
    if (!program_cache_dir_.empty())
    {
      std::error_code ec;
      std::filesystem::create_directories(program_cache_dir_, ec);
      if (ec) program_cache_dir_.clear();   // 不可写 → 关闭
    }
  }
}

ClRuntime::~ClRuntime()
{
  for (auto & kv : programs_) {
    if (kv.second) clReleaseProgram(kv.second);
  }
  if (queue_) clReleaseCommandQueue(queue_);
  if (context_) clReleaseContext(context_);
}

std::string ClRuntime::deviceCacheKey() const
{
  // 设备键：型号 + vendor + driver + OpenCL 版本 + EU/clk。二进制不可跨设备/驱动复用。
  return info_.name + "|" + info_.vendor + "|" + info_.driver_version + "|" +
         info_.opencl_version + "|eu" + std::to_string(info_.eu) + "clk" +
         std::to_string(info_.clock_mhz);
}

cl_program ClRuntime::tryLoadBinary(const std::string & cache_key)
{
  if (program_cache_dir_.empty()) return nullptr;
  const std::string path = program_cache_dir_ + "/" + cache_key + ".bin";
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return nullptr;
  const std::streamsize n = f.tellg();
  if (n <= 0) return nullptr;
  f.seekg(0);
  std::vector<unsigned char> data(static_cast<size_t>(n));
  f.read(reinterpret_cast<char *>(data.data()), n);
  if (!f) return nullptr;

  cl_int err = CL_SUCCESS;
  const unsigned char * p = data.data();
  size_t sz = data.size();
  cl_program prog = clCreateProgramWithBinary(context_, 1, &device_, &sz, &p, nullptr, &err);
  if (err != CL_SUCCESS || !prog) return nullptr;
  // 二进制已含编译选项；再 build 一次以完成链接（options 传 nullptr）。
  err = clBuildProgram(prog, 1, &device_, nullptr, nullptr, nullptr);
  if (err != CL_SUCCESS) { clReleaseProgram(prog); return nullptr; }
  return prog;
}

void ClRuntime::storeBinary(const std::string & cache_key, cl_program prog)
{
  if (program_cache_dir_.empty() || !prog) return;
  size_t sz = 0;
  if (clGetProgramInfo(prog, CL_PROGRAM_BINARY_SIZES, sizeof(sz), &sz, nullptr) != CL_SUCCESS || !sz)
    return;
  std::vector<unsigned char> data(sz);
  unsigned char * ptr = data.data();
  if (clGetProgramInfo(prog, CL_PROGRAM_BINARIES, sizeof(ptr), &ptr, nullptr) != CL_SUCCESS)
    return;
  // 原子写：临时文件 + rename（避免并发进程读到半截文件）。
  const std::string path = program_cache_dir_ + "/" + cache_key + ".bin";
  const std::string tmp  = path + ".tmp." + std::to_string(static_cast<long>(::getpid()));
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) return;
    f.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!f) { std::remove(tmp.c_str()); return; }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) std::remove(tmp.c_str());
}

cl_program ClRuntime::buildProgramCached(
  const std::string & source, const std::string & mem_key, const std::string & disk_key,
  const std::string & options, const std::string & kernel_name)
{
  auto it = programs_.find(mem_key);
  if (it != programs_.end()) return it->second;

  cl_program prog = nullptr;
  if (!program_cache_dir_.empty())
  {
    prog = tryLoadBinary(disk_key);
    if (prog) { ++cache_hits_; programs_[mem_key] = prog; return prog; }
    ++cache_misses_;
  }

  cl_int err;
  const char * sp = source.c_str();
  prog = clCreateProgramWithSource(context_, 1, &sp, nullptr, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("clCreateProgramWithSource failed");
  err = clBuildProgram(prog, 1, &device_, options.c_str(), nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t len = 0;
    clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
    std::vector<char> log(len + 1, 0);
    clGetProgramBuildInfo(prog, device_, CL_PROGRAM_BUILD_LOG, len, log.data(), nullptr);
    clReleaseProgram(prog);
    throw std::runtime_error(
      "clBuildProgram failed for " + kernel_name + " (" + options + "):\n" + log.data());
  }
  storeBinary(disk_key, prog);
  programs_[mem_key] = prog;
  return prog;
}

cl_kernel ClRuntime::buildFromSource(
  const std::string & source, const std::string & kernel_name, const std::string & options)
{
  const std::string mem_key = source.substr(0, 64) + "|" + options + "|" + hex64(fnv1a64(source));
  const std::string disk_key = hex64(fnv1a64(deviceCacheKey() + "|inline|" + options + "|" +
                                             hex64(fnv1a64(source))));
  cl_program prog = buildProgramCached(source, mem_key, disk_key, options, kernel_name);
  cl_int err;
  cl_kernel k = clCreateKernel(prog, kernel_name.c_str(), &err);
  if (err != CL_SUCCESS) throw std::runtime_error("clCreateKernel failed: " + kernel_name);
  return k;
}

cl_kernel ClRuntime::buildKernel(
  const std::string & source_name, const std::string & kernel_name, const std::string & options)
{
  // Cache the .cl text (the previous code re-read the file from disk and re-hashed
  // the whole source on *every* buildKernel call — hundreds of disk reads + full
  // hashes per inference for a ~250-node plan, which starved the GPU queue and
  // inflated the wall/busy gap). Programs are now keyed by (source_name, options).
  auto sit = sources_.find(source_name);
  if (sit == sources_.end())
    sit = sources_.emplace(source_name, readFile(kernel_dir_ + "/" + source_name + ".cl")).first;
  const std::string mem_key  = "file|" + source_name + "|" + options;
  const std::string src_hash = hex64(fnv1a64(sit->second));
  const std::string disk_key = hex64(fnv1a64(deviceCacheKey() + "|" + source_name + "|" +
                                              options + "|" + src_hash));
  cl_program prog = buildProgramCached(sit->second, mem_key, disk_key, options, kernel_name);
  cl_int err;
  cl_kernel k = clCreateKernel(prog, kernel_name.c_str(), &err);
  if (err != CL_SUCCESS) throw std::runtime_error("clCreateKernel failed: " + kernel_name);
  return k;
}

std::string ClRuntime::sourceHash(const std::string & source_name)
{
  auto sit = sources_.find(source_name);
  if (sit == sources_.end())
    sit = sources_.emplace(source_name, readFile(kernel_dir_ + "/" + source_name + ".cl")).first;
  return hex64(fnv1a64(sit->second));
}

std::string ClRuntime::sourcesHash(const std::vector<std::string> & source_names)
{
  std::vector<std::string> ns = source_names;
  std::sort(ns.begin(), ns.end());
  ns.erase(std::unique(ns.begin(), ns.end()), ns.end());
  std::string all;
  for (const auto & n : ns)
  {
    if (n.empty()) continue;
    all += n;
    all.push_back('\0');
    all += sourceHash(n);
    all.push_back('\n');
  }
  return hex64(fnv1a64(all));
}

cl_mem ClRuntime::alloc(size_t bytes, cl_mem_flags flags)
{
  cl_int err;
  cl_mem m = clCreateBuffer(context_, flags, bytes, nullptr, &err);
  if (err != CL_SUCCESS) throw std::runtime_error("clCreateBuffer failed");
  return m;
}

void ClRuntime::write(cl_mem buf, size_t bytes, const void * host, bool blocking)
{
  cl_int err = clEnqueueWriteBuffer(queue_, buf, blocking ? CL_TRUE : CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueWriteBuffer failed");
}

void ClRuntime::read(cl_mem buf, size_t bytes, void * host, bool blocking)
{
  cl_int err = clEnqueueReadBuffer(queue_, buf, blocking ? CL_TRUE : CL_FALSE, 0, bytes, host, 0, nullptr, nullptr);
  if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueReadBuffer failed");
}

void ClRuntime::finish() const { clFinish(queue_); }

cl_event ClRuntime::enqueueND(
  cl_command_queue q, cl_kernel k, cl_uint dim, const size_t * gws, const size_t * lws)
{
  cl_event ev = nullptr;
  cl_int err = clEnqueueNDRangeKernel(q, k, dim, nullptr, gws, lws, 0, nullptr, &ev);
  if (err != CL_SUCCESS) throw std::runtime_error("clEnqueueNDRangeKernel failed: " + std::to_string(err));
  return ev;
}

double ClRuntime::timeMs(
  const std::function<cl_event()> & enqueue, int warmup, int iters, double * min_ms, double * p90_ms)
{
  for (int i = 0; i < warmup; ++i) {
    cl_event ev = enqueue();
    clWaitForEvents(1, &ev);
    clReleaseEvent(ev);
  }
  std::vector<double> ts;
  ts.reserve(static_cast<size_t>(iters));
  for (int i = 0; i < iters; ++i) {
    cl_event ev = enqueue();
    clWaitForEvents(1, &ev);
    cl_ulong t0 = 0, t1 = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
    ts.push_back(static_cast<double>(t1 - t0) * 1e-6);
    clReleaseEvent(ev);
  }
  std::sort(ts.begin(), ts.end());
  if (min_ms) *min_ms = ts.front();
  if (p90_ms) *p90_ms = ts[static_cast<size_t>(0.9 * (ts.size() - 1))];
  return ts[ts.size() / 2];
}

}  // namespace infvino
