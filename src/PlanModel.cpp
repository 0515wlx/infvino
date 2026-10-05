// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/PlanModel.hpp"

#include <CL/cl_ext.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "infvino/Half.hpp"
#include "infvino/Tiles.hpp"
#include "infvino/Autotuner.hpp"
#include "infvino/KernelFamily.hpp"
#include "infvino/LayoutSolver.hpp"

namespace infvino
{

namespace
{
std::vector<std::string> splitCsv(const std::string & s)
{
  std::vector<std::string> out;
  std::stringstream        ss(s);
  std::string              tok;
  while (std::getline(ss, tok, ',')) out.push_back(tok);
  return out;
}

std::string readText(const std::string & path)
{
  std::ifstream f(path);
  if (!f) throw std::runtime_error("PlanModel: cannot open plan: " + path);
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

std::string dirName(const std::string & path)
{
  const auto pos = path.find_last_of('/');
  return (pos == std::string::npos) ? std::string(".") : path.substr(0, pos);
}

// R51 (R48 §4bis P1): 激活张量的**分配元素数**。fsv16 (`b_fs_yx_fsv16` =
// [C/16][H][W][16]) 会把通道数**补齐**到 16 的倍数，因此当 C%16!=0 时，若生产者
// 直写 fsv16 就需要 ceil(C/16)*16*H*W 个元素，否则越界。此前布局规划把所有
// C%16!=0 的张量钉死 NCHW（避免越界）——本函数让池按补齐尺寸分配，从而解锁
// mb 的 88/120/144 等非对齐张量进入 blocked 链。
//
// **只对 4-D NCHW conv 张量**（dims = [N,C,H,W]）做补齐：3-D reshape/decode 张量的
// `dims[size-3]` 是 batch 维（=1），若按其补齐会 16× 膨胀（y11 实测 requested
// 69→140MB、busy +1.2%）。非 4-D 张量也从不被布局规划标记为 fsv16（其生产者不是
// conv 族）。NCHW 消费者只读前 `numel` 个元素，多出的补齐区不被触碰，数值逐位一致。
size_t paddedChannelNumel(const std::vector<int64_t> & d)
{
  int64_t n = 1;
  for (int64_t v : d) n *= (v < 0 ? 0 : v);
  if (d.size() == 4)
  {
    const int64_t c = d[1];
    if (c > 0 && c % 16 != 0)
      n = n / c * ((c + 15) / 16 * 16);
  }
  return static_cast<size_t>(n);
}
}  // namespace

namespace { std::string sourceOfKernel(const std::string & kernel); }  // fwd (defined below)

PlanModel::PlanModel(
  const std::string & plan_path, const std::string & kernel_dir, int platform, int device_index,
  bool profiling)
: rt_(kernel_dir, platform, device_index, profiling),
  plan_path_(plan_path),
  plan_dir_(dirName(plan_path)),
  profiling_(profiling)
{
  // P2: 每节点 dispatch 缓存（默认开）；INFVINO_NO_LAUNCH_CACHE=1 强制走原逐帧路径
  // （用于 A/B 与数值回归排查）。
  {
    const char * nc = std::getenv("INFVINO_NO_LAUNCH_CACHE");
    if (nc && std::string(nc) != "0" && std::string(nc) != "") launch_cache_ = false;
    const char * cb = std::getenv("INFVINO_CMDBUF");
    if (cb && std::string(cb) != "0" && std::string(cb) != "") cmdbuf_enabled_ = true;
  }
  parse();
  buildKernels();
  tuning_ = TuningCache::loadDefault();
  tuning_.setDeviceId(TuningCache::deviceKey(rt_.info()));

  // R45 P0#4: per-plan 选择覆盖（位置相关的全局最优）。默认 <plan>.tuning.json；
  // INFVINO_PLAN_TUNING 可覆盖路径；INFVINO_TUNING=off 时随 tuning_ 一并禁用。
  {
    const char * pt = std::getenv("INFVINO_PLAN_TUNING");
    std::string path = pt ? std::string(pt) : (plan_path_ + ".tuning.json");
    if (path == "none") plan_overrides_.setEnabled(false);
    else {
      plan_overrides_ = TuningCache::load(path);
      plan_overrides_.setDeviceId(tuning_.deviceId());
      if (!tuning_.enabled()) plan_overrides_.setEnabled(false);
      if (std::getenv("INFVINO_PLAN_TUNING_REPORT"))
        std::fprintf(stderr, "[plan-tuning] %s: %zu override(s), device=%s%s\n", path.c_str(),
                     plan_overrides_.size(), plan_overrides_.deviceId().c_str(),
                     plan_overrides_.enabled() ? "" : " (disabled)");
    }
  }

  // R45 P1#7: 数值/编译契约守卫。缓存记录生成时的 kernel 源组合指纹；当前源码与之不符时，
  // 旧 options / 旧 cache_abi 可能语义失配（R39 的 SLM 数值 bug 就是「缓存不知道 kernel 已改」
  // 这类）。默认告警；INFVINO_TUNING_STRICT=1 时整份作废（回退启发式，强制 retune）。
  {
    std::vector<std::string> srcs;
    auto collectSrc = [&](const TuningCache & c) {
      for (const auto & kv : c.entries())
        if (!kv.second.kernel.empty()) srcs.push_back(sourceOfKernel(kv.second.kernel));
    };
    collectSrc(tuning_);
    collectSrc(plan_overrides_);
    if (!srcs.empty())
    {
      const std::string h = rt_.sourcesHash(srcs);
      const bool strict = std::getenv("INFVINO_TUNING_STRICT") != nullptr;
      auto guard = [&](TuningCache & c, const char * what) {
        if (c.size() == 0) return;
        const std::string old = c.sourceHash();
        if (!old.empty() && old != h)
        {
          std::fprintf(stderr,
                       "[tuning] %s kernel_src_hash changed (%s -> %s): %s\n", what, old.c_str(),
                       h.c_str(), strict ? "invalidating (INFVINO_TUNING_STRICT)"
                                         : "options may be stale; retune recommended");
          if (strict) c.setEnabled(false);
        }
        c.setSourceHash(h);  // 保存时写入当前指纹
      };
      guard(tuning_, "config");
      guard(plan_overrides_, "plan");
    }
  }

  // R36/R38: 布局与 (族,布局) 选择必须在 tuning 载入之后、首次 run() 之前（P2 dispatch
  // 缓存会录制参数）。resolveLayoutChoices() 在缓存含 `#blk/#non` 时跑联合不动点，
  // 否则退化为一次 planBlockedLayout()。
  resolveLayoutChoices();

  // P3: 在线调优（opt-in）。默认关闭——开发板上跑长 GPU 任务有风险，部署端若要
  // 自适应再显式打开 `INFVINO_TUNING=online`（需要 profiling=true 才能计时）。
  // 只调优「缓存未命中」的签名，且用 INFVINO_ONLINE_BUDGET 限制数量。
  {
    const char * online = std::getenv("INFVINO_TUNING");
    if (online && std::string(online) == "online" && profiling_) {
      const char * be = std::getenv("INFVINO_ONLINE_BUDGET");
      const char * ie = std::getenv("INFVINO_ONLINE_ITERS");
      const int budget = be ? std::atoi(be) : 4;
      const int iters  = ie ? std::atoi(ie) : 5;
      const char * oe = std::getenv("INFVINO_ONLINE_OPS");
      std::vector<std::string> oops =
        oe ? splitCsv(oe) : std::vector<std::string>{"conv3x3", "conv1x1", "depthwise"};
      const int n = onlineTuneMissing(budget, iters, oops);
      if (n > 0) std::fprintf(stderr, "[online-tune] tuned %d signature(s)\n", n);
    }
  }
}

PlanModel::~PlanModel()
{
  if (cmdbuf_ && cmdbuf_release_)
    reinterpret_cast<clReleaseCommandBufferKHR_fn>(cmdbuf_release_)(static_cast<cl_command_buffer_khr>(cmdbuf_));
  for (cl_kernel k : clone_kernels_)
    if (k) clReleaseKernel(k);
  releaseKernels();
  for (cl_mem m : owned_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_ov_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_blk_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_cin3_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_dwp_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : alias_subs_)
    if (m) clReleaseMemObject(m);
}

cl_mem PlanModel::ovWeight(const std::string & name, Tensor & w, int Cout, int Cin)
{
  auto it = ov_w_.find(name);
  if (it != ov_w_.end()) return it->second;
  const int fmg = (Cout + 31) / 32;
  std::vector<uint16_t> host(static_cast<size_t>(w.numel()));
  rt_.read(w.mem, host.size() * 2, host.data());
  std::vector<uint16_t> sw(static_cast<size_t>(fmg) * Cin * 9 * 32, 0);
  for (int g = 0; g < fmg; ++g)
    for (int ci = 0; ci < Cin; ++ci)
      for (int kk = 0; kk < 9; ++kk) {
        uint16_t * dst = &sw[((static_cast<size_t>(g) * Cin + ci) * 9 + kk) * 32];
        for (int p = 0; p < 32; ++p) {
          const int oc = g * 32 + p;
          dst[p] = (oc < Cout) ? host[(static_cast<size_t>(oc) * Cin + ci) * 9 + kk] : (uint16_t)0;
        }
      }
  cl_mem m = rt_.alloc(sw.size() * 2, CL_MEM_READ_ONLY);
  rt_.write(m, sw.size() * 2, sw.data());
  ov_w_[name] = m;
  owned_ov_.push_back(m);
  return m;
}

cl_mem PlanModel::blkWeight(const std::string & name, Tensor & w, int Cout, int Cin)
{
  auto it = blk_w_.find(name);
  if (it != blk_w_.end()) return it->second;
  const int icb = (Cin + 15) / 16, ocb = (Cout + 15) / 16;
  std::vector<uint16_t> host(static_cast<size_t>(w.numel()));
  rt_.read(w.mem, host.size() * 2, host.data());
  // os_is_yx_isv16_osv16 = [OC/16][IC/16][3][3][isv16][osv16]
  std::vector<uint16_t> sw(static_cast<size_t>(ocb) * icb * 9 * 16 * 16, 0);
  for (int o = 0; o < Cout; ++o)
    for (int i = 0; i < Cin; ++i)
      for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw)
          sw[((((static_cast<size_t>(o / 16) * icb + (i / 16)) * 9) + kh * 3 + kw) * 16 + (i % 16)) * 16 + (o % 16)] =
              host[(static_cast<size_t>(o) * Cin + i) * 9 + kh * 3 + kw];
  cl_mem m = rt_.alloc(sw.size() * 2, CL_MEM_READ_ONLY);
  rt_.write(m, sw.size() * 2, sw.data());
  blk_w_[name] = m;
  owned_blk_.push_back(m);
  return m;
}

cl_mem PlanModel::blk1x1Weight(const std::string & name, Tensor & w, int Cout, int Cin)
{
  auto it = blk1x1_w_.find(name);
  if (it != blk1x1_w_.end()) return it->second;
  const int icb = (Cin + 15) / 16, ocb = (Cout + 15) / 16;
  std::vector<uint16_t> host(static_cast<size_t>(w.numel()));
  rt_.read(w.mem, host.size() * 2, host.data());
  // os_is_yx_isv16_osv16 (1x1): [Cout/16][Cin/16][isv16][osv16]
  std::vector<uint16_t> sw(static_cast<size_t>(ocb) * icb * 16 * 16, 0);
  for (int oc = 0; oc < Cout; ++oc)
    for (int ic = 0; ic < Cin; ++ic)
      sw[(((static_cast<size_t>(oc / 16) * icb + (ic / 16)) * 16) + (ic % 16)) * 16 + (oc % 16)] =
          host[static_cast<size_t>(oc) * Cin + ic];
  cl_mem m = rt_.alloc(sw.size() * 2, CL_MEM_READ_ONLY);
  rt_.write(m, sw.size() * 2, sw.data());
  blk1x1_w_[name] = m;
  owned_blk_.push_back(m);
  return m;
}

cl_mem PlanModel::blkDwWeight(const std::string & name, Tensor & w, int C, int K)
{
  auto it = blk_dw_w_.find(name);
  if (it != blk_dw_w_.end()) return it->second;
  const int cb = (C + 15) / 16;
  std::vector<uint16_t> host(static_cast<size_t>(w.numel()));
  rt_.read(w.mem, host.size() * 2, host.data());
  // [C/16][K][K][16]
  std::vector<uint16_t> sw(static_cast<size_t>(cb) * K * K * 16, 0);
  for (int c = 0; c < C; ++c)
    for (int kk = 0; kk < K * K; ++kk)
      sw[((static_cast<size_t>(c / 16) * K * K) + kk) * 16 + (c % 16)] =
          host[static_cast<size_t>(c) * K * K + kk];
  cl_mem m = rt_.alloc(sw.size() * 2, CL_MEM_READ_ONLY);
  rt_.write(m, sw.size() * 2, sw.data());
  blk_dw_w_[name] = m;
  owned_blk_.push_back(m);
  return m;
}

cl_mem PlanModel::cin3Weight(const std::string & name, Tensor & w, int Cout, int Cin)
{
  auto it = cin3_w_.find(name);
  if (it != cin3_w_.end()) return it->second;
  const int KHW = Cin * 9;
  std::vector<uint16_t> host(static_cast<size_t>(w.numel()));
  rt_.read(w.mem, host.size() * 2, host.data());
  std::vector<uint16_t> sw(static_cast<size_t>(KHW) * Cout);
  for (int c = 0; c < Cout; ++c)
    for (int k = 0; k < KHW; ++k)
      sw[static_cast<size_t>(k) * Cout + c] = host[static_cast<size_t>(c) * KHW + k];
  cl_mem m = rt_.alloc(sw.size() * 2, CL_MEM_READ_ONLY);
  rt_.write(m, sw.size() * 2, sw.data());
  cin3_w_[name] = m;
  owned_cin3_.push_back(m);
  return m;
}

cl_mem PlanModel::blkInput(const std::string & name, Tensor & x, int Cin, int H, int W)
{
  // R36 (P1-layout)：输入本身已是 fsv16（同一 blocked 链的生产者直接写的就是阻塞布局）
  // → 零 reorder、零额外 buffer，直接返回。
  if (x.fsv16) return x.mem;

  // 注意：**每次调用都要重跑 reorder**——输入张量在每次 run() 里会被重新计算/写入，
  // 之前「按名字缓存 reordered buffer 且只重排一次」会让第 2 次及以后的推理用上一帧
  // 的数据（跨推理陈旧 bug；见 2026-10 修复）。这里只缓存**设备 buffer**，重排每 run 一次。
  auto it = blk_in_.find(name);
  cl_mem m;
  if (it != blk_in_.end()) { m = it->second; }
  else
  {
    const size_t bytes = static_cast<size_t>((Cin + 15) / 16) * H * W * 16 * 2;
    m = rt_.alloc(bytes, CL_MEM_READ_WRITE);
    blk_in_[name] = m;
    owned_blk_.push_back(m);
  }
  if (std::getenv("INFVINO_DEBUG_REORDER"))
    std::fprintf(stderr, "[reorder] %-40s Cin=%-4d %dx%d  %.1f KB\n", name.c_str(), Cin, H, W,
                 static_cast<double>(Cin + 15) / 16 * H * W * 16 * 2 / 1024.0);
  // R36：同一帧内同一张量若已被重排过（多个 blocked 消费者共享），直接复用，跳过重复
  // launch。capture 期生效；重放期 blkInput 不再被调用。
  if (!std::getenv("INFVINO_NO_REORDER_DEDUP") && reordered_frame_.count(name)) return m;
  cl_kernel k = getKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
  setArg(k, 0, sizeof(x.mem), &x.mem);
  setArg(k, 1, sizeof(m), &m);
  setArg(k, 2, sizeof(Cin), &Cin);
  setArg(k, 3, sizeof(H), &H);
  setArg(k, 4, sizeof(W), &W);
  const size_t gws[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                         static_cast<size_t>(Cin)};
  cl_event ev = enqueueCmd(k, 3, gws, nullptr, "reorder(blk)", false);
  if (profiling_ && ev)
  {
    clWaitForEvents(1, &ev);
    cl_ulong s = 0, e = 0;
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
    tprof_["reorder(blk)"].first += static_cast<double>(e - s) * 1e-6;
    tprof_["reorder(blk)"].second += 1;
    noteNode(cur_node_, "reorder(blk)", static_cast<double>(e - s) * 1e-6);
    clReleaseEvent(ev);
  }
  reordered_frame_.insert(name);
  return m;
}

cl_mem PlanModel::dwPadInput(const std::string & name, int Cin, int H, int W, int K, int S,
                             int P, int * HpOut, int * WpadOut)
{
  // 用最大的候选 DW_TW（=8）定尺寸，这样任何调优候选的 strip 读都不会越出 buffer；
  // 边界在首次分配时一次性写零，之后每帧只由 depthwise_pad 重写 interior。
  const int    tw     = 8;
  const int    Wout   = (W + 2 * P - K) / S + 1;
  const int    Hp     = H + 2 * P;
  const int    groups = (Wout + tw - 1) / tw;
  const int    strlen = (tw - 1) * S + K;
  int          wpad   = std::max((Wout - 1) * S + K, (groups - 1) * tw * S + strlen);
  if (wpad < P + W) wpad = P + W;  // 不小于数据行本身
  if (HpOut) *HpOut = Hp;
  if (WpadOut) *WpadOut = wpad;

  auto it = dw_pad_.find(name);
  if (it != dw_pad_.end()) return it->second;
  const size_t      bytes = static_cast<size_t>(Cin) * Hp * wpad * 2;
  cl_mem            m     = rt_.alloc(bytes, CL_MEM_READ_WRITE);
  std::vector<char> zeros(bytes, 0);
  rt_.write(m, bytes, zeros.data());  // 一次性零边界（之后只写 interior）
  dw_pad_[name] = m;
  owned_dwp_.push_back(m);
  return m;
}

size_t PlanModel::inputNumel() const
{
  int64_t n = 1;
  for (auto d : input_dims_) n *= d;
  return static_cast<size_t>(n);
}

const std::vector<int64_t> & PlanModel::outputDims(size_t i) const
{
  return T_.at(outputs_.at(i)).dims;
}

size_t PlanModel::outputNumel(size_t i) const
{
  return static_cast<size_t>(T_.at(outputs_.at(i)).numel());
}

void PlanModel::setInput(const void * fp16_host)
{
  if (!input_name_.empty() && inputNumel() > 0)
    rt_.write(T_.at(input_name_).mem, inputNumel() * 2, fp16_host);
}

PlanModel::Tensor & PlanModel::ref(const std::string & name)
{
  auto it = T_.find(name);
  if (it == T_.end()) throw std::runtime_error("PlanModel: missing tensor: " + name);
  return it->second;
}

int PlanModel::attrInt(const Node & n, const char * key, int def) const
{
  auto it = n.attr.find(key);
  return it == n.attr.end() ? def : std::atoi(it->second.c_str());
}

void PlanModel::parse()
{
  const auto lines_owner = readText(plan_path_);
  std::istringstream iss(lines_owner);
  std::string        line;

  auto alloc = [&](const std::string & name, const std::vector<int64_t> & d,
                   bool padChannels = false) -> Tensor & {
    Tensor t;
    t.dims = d;
    // R51: 激活张量按补齐通道分配，使得「生产者直写 fsv16」对 C%16!=0 也安全
    // （见 paddedChannelNumel）。input/init 非 fsv16，保持精确尺寸。
    const size_t elems = padChannels ? paddedChannelNumel(d) : static_cast<size_t>(t.numel());
    t.mem  = rt_.alloc(elems * 2, CL_MEM_READ_WRITE);
    t.base = t.mem;
    t.base_off = 0;
    owned_.push_back(t.mem);
    return T_[name] = t;
  };

  act_names_.clear();
  while (std::getline(iss, line))
  {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ls(line);
    std::string        kind;
    ls >> kind;

    if (kind == "input")
    {
      std::string          name;
      std::vector<int64_t> d;
      ls >> name;
      int64_t v;
      while (ls >> v) d.push_back(v);
      alloc(name, d);
      if (input_name_.empty()) { input_name_ = name; input_dims_ = d; }
    }
    else if (kind == "init")
    {
      std::string          name, file;
      std::vector<int64_t> d;
      ls >> name >> file;
      int64_t v;
      while (ls >> v) d.push_back(v);
      Tensor & t = alloc(name, d);
      std::vector<uint16_t> buf(static_cast<size_t>(t.numel()));
      const std::string     path =
        (file.empty() || file[0] == '/') ? file : (plan_dir_ + "/" + file);
      std::ifstream f(path, std::ios::binary);
      if (!f) throw std::runtime_error("PlanModel: cannot open init " + path);
      f.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
      rt_.write(t.mem, buf.size() * 2, buf.data());
    }
    else if (kind == "tensor")
    {
      std::string          name;
      std::vector<int64_t> d;
      ls >> name;
      int64_t v;
      while (ls >> v) d.push_back(v);
      alloc(name, d, /*padChannels=*/true);
      act_names_.push_back(name);
    }
    else if (kind == "node")
    {
      Node n;
      ls >> n.op;
      std::string ins, outs;
      ls >> ins >> outs;
      n.ins = splitCsv(ins);
      n.outs = splitCsv(outs);
      std::string kv;
      while (ls >> kv)
      {
        auto p = kv.find('=');
        if (p != std::string::npos) n.attr[kv.substr(0, p)] = kv.substr(p + 1);
      }
      nodes_.push_back(std::move(n));
    }
    else if (kind == "output")
    {
      std::string name;
      ls >> name;
      if (!name.empty()) outputs_.push_back(name);
    }
  }

  if (input_name_.empty()) throw std::runtime_error("PlanModel: plan has no 'input' line");
  if (outputs_.empty()) throw std::runtime_error("PlanModel: plan has no 'output' line");
  for (const auto & o : outputs_)
    if (!T_.count(o)) throw std::runtime_error("PlanModel: output tensor not declared: " + o);

  // R30c：把「4 路 concat -> 1x1 conv」融合（Route A）：conv1x1 走 gemm_f16 的
  // CAT4 B-staging（B 的逻辑 K=Cin 行按 ca/cb/cc/cd 重定向到 4 个源张量），
  // 省掉 concat 的物化（写+再读），同时保留 gemm_f16 的 tile/流水。
  fuseConcatConv1x1();

  // R33：把「conv1x1/conv3x3 + ew_binary(add)」的残差加折进卷积 epilogue（RES），
  // 省掉一次 elementwise launch（数值顺序不变：act(conv+bias)+res）。
  // INFVINO_NO_FUSE_RES=1 可关闭以做 A/B。
  if (!std::getenv("INFVINO_NO_FUSE_RES")) fuseResidualAdd();

  // R51 D5：把「x * scale[c]」（通道广播 Mul，如 SE）折进**消费它的 conv1x1** 的 prologue：
  // conv 直接读未缩放的 value + 每输入通道 scale（kernel -DMUL_SCALE=1），省掉 Mul 的
  // 独立 launch 与其物化张量。**opt-in**（`INFVINO_FUSE_SCALE=1`）：实测整网中性，根因是
  // 被去掉的 Mul 同时承担了 D4 的 fsv16 布局转换（见 docs/round51 §7）；仅当 value 生产者
  // 本就输出 fsv16 时才是纯收益。默认关 → 默认路径与 R50 逐位一致。
  if (std::getenv("INFVINO_FUSE_SCALE")) fuseChannelScaleMul();

  // P0：把激活张量改分配到按生存期复用的缓冲池（须在 fusion 之后，节点列表已定稿）。
  allocateActivations();

  // 预取权重到设备后无需再保留主机侧数据；rt_.write 为阻塞写。
}

void PlanModel::fuseConcatConv1x1()
{
  std::map<std::string, size_t> producer;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (const auto & o : nodes_[i].outs)
      if (o != "-") producer[o] = i;

  std::map<std::string, int> useCount;
  for (const auto & n : nodes_)
    for (const auto & in : n.ins)
      if (in != "-") ++useCount[in];

  std::vector<char> remove(nodes_.size(), 0);
  for (auto & n : nodes_)
  {
    if (n.op != "conv1x1") continue;
    if (n.ins.size() < 2 || n.ins[1] == "-") continue;
    const std::string & xname = n.ins[1];
    auto pit = producer.find(xname);
    if (pit == producer.end()) continue;
    const Node & cat = nodes_[pit->second];
    if (cat.op != "concat4") continue;
    if (useCount[xname] != 1) continue;
    if (attrInt(cat, "outer", 1) != 1) continue;
    int ca = attrInt(cat, "ca", 0), cb = attrInt(cat, "cb", 0),
        cc = attrInt(cat, "cc", 0), cd = attrInt(cat, "cd", 0);
    if (ca <= 0) continue;
    // 4 个槽位；缺失（cnt==0 或 "-"）的源用占位（内核不会索引它）。
    std::string s0 = cat.ins[0], s1 = cat.ins[1], s2 = cat.ins[2], s3 = cat.ins[3];
    int o0 = 0, o1 = 0, o2 = 0, o3 = 0;   // 源内的 channel offset
    const int cnts[4] = {ca, cb, cc, cd};
    std::string * ss[4] = {&s0, &s1, &s2, &s3};
    int * offs[4] = {&o0, &o1, &o2, &o3};
    bool ok = true;
    for (int j = 0; j < 4; ++j)
    {
      if (cnts[j] <= 0) continue;
      std::string & s = *ss[j];
      if (s == "-" || !T_.count(s)) { ok = false; break; }
      // 源若是一个 copy_c(X, c0, cnt) 且 cnt==本段，则可直接用 X 本身 + offset c0，
      // 省掉这次 copy（Split 的一半往往就是父张量的一段）。
      auto cp = producer.find(s);
      if (cp != producer.end() && nodes_[cp->second].op == "copy_c" &&
          useCount[s] == 1 &&
          attrInt(nodes_[cp->second], "cnt", 0) == cnts[j] &&
          attrInt(nodes_[cp->second], "dst_off", 0) == 0 &&
          T_.count(nodes_[cp->second].ins[0]))
      {
        *offs[j] = attrInt(nodes_[cp->second], "c0", 0);
        s = nodes_[cp->second].ins[0];
        remove[cp->second] = 1;
      }
    }
    if (!ok) continue;
    // 占位：缺失源用 s0，cnt=0 保证不被索引。
    if (cb <= 0) { s1 = s0; o1 = 0; }
    if (cc <= 0) { s2 = s0; o2 = 0; }
    if (cd <= 0) { s3 = s0; o3 = 0; }
    n.attr["cat_ca"] = std::to_string(ca);
    n.attr["cat_cb"] = std::to_string(cb);
    n.attr["cat_cc"] = std::to_string(cc);
    n.attr["cat_cd"] = std::to_string(cd);
    n.attr["cat_o0"] = std::to_string(o0);
    n.attr["cat_o1"] = std::to_string(o1);
    n.attr["cat_o2"] = std::to_string(o2);
    n.attr["cat_o3"] = std::to_string(o3);
    n.ins[1] = s0;
    n.ins.insert(n.ins.begin() + 2, s1);
    n.ins.insert(n.ins.begin() + 3, s2);
    n.ins.insert(n.ins.begin() + 4, s3);
    n.op = "conv1x1_cat4";
    ++fusions_concat_;
    remove[pit->second] = 1;
  }

  std::vector<Node> kept;
  kept.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    if (!remove[i]) kept.push_back(std::move(nodes_[i]));
  nodes_ = std::move(kept);
}

// R33: `conv -> ew_binary(add)` 折进卷积 RES epilogue。
//
// 只处理算术上安全的形态：ew_binary op=0（add）、无广播（无 bdims）、且卷积输出只有
// 这一个消费者。折进后卷积输出名改写为 ew 的输出名，ew 节点删除——数值顺序保持
// `act(conv+bias) + res`（gemm/conv_ov 的 RES 就是在激活之后加），与原来
// `conv -> act` 再 `ew_binary(+res)` 一致。
//
// RES 能力：gemm_f16 / gemm_sk / conv1x1_gemv（gemm 族）、conv3x3_ov 支持。
// native conv3x3 / conv3x3_blk 不支持 → run() 里对带 res 的 conv3x3 会绕过这两条
// 调优通路、回退到 ov（见 dispatch），保证正确性。
void PlanModel::fuseResidualAdd()
{
  std::map<std::string, size_t> producer;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (const auto & o : nodes_[i].outs)
      if (o != "-") producer[o] = i;
  std::map<std::string, int> useCount;
  for (const auto & n : nodes_)
    for (const auto & in : n.ins)
      if (in != "-") ++useCount[in];

  auto resCapableConv = [&](const std::string & op) {
    // R33 实测：conv3x3 带 res 需绕开 blk/native 调优（它们不支持 RES）→ 强制 ov，
    // 对 20×20/40×40 这类本该走 blk 的层得不偿失（整网 +4%）。因此只融合 gemm 族
    // （conv1x1/conv1x1_cat4）：RES 与原来「act 后单独 add」是同一 fp16 累加路径。
    return op == "conv1x1" || op == "conv1x1_cat4";
  };
  auto resSlotOf = [](const std::string & op) -> size_t {
    if (op == "conv1x1_cat4") return 6;   // [w,b0..b3,bias,res]
    return 3;                             // conv1x1 / conv3x3: [w,x,bias,res]
  };

  std::vector<char> remove(nodes_.size(), 0);
  std::map<std::string, int> isOutput;
  for (const auto & o : outputs_) isOutput[o] = 1;
  int nfused = 0;
  for (size_t i = 0; i < nodes_.size(); ++i)
  {
    Node & e = nodes_[i];
    if (e.op != "ew_binary") continue;
    if (attrInt(e, "op", 0) != 0) continue;                    // add only
    if (e.attr.count("bdims") && attrInt(e, "b_scalar", 0) != 1) continue;  // no broadcast
    if (e.ins.size() < 2) continue;
    for (int k = 0; k < 2; ++k)
    {
      const std::string & pname = e.ins[k];
      const std::string & rname = e.ins[1 - k];
      auto pit = producer.find(pname);
      if (pit == producer.end()) continue;
      Node & c = nodes_[pit->second];
      if (!resCapableConv(c.op)) continue;
      if (useCount[pname] != 1) continue;                      // 卷积输出只喂这个 add
      if (isOutput.count(pname)) continue;                      // 卷积输出是模型输出
      if (c.outs.size() != 1 || c.outs[0] != pname) continue;
      if (rname == "-" || !T_.count(rname) || !T_.count(pname)) continue;
      const size_t rs = resSlotOf(c.op);
      if (c.ins.size() > rs && c.ins[rs] != "-") continue;     // 已有残差
      // 残差必须在卷积之前产出（拓扑序），否则折入会读到未写入的缓冲。
      auto rit = producer.find(rname);
      if (rit == producer.end() || rit->second >= pit->second) continue;
      if (c.ins.size() <= rs) c.ins.resize(rs + 1, "-");
      c.ins[rs]          = rname;
      c.outs[0]          = e.outs[0];   // 卷积改写到 ew 的输出张量
      remove[i]          = 1;
      producer.erase(pname);
      producer[e.outs[0]] = pit->second;
      ++nfused;
      break;
    }
  }
  if (nfused == 0) return;
  fusions_res_ += nfused;
  std::vector<Node> kept;
  kept.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    if (!remove[i]) kept.push_back(std::move(nodes_[i]));
  nodes_ = std::move(kept);
}

// R51 D5: `x * scale[c]`（通道广播 Mul，如 SE）折进消费它的 conv1x1 的 **prologue**。
// 形态：ew_binary(op=2, bdims) 输出是某个 conv1x1 的**激活**（ins[1]）且只有这一个消费者；
// 两个操作数按空间范围区分——scale 是 [1,C,1,1]（H*W==1），value 是 [1,C,H,W]。
// 改写：conv1x1.ins[1] = value，新增 ins[4] = scale；删除 Mul 及其实的名字。
// kernel 侧 `-DMUL_SCALE=1` 在载入激活时做 `half(value*scale[gc])`（与 Mul 的 half 舍入
// 一致）→ 逐位等价，同时省掉 Mul 的独立 launch 与物化张量。
void PlanModel::fuseChannelScaleMul()
{
  std::map<std::string, size_t> producer;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (const auto & o : nodes_[i].outs)
      if (o != "-") producer[o] = i;
  std::map<std::string, int> useCount;
  for (const auto & n : nodes_)
    for (const auto & in : n.ins)
      if (in != "-") ++useCount[in];

  std::vector<char> remove(nodes_.size(), 0);
  std::vector<std::string> dead;
  auto spatial = [](const std::vector<int64_t> & d) -> int64_t {
    return (d.size() >= 2) ? d[d.size() - 1] * d[d.size() - 2] : 1;
  };
  for (size_t i = 0; i < nodes_.size(); ++i)
  {
    Node & m = nodes_[i];
    if (m.op != "ew_binary" || !m.attr.count("bdims")) continue;   // 广播乘
    if (attrInt(m, "op", 0) != 2) continue;                        // multiply only
    if (m.ins.size() < 2 || m.outs.empty() || m.outs[0] == "-") continue;
    const std::string & mulOut = m.outs[0];
    auto d0 = T_.find(m.ins[0]), d1 = T_.find(m.ins[1]);
    if (d0 == T_.end() || d1 == T_.end()) continue;
    int scaleIdx = -1, valIdx = -1;
    if (spatial(d0->second.dims) == 1 && spatial(d1->second.dims) > 1) { scaleIdx = 0; valIdx = 1; }
    else if (spatial(d1->second.dims) == 1 && spatial(d0->second.dims) > 1) { scaleIdx = 1; valIdx = 0; }
    else continue;
    const std::string valT = m.ins[valIdx], scaleT = m.ins[scaleIdx];
    if (valT == "-" || scaleT == "-") continue;
    if (useCount[mulOut] != 1) continue;
    if (std::find(outputs_.begin(), outputs_.end(), mulOut) != outputs_.end()) continue;  // 模型输出
    // 唯一消费者必须是 conv1x1 且在激活槽 1。
    size_t ci = nodes_.size();
    int slot = -1;
    for (size_t j = 0; j < nodes_.size(); ++j)
      for (size_t s = 0; s < nodes_[j].ins.size(); ++s)
        if (nodes_[j].ins[s] == mulOut) { ci = j; slot = static_cast<int>(s); }
    if (ci >= nodes_.size()) continue;
    Node & c = nodes_[ci];
    if (c.op != "conv1x1" || slot != 1) continue;
    if (!T_.count(valT) || !T_.count(scaleT)) continue;
    if (c.ins.size() > 4 && c.ins[4] != "-") continue;             // 已有 scale
    auto vit = producer.find(valT);
    if (vit == producer.end() || vit->second >= ci) continue;      // value 须在 conv 前产出
    const auto & vd = T_[valT].dims, &sd = T_[scaleT].dims;
    if (vd.size() < 3 || sd.size() < 3) continue;
    if (vd[vd.size() - 3] <= 0 || vd[vd.size() - 3] != sd[sd.size() - 3]) continue;
    if (c.ins.size() <= 4) c.ins.resize(5, "-");
    c.ins[1] = valT;
    c.ins[4] = scaleT;
    remove[i] = 1;
    dead.push_back(mulOut);
    ++fusions_scale_;
  }
  if (dead.empty()) return;
  std::vector<Node> kept;
  kept.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i)
    if (!remove[i]) kept.push_back(std::move(nodes_[i]));
  nodes_ = std::move(kept);
  // Mul 输出张量已死：从池候选与 T_ 中移除（避免按「活到最后」多占一块缓冲）。
  for (const auto & t : dead)
  {
    T_.erase(t);
    act_names_.erase(std::remove(act_names_.begin(), act_names_.end(), t), act_names_.end());
  }
}

void PlanModel::allocateActivations()
{
  const size_t N = nodes_.size();
  if (std::getenv("INFVINO_NO_POOL")) return;   // 诊断开关：退回到每张量一块

  // 0) P0-offset：byte-offset 子分配（**默认开**；见 ClRuntime.hpp ActPool 注释）。
  //    实测（Iris Xe）：y8 24.0→20.6 MB / y11 26.5→23.1 MB，e2e −1.4%/−1.8%，
  //    逐位一致。`INFVINO_NO_POOL_OFFSET=1` 退回整块复用（对照/排查用）。
  {
    const char * no = std::getenv("INFVINO_NO_POOL_OFFSET");
    if (!(no && std::string(no) != "0" && std::string(no) != ""))
    {
      act_pool_.setOffsetEnabled(true);
      act_pool_.setAlign(rt_.info().mem_base_align ? rt_.info().mem_base_align : 64);
    }
  }
  std::unordered_set<std::string> alias_ok;   // 成功建立子 buffer 别名的输出张量

  // 1) 每个张量的 [birth, death]（拓扑序 = 执行序）。
  std::unordered_map<std::string, int> birth, death;
  auto touch = [&](const std::string & t, int i, bool write) {
    if (t == "-") return;
    if (write) {
      if (!birth.count(t)) birth[t] = i;
      death[t] = i;
    } else {
      if (!birth.count(t)) birth[t] = i;   // 常量/输入被当输入读
      death[t] = i;
    }
  };
  for (size_t i = 0; i < N; ++i)
  {
    for (const auto & in : nodes_[i].ins) touch(in, static_cast<int>(i), false);
    for (const auto & o : nodes_[i].outs) touch(o, static_cast<int>(i), true);
  }

  // 1b) 零拷贝视图 / 连续切片别名：必须与其源**共享生存期**——否则源 buffer 会在视图仍被
  //     读取时被池复用掉（这是 R-P0 第一版数值 FAIL 的根因）。
  //     (a) reshape/flatten：run() 里 out.mem = in.mem（整块共享）。
  //     (b) 连续 copy_c（dst_off=0）：输出是父张量的一段连续通道范围
  //         [c0*HW, (c0+cnt)*HW)，可用 clCreateSubBuffer 直接别名（消费者零改动）。
  std::unordered_map<std::string, std::string> alias_parent;   // 输出 -> 源（整块）
  std::unordered_map<std::string, std::pair<std::string, int64_t>> alias_sub;  // 输出 -> (源, 字节偏移)
  for (const auto & n : nodes_)
  {
    if (n.ins.empty() || n.outs.empty() || n.ins[0] == "-" || n.outs[0] == "-") continue;
    if (n.op == "reshape" || n.op == "flatten")
      alias_parent[n.outs[0]] = n.ins[0];
    else if (n.op == "copy_c" && attrInt(n, "dst_off", 0) == 0)
    {
      const int64_t HW = attrInt(n, "HW", 0), c0 = attrInt(n, "c0", 0),
                    cnt = attrInt(n, "cnt", 0);
      auto pit = T_.find(n.ins[0]);
      auto oit = T_.find(n.outs[0]);
      // 需求：源是连续张量；输出的元素数 == cnt*HW 且等于父张量的一段（父的通道数
      // 可不严格等于 c0+cnt，但必须有足够数据）。offset = c0*HW 元素 → 字节 = ×2。
      if (HW > 0 && cnt > 0 && pit != T_.end() && oit != T_.end())
      {
        const int64_t need = static_cast<int64_t>(c0 + cnt) * HW;
        if (pit->second.numel() >= need &&
            oit->second.numel() == static_cast<int64_t>(cnt) * HW)
        {
          alias_sub[n.outs[0]] = {n.ins[0], static_cast<int64_t>(c0) * HW * 2};
          alias_parent[n.outs[0]] = n.ins[0];   // 用于生存期并集
        }
      }
    }
  }
  auto root_of = [&](std::string t) {
    int guard = 0;
    while (alias_parent.count(t) && guard++ < 1000) t = alias_parent[t];
    return t;
  };
  // 把同名视图的区间并入根；活动张量统一用根名登记。
  {
    std::unordered_map<std::string, std::pair<int, int>> merged;
    for (const auto & name : act_names_)
    {
      auto bit = T_.find(name);
      if (bit == T_.end()) continue;
      int b = birth.count(name) ? birth[name] : 0;
      int d = death.count(name) ? death[name] : static_cast<int>(N ? N - 1 : 0);
      std::string r = root_of(name);
      auto it = merged.find(r);
      if (it == merged.end()) merged[r] = {b, d};
      else { it->second.first = std::min(it->second.first, b);
             it->second.second = std::max(it->second.second, d); }
    }
    // 用根的并集区间覆盖每个成员的 birth/death，保证同根张量占据同一生存期。
    for (const auto & name : act_names_)
      if (birth.count(name) || death.count(name) || merged.count(root_of(name)))
      {
        auto it = merged.find(root_of(name));
        if (it != merged.end()) { birth[name] = it->second.first; death[name] = it->second.second; }
      }
  }

  std::unordered_map<std::string, int> out_set;
  for (size_t i = 0; i < outputs_.size(); ++i) out_set[outputs_[i]] = 1;

  // R51 (R48 §4bis P1): 只有**可能被持久化为 fsv16** 的张量才按补齐通道分配。否则
  // 无谓地放大所有非对齐张量（y11 实测 requested 69→140MB，busy +1.2%）。
  //
  // 判据是 planBlockedLayout/mincut 标记条件的**超集**（按族 supports，而非当前选中）：
  //   * 4-D conv 张量；
  //   * 生产者 op 有族/候选声明 canOutFsv16（conv1x1/depthwise/conv3x3/ew_binary_ch）；
  //   * 每个消费者都能在其「激活输入槽」读 fsv16（conv1x1/depthwise/conv3x3）。
  // 超集保证：任何实际被标记 fsv16 的张量都已被补齐（不越界）。
  auto opMayOutFsv16 = [&](const Node & nd) -> bool {
    bool ok = false;
    const OpSignature s = nodeSignature(nd, &ok);
    if (!ok) return false;
    for (const auto & f : kernelFamilies())
    {
      if (!(f.actMask & (1 << s.act))) continue;
      if (f.supports && !f.supports(s)) continue;
      if (f.layout.canOutFsv16) return true;
      if (f.candidates)
        for (const auto & c : f.candidates(s))
          if (c.canOutFsv16) return true;
    }
    return false;
  };
  auto opCanReadFsv16 = [&](const Node & nd, int slot) -> bool {
    bool ok = false;
    const OpSignature s = nodeSignature(nd, &ok);
    if (!ok) return false;
    for (const auto & f : kernelFamilies())
    {
      if (f.layout.in != Layout::FSV16 || f.layout.inIndex != slot) continue;
      if (!(f.actMask & (1 << s.act))) continue;
      if (f.supports && !f.supports(s)) continue;
      return true;
    }
    return false;
  };
  std::unordered_map<std::string, int> producerIdx;
  for (size_t i = 0; i < N; ++i)
    for (const auto & o : nodes_[i].outs)
      if (o != "-") producerIdx[o] = static_cast<int>(i);
  std::unordered_map<std::string, std::vector<std::pair<int, int>>> consumerIdx;
  for (size_t i = 0; i < N; ++i)
    for (size_t s = 0; s < nodes_[i].ins.size(); ++s)
      if (nodes_[i].ins[s] != "-")
        consumerIdx[nodes_[i].ins[s]].push_back({static_cast<int>(i), static_cast<int>(s)});
  auto mayBeFsv16 = [&](const std::string & name) -> bool {
    auto bit = T_.find(name);
    if (bit == T_.end() || bit->second.dims.size() != 4) return false;
    if (out_set.count(name)) return false;               // 网络输出必须 NCHW
    auto pit = producerIdx.find(name);
    if (pit == producerIdx.end()) return false;          // 网络输入
    if (!opMayOutFsv16(nodes_[pit->second])) return false;
    auto cit = consumerIdx.find(name);
    if (cit == consumerIdx.end() || cit->second.empty()) return false;
    for (const auto & c : cit->second)
      if (!opCanReadFsv16(nodes_[c.first], c.second)) return false;
    return true;
  };

  // 2) 标记哪些 activation 张量参与复用；给每个一个 id。
  struct Live { int id, birth, death; int64_t bytes; std::string name; };
  std::vector<Live> lives;
  std::unordered_map<std::string, int> id_of;
  for (const auto & name : act_names_)
  {
    auto bit = T_.find(name);
    if (bit == T_.end()) continue;
    int b = birth.count(name) ? birth[name] : 0;
    int d = death.count(name) ? death[name] : static_cast<int>(N ? N - 1 : 0);
    if (out_set.count(name)) d = static_cast<int>(N ? N - 1 : 0);   // 输出活到最后
    int id = static_cast<int>(lives.size());
    id_of[name] = id;
    // R51: 与 parse 的分配一致 —— 按补齐通道计算 footprint（可容纳生产者直写 fsv16）。
    const bool pad = mayBeFsv16(name);
    if (pad) fsv16_capable_.insert(name);
    const int64_t bytes =
        static_cast<int64_t>(pad ? paddedChannelNumel(bit->second.dims) : bit->second.numel()) * 2;
    lives.push_back({id, b, d, bytes, name});
  }

  // 3) 冲突集：生存期重叠（闭区间相交）的 id 集合。
  std::vector<std::vector<int>> conflict(lives.size());
  for (size_t i = 0; i < lives.size(); ++i)
    for (size_t j = 0; j < lives.size(); ++j)
      if (i != j && lives[i].birth <= lives[j].death && lives[j].birth <= lives[i].death)
        conflict[i].push_back(lives[j].id);

  // 4) 逐个 acquire；旧 buffer 释放（从 owned_ 中摘除）。
  //    按 footprint 降序分配，让大块优先占据紧凑区域（减少碎片/浪费）。
  std::vector<size_t> order(lives.size());
  for (size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::sort(order.begin(), order.end(),
            [&](size_t a, size_t b) { return lives[a].bytes > lives[b].bytes; });

  // 诊断：分配后校验「共享同一 buffer 的任意两张量生存期不相交」。
  auto verify = [&]() {
    for (const auto & s : act_pool_.debugSharing())
    {
      std::vector<int> ids;
      size_t p = s.find("users=");
      std::string u = s.substr(p + 6);
      std::stringstream ss(u);
      std::string tok;
      while (std::getline(ss, tok, ',')) if (!tok.empty()) ids.push_back(std::atoi(tok.c_str()));
      for (size_t a = 0; a < ids.size(); ++a)
        for (size_t b = a + 1; b < ids.size(); ++b)
        {
          const Live & x = lives[ids[a]], &y = lives[ids[b]];
          if (x.birth <= y.death && y.birth <= x.death)
            std::fprintf(stderr, "[pool][BUG] overlap share: %s [%d,%d] <-> %s [%d,%d]\n",
                         x.name.c_str(), x.birth, x.death, y.name.c_str(), y.birth, y.death);
        }
    }
  };

  std::unordered_map<cl_mem, int> old_ref;
  for (auto & l : lives)
    old_ref[T_[l.name].mem] = id_of[l.name];

  // 诊断：INFVINO_POOL_LIMIT=k → 只对前 k 个（声明序）张量做池化，其余保持独立 buffer。
  size_t pool_limit = lives.size();
  if (const char * lim = std::getenv("INFVINO_POOL_LIMIT"))
    pool_limit = static_cast<size_t>(std::atoi(lim));

  for (size_t k : order)
  {
    auto & l = lives[k];
    if (alias_parent.count(l.name)) continue;   // 别名：不单独分配，稍后指向源
    if (static_cast<size_t>(l.id) >= pool_limit) continue;
    ActPool::Ref pref = act_pool_.acquire(
      [&](size_t bytes) { return rt_.alloc(bytes, CL_MEM_READ_WRITE); }, l.bytes,
      conflict[l.id], l.id);
    // 旧 buffer 释放 + 从 owned_ 移除
    cl_mem old = T_[l.name].mem;
    if (old)
    {
      auto oit = std::find(owned_.begin(), owned_.end(), old);
      if (oit != owned_.end()) owned_.erase(oit);
      if (old_ref.count(old) == 1) { clReleaseMemObject(old); old_ref.erase(old); }
    }
    T_[l.name].mem      = pref.mem;
    T_[l.name].base     = pref.base;
    T_[l.name].base_off = static_cast<int64_t>(pref.off);
  }

  // 4b) 视图/切片别名：reshape/flatten 直接指向源；连续 copy_c 建子 buffer（源内偏移）。
  //     源此刻已经拿到池 buffer（root_of 保证别名在源之后处理：源不是别名）。
  for (auto & kv : alias_parent)
  {
    const std::string & out_name = kv.first;
    auto oit = T_.find(out_name);
    if (oit == T_.end()) continue;
    cl_mem old = oit->second.mem;
    if (old)
    {
      auto fit = std::find(owned_.begin(), owned_.end(), old);
      if (fit != owned_.end()) owned_.erase(fit);
      if (old_ref.count(old) == 1) { clReleaseMemObject(old); old_ref.erase(old); }
    }
    const std::string & src = kv.second;
    auto sit = T_.find(src);
    if (sit == T_.end()) { continue; }
    auto sub = alias_sub.find(out_name);
    if (sub == alias_sub.end())
    {
      oit->second.mem      = sit->second.mem;         // 整块视图（reshape/flatten）
      oit->second.base     = sit->second.base;
      oit->second.base_off = sit->second.base_off;
    }
    else
    {
      // 连续 copy_c：在源张量底层 arena（base）上按「源内偏移 + 切片偏移」建子 buffer。
      // 子 buffer 不能基于子 buffer 再切，所以必须用 base 而不是源自身的 mem。
      const size_t origin = static_cast<size_t>(sit->second.base_off + sub->second.second);
      const size_t nbytes  = static_cast<size_t>(oit->second.numel() * 2);
      cl_int err = CL_SUCCESS;
      cl_mem sb = nullptr;
      if (origin % (rt_.info().mem_base_align ? rt_.info().mem_base_align : 1) == 0)
      {
        cl_buffer_region region{origin, 0};
        region.size = nbytes;
        sb = clCreateSubBuffer(sit->second.base, CL_MEM_READ_WRITE,
                               CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
      }
      if (err == CL_SUCCESS && sb)
      {
        oit->second.mem      = sb;
        oit->second.base     = sit->second.base;
        oit->second.base_off = static_cast<int64_t>(origin);
        alias_subs_.push_back(sb);
        alias_ok.insert(out_name);
      }
      else
      {
        // 无法对齐/建子 buffer → 不别名，给输出单独分配，让 copy_c 正常执行（保数值）。
        cl_mem fresh = rt_.alloc(nbytes, CL_MEM_READ_WRITE);
        oit->second.mem = fresh; oit->second.base = fresh; oit->second.base_off = 0;
        owned_.push_back(fresh);
      }
    }
  }

  // 4c) 标记「被别名取代」的 copy_c 节点：run() 不再 launch 它（源数据已在别名里）。
  node_skipped_.assign(N, 0);
  for (size_t i = 0; i < N; ++i)
    if (nodes_[i].op == "copy_c" && !nodes_[i].outs.empty() &&
        alias_ok.count(nodes_[i].outs[0]))
      node_skipped_[i] = 1;
  verify();
  if (std::getenv("INFVINO_POOL_MAP"))
  {
    std::fprintf(stderr, "[map] requested=%zu allocated=%zu bufs=%zu\n",
                 act_pool_.requestedBytes(), act_pool_.allocatedBytes(),
                 act_pool_.bufferCount());
    for (auto & s : act_pool_.debugSharing()) std::fprintf(stderr, "[map] %s\n", s.c_str());
    for (const auto & l : lives)
      std::fprintf(stderr, "[map] id=%d [%d,%d] %s\n", l.id, l.birth, l.death, l.name.c_str());
  }
}

bool PlanModel::convWillUseBlk(const Node & n) const
{
  // 与 run() 里 conv3x3 分支的 kernel 选择保持**完全一致**，否则布局规划会和实际执行的
  // 通路脱节（规划说 fsv16、执行却走了 native，消费者就会读到非阻塞布局）。
  if (n.op != "conv3x3" || n.ins.empty() || n.outs.empty()) return false;
  auto xit = T_.find(n.ins[0]);
  auto oit = T_.find(n.outs[0]);
  if (xit == T_.end() || oit == T_.end()) return false;
  const auto & id = xit->second.dims;
  const auto & od = oit->second.dims;
  const size_t xb = id.size() >= 3 ? id.size() - 3 : 0;
  const size_t ob = od.size() >= 3 ? od.size() - 3 : 0;
  const int Cin = static_cast<int>(id[xb]);
  const int Cout = static_cast<int>(od[ob]);
  const int stride = attrInt(n, "stride", 1), pad = attrInt(n, "pad", 1),
            act = attrInt(n, "act", 0);
  const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
  const bool dres = n.ins.size() > 3 && n.ins[3] != "-";
  const OpSignature tsig = OpSignature::conv3x3(Wout, Hout, stride, pad, Cin, Cout, act);
  // R38: prefer the per-node joint-fixpoint choice; else the signature cache.
  const size_t ni = static_cast<size_t>(&n - nodes_.data());
  const TuningEntry * te = choiceEntry(ni, tsig);
  // R33: 带残差的 conv3x3 只有 conv3x3_ov 支持 RES；非 ov 命中会被绕回。
  if (dres && te && te->kernel != "conv3x3_ov") te = nullptr;
  if (te && te->kernel == "conv3x3_blk") return true;
  // 未命中时的内置启发式：只有显式 blk 属性才走 blocked（且 blk 不支持 RES）。
  if (attrInt(n, "blk", 0) != 0 && !te && !dres) return true;
  return false;
}

void PlanModel::planBlockedLayout()
{
  // R49: mincut 一旦接管布局，本函数不再重新推导（否则会覆盖全局最解）。
  if (mincut_active_) return;
  // 可重入：先清空所有张量的 fsv16，再依据**当前 tuning_**重新规划（autotune 之后
  // 布局会变；本函数在构造期与 autotune 末尾各调一次）。
  for (auto & kv : T_) kv.second.fsv16 = false;
  if (std::getenv("INFVINO_NO_BLOCK_LAYOUT")) return;
  // R48 D4 诊断开关：关闭「按族激活槽判定持久化」与「小算子生产者直写 fsv16」，用于 A/B。
  const bool d4 = std::getenv("INFVINO_NO_D4") == nullptr;

  std::unordered_map<std::string, std::vector<std::pair<size_t, int>>> consumers;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (size_t s = 0; s < nodes_[i].ins.size(); ++s)
      if (nodes_[i].ins[s] != "-") consumers[nodes_[i].ins[s]].push_back({i, static_cast<int>(s)});

  const std::unordered_set<std::string> out_set(outputs_.begin(), outputs_.end());

  // 通用布局框架：每个节点「选中的族」（由 tuning 驱动）给出它的 in/out 布局契约。
  // 生产者能写 FSV16 且其**所有**消费者都只吃 FSV16 → 该张量持久 FSV16（零 reorder）。
  // 这是 R36「只覆盖 conv3x3」的推广：新增 blocked 族（conv1x1_blk/depthwise_blk）
  // 只要在注册表声明布局，规划器自动接上，无需改这里。
  auto nodeFamily = [&](const Node & n) -> const KernelFamily * {
    if (n.op == "conv3x3")
      return familyByName(convWillUseBlk(n) ? "conv3x3_blk" : "conv3x3_ov");
    if (n.op == "conv1x1" && n.ins.size() >= 2 && !n.outs.empty()) {
      auto wit = T_.find(n.ins[0]), xit = T_.find(n.ins[1]);
      if (wit == T_.end() || xit == T_.end() || wit->second.dims.size() < 2) return nullptr;
      const int Cout = static_cast<int>(wit->second.dims[0]);
      const int Cin = static_cast<int>(wit->second.dims[1]);
      const int64_t xn = xit->second.numel();
      const int N = Cin > 0 ? static_cast<int>(xn / Cin) : 0;
      const int act = attrInt(n, "act", 0);
      const OpSignature sig = conv1x1Sig(n, Cout, N, Cin, act);
      const TuningEntry * e = choiceEntry(static_cast<size_t>(&n - nodes_.data()), sig);
      if (e && e->kernel == "conv1x1_blk") return familyByName("conv1x1_blk");
    }
    if (n.op == "conv_general" && n.ins.size() >= 2 && !n.outs.empty()) {
      auto xit = T_.find(n.ins[0]), oit = T_.find(n.outs[0]);
      if (xit == T_.end() || oit == T_.end()) return nullptr;
      const auto & id = xit->second.dims;
      const auto & od = oit->second.dims;
      if (id.size() < 3 || od.size() < 3) return nullptr;
      const int Cin = static_cast<int>(id[id.size() - 3]);
      const int Cout = static_cast<int>(od[od.size() - 3]);
      const int K = attrInt(n, "K", 3), S = attrInt(n, "S", 1), P = attrInt(n, "P", 1),
                act = attrInt(n, "act", 0), G = attrInt(n, "G", 1);
      if (!(G == Cin && (K == 3 || K == 5) && Cin == Cout)) return nullptr;
      const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
      const OpSignature sig = OpSignature::depthwise(Wout, Hout, S, P, Cin, K, act);
      const TuningEntry * e = choiceEntry(static_cast<size_t>(&n - nodes_.data()), sig);
      if (e && e->kernel == "depthwise_blk") return familyByName("depthwise_blk");
    }
    // R51 D4+: gap 视为「可读 fsv16」的消费者（kernel 侧 -DGAP_IN_FSV16）。
    if (n.op == "gap") return familyByName("gap_fsv16");
    return nullptr;
  };

  int marked = 0;
  for (size_t i = 0; i < nodes_.size(); ++i)
  {
    const Node & n = nodes_[i];
    const KernelFamily * prod = nodeFamily(n);
    if (!prod || !prod->layout.canOutFsv16 || n.outs.empty()) continue;
    const std::string & t = n.outs[0];
    auto tit = T_.find(t);
    if (tit == T_.end()) continue;
    if (out_set.count(t)) continue;                 // 网络输出必须是 NCHW
    const auto & od = tit->second.dims;
    const size_t ob = od.size() >= 3 ? od.size() - 3 : 0;
    const int Cout = static_cast<int>(od[ob]);
    // R51: fsv16 按 16 通道补齐；分配池已按 paddedChannelNumel 分配（含补齐），因此
    // C%16!=0 不再越界 —— 移除此前的 NCHW pin，解锁非对齐张量进 blocked 链。
    if (Cout <= 0) continue;
    auto cit = consumers.find(t);
    if (cit == consumers.end() || cit->second.empty()) continue;
    bool all_blk = true;
    for (const auto & c : cit->second)
    {
      const KernelFamily * cf = nodeFamily(nodes_[c.first]);
      // R48 D4: 用族声明的「激活输入槽」判定——conv1x1/gemm 的激活在槽 1（槽 0 是权重）。
      const int slot = d4 && cf ? cf->layout.inIndex : 0;
      if (!cf || cf->layout.in != Layout::FSV16 || c.second != slot)
      {
        all_blk = false;
        break;
      }
    }
    if (!all_blk) continue;                          // 有非 blocked 消费者 → 保持 NCHW
    tit->second.fsv16 = true;
    ++marked;
  }

  // R48 D4: 小算子生产者直接写 fsv16（消费者全是 blocked 时），省掉独立 reorder pass。
  // 目前只覆盖 `ew_binary_ch`（通道广播，如 SE 的 Mul；唯一实现了 -DEWCH_OUT_FSV16）。
  // 与 run() 的 kernel 选择保持一致：优先 tuning 选中的 kernel，否则内置启发式。
  for (size_t i = 0; d4 && i < nodes_.size(); ++i)
  {
    const Node & n = nodes_[i];
    if (n.op != "ew_binary" || n.outs.empty()) continue;
    const std::string & t = n.outs[0];
    if (out_set.count(t)) continue;
    auto tit = T_.find(t);
    if (tit == T_.end()) continue;
    const auto & od = tit->second.dims;
    if (od.size() < 3) continue;
    const int C = static_cast<int>(od[od.size() - 3]);
    if (C <= 0) continue;
    const OpSignature sig = smallSig(n);
    if (sig.op != "ew_binary_bcast") continue;   // 通道广播（bdims）才可能走 ew_binary_ch
    const long nn = sig.params.size() > 0 ? sig.params[0] : 0;
    const long Cs = sig.params.size() > 3 ? sig.params[3] : 0;
    if (!(Cs > 0 && Cs < nn && nn % Cs == 0)) continue;
    const TuningEntry * e = choiceEntry(i, sig);
    const bool isCh = (e && !e->kernel.empty()) ? (e->kernel == "ew_binary_ch")
                                                : (Cs > 0 && Cs < nn && nn % Cs == 0);
    if (!isCh) continue;
    auto cit = consumers.find(t);
    if (cit == consumers.end() || cit->second.empty()) continue;
    bool all_blk = true;
    for (const auto & c : cit->second)
    {
      const KernelFamily * cf = nodeFamily(nodes_[c.first]);
      if (!cf || cf->layout.in != Layout::FSV16 || c.second != cf->layout.inIndex)
      {
        all_blk = false;
        break;
      }
    }
    if (!all_blk) continue;
    tit->second.fsv16 = true;
    ++marked;
  }
  (void)d4;

  if (std::getenv("INFVINO_LAYOUT_REPORT"))
    std::fprintf(stderr, "[layout] fsv16 tensors: %d (family-driven persistent block layout)\n",
                 marked);
}

OpSignature PlanModel::planNodeKey(const std::string & out_name) const
{
  return OpSignature::custom("plan@" + out_name, {});
}

const TuningEntry * PlanModel::choiceEntry(size_t ni, const OpSignature & sig) const
{
  // R45 P0#4: per-plan 覆盖优先级最高（本图的位置相关全局最优）。
  if (ni < nodes_.size() && !nodes_[ni].outs.empty())
    if (const TuningEntry * e = plan_overrides_.lookup(planNodeKey(nodes_[ni].outs[0])))
      if (!e->kernel.empty()) return e;
  if (ni < node_choice_.size() && !node_choice_[ni].kernel.empty()) return &node_choice_[ni];
  return tuning_.lookup(sig);
}

void PlanModel::resolveLayoutChoices()
{
  node_choice_.assign(nodes_.size(), {});
  // Only conv3x3 has the blk/non-blk reorder dichotomy today; other families keep the
  // plain signature-cache choice (their conservative billing lives in Autotuner).
  std::vector<LayoutAlt> alt(nodes_.size());
  bool any = false;
  for (size_t ni = 0; ni < nodes_.size(); ++ni)
  {
    bool ok = false;
    const OpSignature sig = nodeSignature(nodes_[ni], &ok);
    if (!ok) continue;   // R43: generalized beyond conv3x3 (any op with #blk/#non)
    const TuningEntry * b = tuning_.lookup(OpSignature::custom(sig.str() + "#blk", {}));
    const TuningEntry * n = tuning_.lookup(OpSignature::custom(sig.str() + "#non", {}));
    if (!b || !n) continue;
    alt[ni].blk = *b;
    alt[ni].non = *n;
    if (const TuningEntry * r = tuning_.lookup(OpSignature::custom(sig.str() + "#reorder", {})))
      alt[ni].reorder = *r;
    // R50: 输出 fsv16 的 blocked 成本（契约 canOutFsv16）。缺失时回退为 blk.ms（旧行为）。
    if (const TuningEntry * bf = tuning_.lookup(OpSignature::custom(sig.str() + "#blkfsv16", {})))
      if (!bf->kernel.empty()) alt[ni].blkFsv16 = *bf;
    alt[ni].has = true;
    any = true;
  }
  if (!any)
  {
    planBlockedLayout();  // R36 behavior
    return;
  }

  const bool mincutOn = std::getenv("INFVINO_LAYOUT_MINCUT") != nullptr;

  const int kIters = 4;
  for (int it = 0; it < kIters; ++it)
  {
    planBlockedLayout();
    bool changed = false;
    for (size_t ni = 0; ni < nodes_.size(); ++ni)
    {
      if (!alt[ni].has) continue;
      const Node & n = nodes_[ni];
      // Is this node's input already persisted fsv16 (reorder free)?
      // NB: conv1x1 keeps weights first (ins[0]) and the activation at ins[1];
      // conv3x3 / depthwise keep the activation at ins[0].
      const size_t inIdx = (n.op == "conv1x1" || n.op == "conv1x1_cat4") ? 1 : 0;
      bool in_fsv16 = false;
      if (n.ins.size() > inIdx)
      {
        auto xit = T_.find(n.ins[inIdx]);
        if (xit != T_.end()) in_fsv16 = xit->second.fsv16;
      }
      // R50: 若本节点输出已被规划为 fsv16（契约 canOutFsv16），用 fsv16 输出的实测成本
      // （`#blkfsv16`）；否则用 bfyx 输出成本。缺失 `#blkfsv16` 时回退为旧行为。
      bool out_fsv16 = false;
      if (!n.outs.empty())
      {
        auto oit = T_.find(n.outs[0]);
        if (oit != T_.end()) out_fsv16 = oit->second.fsv16;
      }
      const double blkKernelMs =
          (out_fsv16 && !alt[ni].blkFsv16.kernel.empty()) ? alt[ni].blkFsv16.ms : alt[ni].blk.ms;
      double blkCost = blkKernelMs + (in_fsv16 ? 0.0 : alt[ni].reorder.ms);
      const TuningEntry & want = (blkCost <= alt[ni].non.ms) ? alt[ni].blk : alt[ni].non;
      if (node_choice_[ni].kernel != want.kernel || node_choice_[ni].options != want.options)
      {
        node_choice_[ni] = want;
        changed = true;
      }
    }
    if (!changed) break;
  }
  planBlockedLayout();  // final layout consistent with node_choice_
  if (std::getenv("INFVINO_LAYOUT_REPORT"))
  {
    int nblk = 0;
    for (const auto & c : node_choice_) if (c.kernel == "conv3x3_blk") ++nblk;
    std::fprintf(stderr, "[layout] joint fixpoint: %d conv3x3 nodes, %d -> blk\n",
                 static_cast<int>(node_choice_.size()), nblk);
  }

  // R49 步骤 2：baseline（联合不动点）就绪后，把布局决策交给**精确最小割**。
  // 隔离 ms 只用于**生成**布局提案；若开启验收门（INFVINO_LAYOUT_MINCUT_GATE=1）且
  // profiling 可用，则用**整网交错 median** 对比 baseline vs mincut，只有改善 > 1% 才接受，
  // 否则回退——不在噪声上下结论（这是 R44–R48 反复踩的坑）。
  if (mincutOn)
  {
    struct LayoutState
    {
      std::vector<TuningEntry>              choice;
      std::unordered_map<std::string, char> fsv;
      bool                                  mincut = false;
    };
    auto snap = [&]() {
      LayoutState s;
      s.choice = node_choice_;
      s.mincut = mincut_active_;
      for (const auto & kv : T_) s.fsv[kv.first] = kv.second.fsv16 ? 1 : 0;
      return s;
    };
    auto restore = [&](const LayoutState & s) {
      node_choice_ = s.choice;
      mincut_active_ = s.mincut;
      for (auto & kv : T_)
      {
        auto it = s.fsv.find(kv.first);
        if (it != s.fsv.end()) kv.second.fsv16 = (it->second != 0);
      }
      invalidateCapture();  // 布局改变 ⇒ 已录制的 dispatch 失效（否则回退后仍重放旧录制）
    };
    const LayoutState base = snap();
    if (!resolveLayoutMinCut(alt))
    {
      restore(base);  // 不适配 → 保留 baseline
    }
    else if (std::getenv("INFVINO_LAYOUT_MINCUT_GATE") && profiling_ &&
             !input_name_.empty() && inputNumel() > 0)
    {
      std::vector<uint16_t> zeros(inputNumel(), 0);
      setInput(zeros.data());
      auto busyMedian = [&](int reps) -> double {
        invalidateCapture();
        std::vector<double> v;
        try
        {
          for (int r = 0; r <= reps; ++r)
          {
            clearProfile();
            run();
            if (r == 0) continue;  // capture/热身
            const double b = profile().busy_ms;
            if (b > 0.0) v.push_back(b);
          }
        }
        catch (const std::exception &) { return 1e300; }
        if (v.empty()) return 1e300;
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
      };
      const LayoutState cut = snap();
      std::vector<double> bmed, mmed;
      for (int r = 0; r < 3; ++r)
      {
        restore(base);
        bmed.push_back(busyMedian(3));
        restore(cut);
        mmed.push_back(busyMedian(3));
      }
      auto medOf = [](std::vector<double> & v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
      const double bm = medOf(bmed), mm = medOf(mmed);
      if (!(mm < bm * 0.99))
      {
        restore(base);
        std::fprintf(stderr, "[layout] mincut gate REJECT: median %.4f -> %.4f ms\n", bm, mm);
      }
      else
      {
        std::fprintf(stderr, "[layout] mincut gate ACCEPT: median %.4f -> %.4f ms\n", bm, mm);
      }
    }
  }
}

bool PlanModel::resolveLayoutMinCut(const std::vector<LayoutAlt> & alt)
{
  // ---- 注册表驱动：族 → 布局契约（inIndex/canOutFsv16/in）。----
  auto blkFamilyOf = [&](size_t ni) -> const KernelFamily * {
    return familyByName(alt[ni].blk.kernel);
  };
  auto actSlot = [&](size_t ni) -> int {
    const KernelFamily * f = blkFamilyOf(ni);
    return f ? f->layout.inIndex : 1;
  };
  auto kernelCanOutFsv16 = [&](const std::string & kernel, const OpSignature & sig) -> bool {
    if (const KernelFamily * f = familyByName(kernel)) return f->layout.canOutFsv16;
    for (const auto & c : candidatesFromRegistry(sig))
      if (c.kernel == kernel) return c.canOutFsv16;
    return false;
  };
  auto kernelConsumesFsv16 = [&](const std::string & kernel) -> const KernelFamily * {
    const KernelFamily * f = familyByName(kernel);
    if (f && f->layout.in == Layout::FSV16) return f;
    // R51 D4+: gap 变体（gap/gap_r）在输入 fsv16 时用 -DGAP_IN_FSV16 直读。
    if (kernel == "gap" || kernel == "gap_r") return familyByName("gap_fsv16");
    return nullptr;
  };

  // ---- 参与 mincut 的节点：有 #blk/#non 备选、blk 族声明了布局契约。----
  // 步骤 3：从 conv1x1 扩展到 depthwise（任何声明了布局契约的族）。conv3x3_blk 结构性
  // 弱于 ov（R49 缺口 C），暂不纳入；其张量由 producer/consumer 能力门自动 pin。
  std::vector<size_t> nodes;
  for (size_t ni = 0; ni < nodes_.size(); ++ni)
  {
    if (!alt[ni].has || nodes_[ni].outs.empty()) continue;
    if (nodes_[ni].op == "conv3x3") continue;
    const int slot = actSlot(ni);
    if (slot < 0 || static_cast<size_t>(slot) >= nodes_[ni].ins.size()) continue;
    if (!blkFamilyOf(ni) || !blkFamilyOf(ni)->layout.canOutFsv16) continue;
    nodes.push_back(ni);
  }
  if (nodes.empty()) return false;
  std::unordered_set<int> inMincut(nodes.begin(), nodes.end());


  // ---- 张量 → 生产者/消费者索引。----
  std::unordered_map<std::string, int> producerOf;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (const auto & o : nodes_[i].outs) producerOf[o] = static_cast<int>(i);
  std::unordered_map<std::string, std::vector<std::pair<int, int>>> consumersOf;
  for (size_t i = 0; i < nodes_.size(); ++i)
    for (size_t s = 0; s < nodes_[i].ins.size(); ++s)
      if (nodes_[i].ins[s] != "-")
        consumersOf[nodes_[i].ins[s]].push_back({static_cast<int>(i), static_cast<int>(s)});

  auto selectedKernel = [&](size_t ni, OpSignature * sigOut) -> std::string {
    bool ok = false;
    const OpSignature s = nodeSignature(nodes_[ni], &ok);
    if (sigOut) *sigOut = s;
    if (!ok) return "";
    const TuningEntry * e = choiceEntry(ni, s);
    return (e && !e->kernel.empty()) ? e->kernel : "";
  };
  auto producerCanFsv16 = [&](const std::string & t) -> bool {
    auto it = producerOf.find(t);
    if (it == producerOf.end()) return false;  // 网络输入
    const size_t pi = static_cast<size_t>(it->second);
    if (inMincut.count(static_cast<int>(pi)))
      return blkFamilyOf(pi) && blkFamilyOf(pi)->layout.canOutFsv16;
    OpSignature s;
    const std::string k = selectedKernel(pi, &s);
    return !k.empty() && kernelCanOutFsv16(k, s);
  };
  auto consumerCanReadFsv16 = [&](const std::string & t) -> bool {
    auto it = consumersOf.find(t);
    if (it == consumersOf.end()) return true;
    for (const auto & cs : it->second)
    {
      const size_t ci = static_cast<size_t>(cs.first);
      const int slot = cs.second;
      if (inMincut.count(static_cast<int>(ci)))
      {
        if (slot != actSlot(ci)) return false;  // 残差/辅助槽不能是 fsv16
        continue;
      }
      OpSignature s;
      const std::string k = selectedKernel(ci, &s);
      const KernelFamily * cf = k.empty() ? nullptr : kernelConsumesFsv16(k);
      if (!cf || cf->layout.inIndex != slot) return false;
    }
    return true;
  };

  // ---- 变量：本族节点的激活输入/输出张量（同一张量只一个变量）。----
  std::unordered_map<std::string, int> var;
  std::vector<std::string>             varName;
  auto varId = [&](const std::string & name) {
    auto it = var.find(name);
    if (it != var.end()) return it->second;
    const int id = static_cast<int>(varName.size());
    var[name] = id;
    varName.push_back(name);
    return id;
  };
  for (size_t ni : nodes)
  {
    varId(nodes_[ni].ins[static_cast<size_t>(actSlot(ni))]);
    varId(nodes_[ni].outs[0]);
  }

  const std::unordered_set<std::string> out_set(outputs_.begin(), outputs_.end());
  BinaryEnergy en(static_cast<int>(varName.size()));

  // pin：网络输入/输出、生产者无法产 FSV16、消费者无法读 FSV16 → 钉死 NCHW。
  // R51: 移除了 R50 加的「通道非 16 对齐 → pin」——分配池现在按 paddedChannelNumel
  // 补齐通道，非对齐张量也能安全持久 fsv16。producerCanFsv16/consumerCanReadFsv16
  // 仍会 pin 掉「生产者不能直写 / 消费者不能直读」的张量。
  for (size_t v = 0; v < varName.size(); ++v)
  {
    const std::string & t = varName[v];
    const bool pin = out_set.count(t) > 0 || !producerCanFsv16(t) || !consumerCanReadFsv16(t);
    if (pin) en.fix(static_cast<int>(v), 0);
  }

  // ---- 节点代价表 f(x_in, x_out)（含 reorder + 可选 R47 L3/占用会计，submodular）。----
  // R49 缺口 1：把 R47 的**占用压力会计**接进节点代价（线性、可加）。隔离 ms 之外再计一项
  // `occupancyPressure × kL3SpillPerByteMs`。
  //
  // ⚠️ 负结果（见 docs/round49 §10）：逐节点 L3 项**破坏了可加性/量级校准**——占用压力是
  // 「总并发足迹」而非「miss 字节」，直接当节点成本会压过隔离 ms、把选择推向过度持久化
  // （mb 提案 1.57→2.46 ms）。R47 的正确形态是**整网 LRU 模拟**（顺序相关），不能塞进成对
  // min-cut 的节点项。因此本项**默认关**，仅作消融/实验旋钮（`INFVINO_LAYOUT_L3=1`）。
  const bool useL3 = std::getenv("INFVINO_LAYOUT_L3") != nullptr;
  auto inCurveMs = [&](const TuningEntry & e, const OpSignature & s) -> double {
    return e.ms + (useL3 ? occupancyPressure(e, s) * kL3SpillPerByteMs : 0.0);
  };
  //   f(0,0) = min(eN, eB+r)   NCHW 入/出：non，或 blk 付输入 reorder 写 NCHW
  //   f(0,1) = eB + r          NCHW 入 / FSV16 出：必须 blk，付输入 reorder
  //   f(1,0) = eB              FSV16 入 / NCHW 出：blk
  //   f(1,1) = eB              FSV16 入 / FSV16 出：blk
  //   K = (f00+f11-f01-f10)/2 = (min(eN,eB+r) - eB - r)/2 ≤ 0 → submodular。
  for (size_t ni : nodes)
  {
    const Node & n = nodes_[ni];
    const int a = varId(n.ins[static_cast<size_t>(actSlot(ni))]);
    const int b = varId(n.outs[0]);
    bool ok = false;
    const OpSignature s = nodeSignature(n, &ok);
    const double eB = inCurveMs(alt[ni].blk, s), eN = inCurveMs(alt[ni].non, s),
                 r = alt[ni].reorder.ms;
    // R50: 输出 fsv16 时用契约成本（同一 blk kernel + OUT_FSV16=1），否则用 bfyx 成本。
    const double eBf = alt[ni].blkFsv16.kernel.empty() ? eB : inCurveMs(alt[ni].blkFsv16, s);
    const double f00 = std::min(eN, eB + r), f01 = eBf + r, f10 = eB, f11 = eBf;
    if (!en.addPairwiseTable(a, b, f00, f01, f10, f11))
      return false;  // 非 submodular → 回退
  }

  const BinarySolution sol = solveBinaryMinCut(en);
  if (!sol.optimal) return false;

  // ---- 落地：张量 fsv16 + 每节点 kernel（与代价表的 argmin 一致）。----
  for (size_t v = 0; v < varName.size(); ++v)
  {
    auto it = T_.find(varName[v]);
    if (it != T_.end()) it->second.fsv16 = (sol.labels[v] == 1);
  }
  for (size_t ni : nodes)
  {
    const Node & n = nodes_[ni];
    const int a = varId(n.ins[static_cast<size_t>(actSlot(ni))]);
    const int b = varId(n.outs[0]);
    const bool la = (sol.labels[static_cast<size_t>(a)] == 1);
    const bool lb = (sol.labels[static_cast<size_t>(b)] == 1);
    bool ok = false;
    const OpSignature s = nodeSignature(n, &ok);
    const double eB = inCurveMs(alt[ni].blk, s), eN = inCurveMs(alt[ni].non, s),
                 r = alt[ni].reorder.ms;
    const bool useBlk = la || lb || (eB + r <= eN);
    node_choice_[ni] = useBlk ? alt[ni].blk : alt[ni].non;
  }
  mincut_active_ = true;
  if (std::getenv("INFVINO_LAYOUT_REPORT"))
  {
    int nfsv = 0;
    for (size_t v = 0; v < varName.size(); ++v) if (sol.labels[v] == 1) ++nfsv;
    std::fprintf(stderr, "[layout] mincut: %zu nodes, %zu vars, %d fsv16, E=%.4f\n",
                 nodes.size(), varName.size(), nfsv, sol.energy);
    if (std::getenv("INFVINO_LAYOUT_DEBUG"))
      for (size_t ni : nodes)
      {
        const OpSignature s = nodeSignature(nodes_[ni], nullptr);
        std::fprintf(stderr, "  [mincut] %-40s blk=%.4f non=%.4f r=%.4f -> %s\n",
                     nodes_[ni].outs[0].c_str(), alt[ni].blk.ms, alt[ni].non.ms, alt[ni].reorder.ms,
                     node_choice_[ni].kernel.c_str());
        std::fprintf(stderr, "           sig=%s\n", s.str().c_str());
      }
  }
  return true;
}

void PlanModel::buildKernels()
{
  kGemm_    = rt_.buildKernel("gemm", "gemm_f16", Tiles().options());
  kConvG_   = rt_.buildKernel("conv_general", "conv_general");
  kBin_     = rt_.buildKernel("ops", "ew_binary");
  kBinB_    = rt_.buildKernel("ops", "ew_binary_bcast");
  kUn_      = rt_.buildKernel("ops", "ew_unary");
  kCopy_    = rt_.buildKernel("ops", "copy_c");
  kSlice_   = rt_.buildKernel("ops", "slice_axis");
  kConcat_  = rt_.buildKernel("ops", "concat4");
  kPool_    = rt_.buildKernel("ops", "maxpool");
  kResize_  = rt_.buildKernel("ops", "resize_nn");
  kSoftmax_ = rt_.buildKernel("ops", "softmax_axis");
  kPerm_    = rt_.buildKernel("ops", "permute_0213");
  kGap_     = rt_.buildKernel("ops", "gap");
  kBias_    = rt_.buildKernel("ops", "bias_add");
  kBmm_     = rt_.buildKernel("ops", "bmm");
}

void PlanModel::releaseKernels()
{
  cl_kernel * ks[] = {
    &kGemm_, &kConvG_, &kBin_, &kBinB_, &kUn_, &kCopy_, &kSlice_, &kConcat_,
    &kPool_, &kResize_, &kSoftmax_, &kPerm_, &kGap_, &kBias_, &kBmm_};
  for (cl_kernel * k : ks)
    if (*k) { clReleaseKernel(*k); *k = nullptr; }
  for (auto & kv : kcache_)
    if (kv.second) clReleaseKernel(kv.second);
  kcache_.clear();
}

cl_kernel PlanModel::getKernel(
  const std::string & src, const std::string & name, const std::string & opts)
{
  const std::string key = src + "|" + name + "|" + opts;
  auto it = kcache_.find(key);
  if (it != kcache_.end()) return it->second;
  cl_kernel k = rt_.buildKernel(src, name, opts);
  kcache_[key] = k;
  return k;
}

// ---------------------------------------------------------------------------
// P2: per-node dispatch cache (see PlanModel.hpp). 语义：首帧录制、之后重放，
// 数值与首帧逐位一致；只消除 host 侧重复决策/参数设置。
// ---------------------------------------------------------------------------
cl_kernel PlanModel::cloneKernel(cl_kernel src)
{
  cl_program prog = nullptr;
  clGetKernelInfo(src, CL_KERNEL_PROGRAM, sizeof(prog), &prog, nullptr);
  size_t sz = 0;
  clGetKernelInfo(src, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &sz);
  std::vector<char> name(sz + 1, '\0');
  clGetKernelInfo(src, CL_KERNEL_FUNCTION_NAME, sz, name.data(), nullptr);
  cl_int err = CL_SUCCESS;
  cl_kernel k = clCreateKernel(prog, name.data(), &err);
  if (err != CL_SUCCESS) throw std::runtime_error("PlanModel: cloneKernel failed");
  clone_kernels_.push_back(k);
  return k;
}

void PlanModel::setArg(cl_kernel k, cl_uint index, size_t size, const void * value)
{
  if (capturing_ && value != nullptr)
  {
    PlanArg a;
    a.index = index;
    a.size  = size;
    const auto * p = static_cast<const unsigned char *>(value);
    a.bytes.assign(p, p + size);
    cap_args_[k].push_back(std::move(a));
  }
  clSetKernelArg(k, index, size, value);
}

cl_event PlanModel::enqueueCmd(cl_kernel k, cl_uint dim, const size_t * gws, const size_t * lws,
                               const char * tag, bool useLws)
{
  if (capturing_)
  {
    PlanCmd c;
    c.k   = cloneKernel(k);
    c.dim = dim;
    for (cl_uint i = 0; i < dim && i < 3; ++i) c.gws[i] = gws[i];
    c.useLws = useLws;
    if (lws)
      for (cl_uint i = 0; i < dim && i < 3; ++i) c.lws[i] = lws[i];
    c.tag = tag ? tag : "";
    // 把本节点为该 kernel 记录下的参数一次性设到克隆上；此后每帧不再 setArg。
    auto it = cap_args_.find(k);
    if (it != cap_args_.end())
    {
      for (const PlanArg & a : it->second)
        setArg(c.k, a.index, a.size, a.bytes.data());
      cap_args_.erase(it);
    }
    cap_cmds_.push_back(std::move(c));
  }
  return ClRuntime::enqueueND(rt_.queue(), k, dim, gws, lws);
}

void PlanModel::replayNode(size_t ni)
{
  for (const PlanCmd & c : node_cmds_[ni])
  {
    cl_event ev = nullptr;
    if (profiling_)
    {
      const auto enq0 = std::chrono::steady_clock::now();
      ev = ClRuntime::enqueueND(rt_.queue(), c.k, c.dim, c.gws, c.useLws ? c.lws : nullptr);
      prof_enqueue_ms_ += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - enq0).count();
      const auto w0 = std::chrono::steady_clock::now();
      clWaitForEvents(1, &ev);
      prof_wait_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - w0).count();
      cl_ulong s = 0, e = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
      tprof_[c.tag].first += static_cast<double>(e - s) * 1e-6;
      tprof_[c.tag].second += 1;
      noteNode(ni, c.tag, static_cast<double>(e - s) * 1e-6);
    }
    else
    {
      ev = ClRuntime::enqueueND(rt_.queue(), c.k, c.dim, c.gws, c.useLws ? c.lws : nullptr);
    }
    if (ev) clReleaseEvent(ev);
  }
}

void PlanModel::noteNode(size_t ni, const std::string & tag, double ms)
{
  if (ni >= node_ms_.size()) return;
  node_ms_[ni] += ms;
  node_calls_[ni] += 1;
  // tag 取该节点的主要算子：reorder(blk) 是该节点的附属 dispatch，不要占主标签。
  if (node_tag_[ni].empty() || (node_tag_[ni] == "reorder(blk)" && tag != "reorder(blk)"))
    node_tag_[ni] = tag;
}

void PlanModel::invalidateCapture()
{
  captured_ = false;
  node_cmds_.clear();
  if (cmdbuf_ && cmdbuf_release_)
  {
    reinterpret_cast<clReleaseCommandBufferKHR_fn>(cmdbuf_release_)(
      static_cast<cl_command_buffer_khr>(cmdbuf_));
    cmdbuf_ = nullptr;
  }
}

PlanModel::PlanProfile PlanModel::profile() const
{
  PlanProfile p;
  p.plan_nodes = static_cast<int>(nodes_.size());
  for (char s : node_skipped_)
    if (s) p.skipped_nodes++;
  p.dispatched_nodes = p.plan_nodes - p.skipped_nodes;
  for (const auto & kv : tprof_) p.dispatches += kv.second.second;
  auto rit = tprof_.find("reorder(blk)");
  if (rit != tprof_.end())
  {
    p.reorder_calls = rit->second.second;
    p.reorder_ms    = rit->second.first;
  }
  p.fusions_res    = fusions_res_;
  p.fusions_concat = fusions_concat_;
  for (const auto & kv : T_)
    if (kv.second.fsv16) p.fsv16_tensors++;
  p.pool_requested = act_pool_.requestedBytes();
  p.pool_allocated = act_pool_.allocatedBytes();
  p.pool_buffers   = act_pool_.bufferCount();
  p.wall_ms        = last_run_ms_;
  p.enqueue_ms     = prof_enqueue_ms_;
  p.sync_ms        = prof_wait_ms_;
  p.busy_ms        = 0.0;
  for (const auto & kv : tprof_) p.busy_ms += kv.second.first;
  p.nodes.reserve(node_ms_.size());
  for (size_t i = 0; i < node_ms_.size(); ++i)
  {
    if (node_ms_[i] == 0.0 && node_calls_[i] == 0) continue;
    PlanProfile::NodeRow r;
    r.index = static_cast<int>(i);
    r.op    = i < nodes_.size() ? nodes_[i].op : std::string();
    r.tag   = node_tag_[i];
    if (i < nodes_.size())
    {
      try
      {
        bool ok = true;
        const OpSignature s = nodeSignature(nodes_[i], &ok);
        if (ok) r.signature = s.str();
      }
      catch (const std::exception &) {}  // 诊断字段，绝不因签名失败影响主流程
    }
    r.ms    = node_ms_[i];
    r.calls = node_calls_[i];
    p.nodes.push_back(std::move(r));
  }
  return p;
}

// ---------------------------------------------------------------------------
// R32 原型：cl_khr_command_buffer 整帧重放（CUDA-graph 类比）。
//
// 在首帧 launch 缓存录制完成后调用：把所有节点的 dispatch（含 blk 重排等）录制进
// 一个 command buffer，之后每帧只用一次 clEnqueueCommandBufferKHR 提交。
// 目的：量「每 dispatch host 提交」与「kernel 间前端气泡」到底占 net 多少。
// 驱动不导出符号，用 clGetExtensionFunctionAddressForPlatform 解析。
// ---------------------------------------------------------------------------
void PlanModel::buildCommandBuffer()
{
  if (!cmdbuf_enabled_ || cmdbuf_failed_ || cmdbuf_) return;
  cl_platform_id plat = nullptr;
  if (clGetDeviceInfo(rt_.device(), CL_DEVICE_PLATFORM, sizeof(plat), &plat, nullptr) != CL_SUCCESS ||
      !plat) {
    cmdbuf_failed_ = true;
    return;
  }
  auto pCreate  = reinterpret_cast<clCreateCommandBufferKHR_fn>(
    clGetExtensionFunctionAddressForPlatform(plat, "clCreateCommandBufferKHR"));
  auto pNDR     = reinterpret_cast<clCommandNDRangeKernelKHR_fn>(
    clGetExtensionFunctionAddressForPlatform(plat, "clCommandNDRangeKernelKHR"));
  auto pFinal   = reinterpret_cast<clFinalizeCommandBufferKHR_fn>(
    clGetExtensionFunctionAddressForPlatform(plat, "clFinalizeCommandBufferKHR"));
  auto pRelease = reinterpret_cast<clReleaseCommandBufferKHR_fn>(
    clGetExtensionFunctionAddressForPlatform(plat, "clReleaseCommandBufferKHR"));
  auto pEnqueue = reinterpret_cast<clEnqueueCommandBufferKHR_fn>(
    clGetExtensionFunctionAddressForPlatform(plat, "clEnqueueCommandBufferKHR"));
  if (!pCreate || !pNDR || !pFinal || !pRelease || !pEnqueue) {
    std::fprintf(stderr, "[cmdbuf] cl_khr_command_buffer not resolvable; falling back to replay\n");
    cmdbuf_failed_ = true;
    return;
  }
  cl_command_queue q = rt_.queue();
  cl_int err = CL_SUCCESS;
  cl_command_buffer_khr cb = pCreate(1, &q, nullptr, &err);
  if (!cb || err != CL_SUCCESS) {
    std::fprintf(stderr, "[cmdbuf] create failed (%d); falling back\n", (int)err);
    cmdbuf_failed_ = true;
    return;
  }
  int nrec = 0;
  for (size_t ni = 0; ni < node_cmds_.size(); ++ni) {
    for (const PlanCmd & c : node_cmds_[ni]) {
      cl_int e = pNDR(cb, q, nullptr, c.k, c.dim, nullptr, c.gws,
                      c.useLws ? c.lws : nullptr, 0, nullptr, nullptr, nullptr);
      if (e != CL_SUCCESS) {
        std::fprintf(stderr, "[cmdbuf] record failed at node %zu (%d)\n", ni, (int)e);
        pRelease(cb);
        cmdbuf_failed_ = true;
        return;
      }
      ++nrec;
    }
  }
  if (pFinal(cb) != CL_SUCCESS) {
    std::fprintf(stderr, "[cmdbuf] finalize failed; falling back\n");
    pRelease(cb);
    cmdbuf_failed_ = true;
    return;
  }
  cmdbuf_ = cb;
  cmdbuf_enqueue_ = reinterpret_cast<void *>(pEnqueue);
  cmdbuf_release_ = reinterpret_cast<void *>(pRelease);
  std::fprintf(stderr, "[cmdbuf] recorded %d dispatches into one command buffer\n", nrec);
}

cl_mem PlanModel::bcastDims(const std::string & spec)
{
  auto it = small_buf_.find(spec);
  if (it != small_buf_.end()) return it->second;
  std::vector<int> all;
  std::stringstream ss(spec);
  std::string tk;
  while (std::getline(ss, tk, ',')) all.push_back(std::atoi(tk.c_str()));
  cl_mem m = rt_.alloc(all.size() * 4, CL_MEM_READ_ONLY);
  rt_.write(m, all.size() * 4, all.data());
  small_buf_[spec] = m;
  owned_.push_back(m);   // 生命周期与模型一致（比每次 run 分配/释放更省）
  return m;
}

OpSignature PlanModel::smallSig(const Node & n) const
{
  const int64_t nout = T_.at(n.outs[0]).numel();
  if (n.op == "ew_binary")
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0),
              bs = attrInt(n, "b_scalar", 0);
    // R30: 若 b 是**真标量**（b_scalar=1），即使 plan 里带了 bdims，也应走标量核，
    // 而不是通用 rank 核（后者每元素 4×整数 div/mod）。数值等价：a 与 out 同形且连续，
    // aidx==i、bidx==0。
    if (n.attr.count("bdims") && bs != 1)
    {
      int C = 0;
      const Tensor & ta = T_.at(n.ins[0]);
      const Tensor & tb = T_.at(n.ins[1]);
      const Tensor * ts = nullptr;
      if (ta.numel() != nout && tb.numel() == nout) ts = &ta;
      else if (tb.numel() != nout && ta.numel() == nout) ts = &tb;
      if (ts && ts->numel() > 0)
      {
        int nonunit = 0, dimpos = -1;
        for (size_t i = 0; i < ts->dims.size(); ++i)
          if (ts->dims[i] != 1) { ++nonunit; dimpos = static_cast<int>(i); }
        const Tensor & tl = (ts == &ta) ? tb : ta;
        const auto &   ld = tl.dims;
        const int      chidx = ld.size() >= 3 ? static_cast<int>(ld.size()) - 3 : 0;
        if (nonunit == 1 && dimpos == chidx && ld[chidx] == ts->numel())
          C = static_cast<int>(ts->numel());
      }
      // R30: 有效秩（去掉 extent==1 的维）= 是否能用 3-D 网格 bcast 核（≤4）的判据。
      int effRank = 0;
      {
        std::vector<int> vals;
        std::stringstream ss(n.attr.at("bdims"));
        std::string tk;
        while (std::getline(ss, tk, ',')) vals.push_back(std::atoi(tk.c_str()));
        const int rank = vals.empty() ? 0 : static_cast<int>(vals.size() / 3);
        for (int r = 0; r < rank; ++r)
          if (vals[r] != 1) ++effRank;
      }
      return OpSignature::custom("ew_binary_bcast", {nn, op, bs, C, effRank});
    }
    return OpSignature::custom("ew_binary", {nn, op, bs});
  }
  if (n.op == "ew_unary")
    return OpSignature::custom("ew_unary", {static_cast<int>(nout), attrInt(n, "op", 0)});
  if (n.op == "copy_c")
    return OpSignature::custom("copy_c", {attrInt(n, "HW", 1), attrInt(n, "cnt", 0)});
  if (n.op == "slice_axis")
    return OpSignature::custom("slice_axis", {attrInt(n, "outer", 1), attrInt(n, "axdim", 0),
                                              attrInt(n, "inner", 1), attrInt(n, "start", 0),
                                              attrInt(n, "len", 0)});
  if (n.op == "concat4")
    return OpSignature::custom("concat4", {attrInt(n, "outer", 1), attrInt(n, "inner", 1),
                                           attrInt(n, "ca", 0), attrInt(n, "cb", 0),
                                           attrInt(n, "cc", 0), attrInt(n, "cd", 0)});
  if (n.op == "maxpool")
    return OpSignature::custom("maxpool", {attrInt(n, "C", 0), attrInt(n, "H", 0),
                                           attrInt(n, "W", 0), attrInt(n, "Hout", 0),
                                           attrInt(n, "Wout", 0), attrInt(n, "K", 5),
                                           attrInt(n, "S", 1), attrInt(n, "P", 2)});
  if (n.op == "resize_nn")
    return OpSignature::custom("resize_nn", {attrInt(n, "C", 0), attrInt(n, "H", 0),
                                             attrInt(n, "W", 0), attrInt(n, "S", 2)});
  if (n.op == "permute_0213")
    return OpSignature::custom("permute_0213", {attrInt(n, "D1", 1), attrInt(n, "D2", 1),
                                                attrInt(n, "I", 1), attrInt(n, "mode", 0)});
  if (n.op == "softmax_axis")
    return OpSignature::custom("softmax_axis", {attrInt(n, "outer", 1), attrInt(n, "axdim", 0),
                                                attrInt(n, "inner", 1)});
  if (n.op == "bmm")
  {
    // dims: A=[B0,B1,M,K], B=[B0,B1,K,N]（这里的首两维可能是 1，需按张量自身形状取）。
    const auto & ad = T_.at(n.ins[0]).dims;
    const auto & bd = T_.at(n.ins[1]).dims;
    auto at = [](const std::vector<int64_t> & d, size_t i) -> int {
      return i < d.size() ? static_cast<int>(d[i]) : 1;
    };
    return OpSignature::custom("bmm", {at(ad, 0), at(ad, 1), at(ad, 2),
                                       at(ad, ad.size() - 1), at(bd, bd.size() - 1)});
  }
  if (n.op == "gap")
    return OpSignature::gap(attrInt(n, "C", 0), attrInt(n, "HW", 1));
  return OpSignature::custom(n.op, {});
}

void PlanModel::smallLaunch(const Node & n, cl_kernel k, const std::string & kernel,
                            const std::string & opts, cl_uint & dim, size_t * gws,
                            size_t * lws, bool & useLws)
{
  dim = 1;
  useLws = false;
  auto inMem = [&](size_t i) -> cl_mem {
    return (i < n.ins.size() && n.ins[i] != "-" && T_.count(n.ins[i])) ? ref(n.ins[i]).mem
                                                                      : nullptr;
  };
  const int64_t nout = ref(n.outs[0]).numel();
  cl_mem dy = ref(n.outs[0]).mem;
  auto optInt = [&](const char * key, int def) {
    const auto p = opts.find(key);
    return p == std::string::npos ? def : std::atoi(opts.c_str() + p + std::strlen(key));
  };
  const bool isVec = kernel.find("_v") != std::string::npos;
  auto vecN = [&](int64_t count) -> size_t {
    const int v = std::max(1, optInt("-DEW_VEC=", 4));
    return static_cast<size_t>((count + v - 1) / v);
  };

  if (n.op == "ew_binary" && (!n.attr.count("bdims") || attrInt(n, "b_scalar", 0) == 1))
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
    cl_mem da = inMem(0), db = inMem(1);
    setArg(k, 0, sizeof(da), &da);
    setArg(k, 1, sizeof(db), &db);
    setArg(k, 2, sizeof(dy), &dy);
    setArg(k, 3, sizeof(nn), &nn);
    setArg(k, 4, sizeof(op), &op);
    const int bs = attrInt(n, "b_scalar", 0);
    setArg(k, 5, sizeof(bs), &bs);
    gws[0] = isVec ? vecN(nn) : static_cast<size_t>(nn);
    return;
  }
  if (n.op == "ew_binary_bcast" || (n.op == "ew_binary" && n.attr.count("bdims")))
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
    cl_mem da = inMem(0), db = inMem(1);
    if (kernel == "ew_binary_ch")
    {
      const Tensor & ta = ref(n.ins[0]);
      const Tensor & tb = ref(n.ins[1]);
      const Tensor * ts = nullptr;
      int a_ch = 0;
      if (ta.numel() != nout && tb.numel() == nout) { ts = &ta; a_ch = 1; }
      else if (tb.numel() != nout && ta.numel() == nout) { ts = &tb; a_ch = 0; }
      const int C  = (ts && ts->numel() > 0) ? static_cast<int>(ts->numel()) : 1;
      const int HW = C > 0 ? static_cast<int>(nout / C) : static_cast<int>(nout);
      setArg(k, 0, sizeof(da), &da);
      setArg(k, 1, sizeof(db), &db);
      setArg(k, 2, sizeof(dy), &dy);
      setArg(k, 3, sizeof(HW), &HW);
      setArg(k, 4, sizeof(C), &C);
      setArg(k, 5, sizeof(op), &op);
      setArg(k, 6, sizeof(a_ch), &a_ch);
      dim = 2;
      gws[0] = static_cast<size_t>(HW);
      gws[1] = static_cast<size_t>(C);
      return;
    }
    auto bit = n.attr.find("bdims");
    const std::string spec = (bit != n.attr.end()) ? bit->second : std::string();
    int rank = 0;
    if (!spec.empty())
      rank = static_cast<int>((std::count(spec.begin(), spec.end(), ',') + 1) / 3);
    if (kernel == "ew_binary_bcast4")
    {
      // R30: 3-D 网格 + 运行期 stride，去掉通用 rank 核的逐元素 div/mod。
      // 有效维按「最内层在前」收集（p0 最连续）；最多 4 维，第 3/4 维折叠进 gid2。
      std::vector<int> vals;
      {
        std::stringstream ss(spec);
        std::string tk;
        while (std::getline(ss, tk, ',')) vals.push_back(std::atoi(tk.c_str()));
      }
      const int rk = vals.empty() ? 0 : static_cast<int>(vals.size() / 3);
      // 有效维 = 跳过 extent-1 的维（它们坐标恒为 0，对右对齐布局无影响），按
      // 「最内层在前」收集，最多 4 维。每个有效维保留它自己的 stride——不同有效维
      // 的 stride 恰好右对齐，所以 g0..g3 直接点乘即可。>4 个有效维时回退通用核。
      int p[4] = {1, 1, 1, 1}, as[4] = {0, 0, 0, 0}, bs[4] = {0, 0, 0, 0}, kk = 0;
      bool safe = true;
      for (int r = rk - 1; r >= 0; --r)
      {
        if (vals[r] == 1) continue;
        if (kk < 4)
        {
          p[kk]  = vals[r];
          as[kk] = vals[rk + r];
          bs[kk] = vals[2 * rk + r];
          ++kk;
        }
        else { safe = false; break; }
      }
      if (!safe)
      {
        // 中部有 extent-1 维（或有效秩 >4）：回退通用 rank 核。它用 rank 参数，
        // 与 bcast4 的 19 个参数不同，必须重新 build 通用 kernel 句柄。
        k = getKernel("ops", "ew_binary_bcast", "");
        cl_mem dm = bcastDims(spec);
        setArg(k, 0, sizeof(da), &da);
        setArg(k, 1, sizeof(db), &db);
        setArg(k, 2, sizeof(dy), &dy);
        setArg(k, 3, sizeof(nn), &nn);
        setArg(k, 4, sizeof(op), &op);
        const int rr = rk;
        setArg(k, 5, sizeof(rr), &rr);
        setArg(k, 6, sizeof(dm), &dm);
        dim = 1;
        gws[0] = static_cast<size_t>(nn);
        return;
      }
      const int d0 = p[0], d1 = p[1], d2 = p[2], d3 = p[3];
      const int os1 = d0, os2 = d0 * d1, os3 = d0 * d1 * d2;
      setArg(k, 0, sizeof(da), &da);
      setArg(k, 1, sizeof(db), &db);
      setArg(k, 2, sizeof(dy), &dy);
      setArg(k, 3, sizeof(op), &op);
      setArg(k, 4, sizeof(d0), &d0);
      setArg(k, 5, sizeof(d1), &d1);
      setArg(k, 6, sizeof(d2), &d2);
      setArg(k, 7, sizeof(d3), &d3);
      setArg(k, 8, sizeof(os1), &os1);
      setArg(k, 9, sizeof(os2), &os2);
      setArg(k, 10, sizeof(os3), &os3);
      setArg(k, 11, sizeof(as[0]), &as[0]);
      setArg(k, 12, sizeof(as[1]), &as[1]);
      setArg(k, 13, sizeof(as[2]), &as[2]);
      setArg(k, 14, sizeof(as[3]), &as[3]);
      setArg(k, 15, sizeof(bs[0]), &bs[0]);
      setArg(k, 16, sizeof(bs[1]), &bs[1]);
      setArg(k, 17, sizeof(bs[2]), &bs[2]);
      setArg(k, 18, sizeof(bs[3]), &bs[3]);
      dim = 3;
      gws[0] = static_cast<size_t>(d0);
      gws[1] = static_cast<size_t>(d1);
      gws[2] = static_cast<size_t>(d2) * static_cast<size_t>(d3);
      return;
    }
    cl_mem dm = bcastDims(spec);
    setArg(k, 0, sizeof(da), &da);
    setArg(k, 1, sizeof(db), &db);
    setArg(k, 2, sizeof(dy), &dy);
    setArg(k, 3, sizeof(nn), &nn);
    setArg(k, 4, sizeof(op), &op);
    setArg(k, 5, sizeof(rank), &rank);
    setArg(k, 6, sizeof(dm), &dm);
    gws[0] = static_cast<size_t>(nn);
    return;
  }
  if (n.op == "ew_unary")
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(nn), &nn);
    setArg(k, 3, sizeof(op), &op);
    gws[0] = isVec ? vecN(nn) : static_cast<size_t>(nn);
    return;
  }
  if (n.op == "copy_c")
  {
    int HW = attrInt(n, "HW", 1), c0 = attrInt(n, "c0", 0), cnt = attrInt(n, "cnt", 0),
        dst = attrInt(n, "dst_off", 0);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(HW), &HW);
    setArg(k, 3, sizeof(c0), &c0);
    setArg(k, 4, sizeof(cnt), &cnt);
    setArg(k, 5, sizeof(dst), &dst);
    if (kernel == "copy_c2")
    {
      dim = 2;
      gws[0] = static_cast<size_t>(HW);
      gws[1] = static_cast<size_t>(cnt);
    }
    else
    {
      gws[0] = static_cast<size_t>(cnt) * static_cast<size_t>(HW);
    }
    return;
  }
  if (n.op == "slice_axis")
  {
    int outer = attrInt(n, "outer", 1), axdim = attrInt(n, "axdim", 0),
        inner = attrInt(n, "inner", 1), start = attrInt(n, "start", 0),
        len = attrInt(n, "len", 0);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(outer), &outer);
    setArg(k, 3, sizeof(axdim), &axdim);
    setArg(k, 4, sizeof(inner), &inner);
    setArg(k, 5, sizeof(start), &start);
    setArg(k, 6, sizeof(len), &len);
    if (kernel == "slice_axis3")
    {
      dim = 3;
      gws[0] = static_cast<size_t>(inner);
      gws[1] = static_cast<size_t>(len);
      gws[2] = static_cast<size_t>(outer);
    }
    else
    {
      gws[0] = static_cast<size_t>(outer) * static_cast<size_t>(len) *
               static_cast<size_t>(inner);
    }
    return;
  }
  if (n.op == "concat4")
  {
    int ca = attrInt(n, "ca", 0), cb = attrInt(n, "cb", 0), cc = attrInt(n, "cc", 0),
        cd = attrInt(n, "cd", 0);
    int outer = attrInt(n, "outer", 1), inner = attrInt(n, "inner", 1);
    cl_mem ia = ca ? inMem(0) : nullptr, ib = cb ? inMem(1) : nullptr,
           ic = cc ? inMem(2) : nullptr, id = cd ? inMem(3) : nullptr;
    setArg(k, 0, sizeof(ia), &ia);
    setArg(k, 1, sizeof(ca), &ca);
    setArg(k, 2, sizeof(ib), &ib);
    setArg(k, 3, sizeof(cb), &cb);
    setArg(k, 4, sizeof(ic), &ic);
    setArg(k, 5, sizeof(cc), &cc);
    setArg(k, 6, sizeof(id), &id);
    setArg(k, 7, sizeof(cd), &cd);
    setArg(k, 8, sizeof(dy), &dy);
    setArg(k, 9, sizeof(outer), &outer);
    setArg(k, 10, sizeof(inner), &inner);
    const int csum = ca + cb + cc + cd;
    dim = 3;
    gws[0] = isVec ? vecN(inner) : static_cast<size_t>(inner);
    gws[1] = static_cast<size_t>(csum);
    gws[2] = static_cast<size_t>(outer);
    return;
  }
  if (n.op == "maxpool")
  {
    int C = attrInt(n, "C", 0), H = attrInt(n, "H", 0), W = attrInt(n, "W", 0),
        ho = attrInt(n, "Hout", 0), wo = attrInt(n, "Wout", 0), K = attrInt(n, "K", 5),
        S = attrInt(n, "S", 1), P = attrInt(n, "P", 2);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(C), &C);
    setArg(k, 3, sizeof(H), &H);
    setArg(k, 4, sizeof(W), &W);
    setArg(k, 5, sizeof(ho), &ho);
    setArg(k, 6, sizeof(wo), &wo);
    setArg(k, 7, sizeof(K), &K);
    setArg(k, 8, sizeof(S), &S);
    setArg(k, 9, sizeof(P), &P);
    if (kernel == "maxpool3")
    {
      dim = 3;
      gws[0] = static_cast<size_t>(wo);
      gws[1] = static_cast<size_t>(ho);
      gws[2] = static_cast<size_t>(C);
    }
    else
    {
      gws[0] = static_cast<size_t>(C) * static_cast<size_t>(ho) * static_cast<size_t>(wo);
    }
    return;
  }
  if (n.op == "resize_nn")
  {
    int C = attrInt(n, "C", 0), H = attrInt(n, "H", 0), W = attrInt(n, "W", 0),
        S = attrInt(n, "S", 2);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(C), &C);
    setArg(k, 3, sizeof(H), &H);
    setArg(k, 4, sizeof(W), &W);
    setArg(k, 5, sizeof(S), &S);
    if (kernel == "resize_nn3")
    {
      dim = 3;
      gws[0] = static_cast<size_t>(W) * static_cast<size_t>(S);
      gws[1] = static_cast<size_t>(H) * static_cast<size_t>(S);
      gws[2] = static_cast<size_t>(C);
    }
    else
    {
      gws[0] = static_cast<size_t>(C) * static_cast<size_t>(H) * static_cast<size_t>(S) *
               static_cast<size_t>(W) * static_cast<size_t>(S);
    }
    return;
  }
  if (n.op == "permute_0213")
  {
    int D1 = attrInt(n, "D1", 1), D2 = attrInt(n, "D2", 1), I = attrInt(n, "I", 1),
        mode = attrInt(n, "mode", 0);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(D1), &D1);
    setArg(k, 3, sizeof(D2), &D2);
    setArg(k, 4, sizeof(I), &I);
    setArg(k, 5, sizeof(mode), &mode);
    if (kernel == "permute_0213_3d")
    {
      dim = 3;
      if (mode == 0)
      {
        gws[0] = static_cast<size_t>(I);
        gws[1] = static_cast<size_t>(D2);
      }
      else
      {
        gws[0] = static_cast<size_t>(D2);
        gws[1] = static_cast<size_t>(I);
      }
      gws[2] = static_cast<size_t>(D1);
    }
    else
    {
      gws[0] = static_cast<size_t>(D1) * static_cast<size_t>(D2) * static_cast<size_t>(I);
    }
    return;
  }
  if (n.op == "bmm")
  {
    const Tensor & A = ref(n.ins[0]);
    const Tensor & B2 = ref(n.ins[1]);
    const int B0 = static_cast<int>(A.dims[0]);
    const int B1 = static_cast<int>(A.dims[1]);
    const int M  = static_cast<int>(A.dims[2]);
    const int K  = static_cast<int>(A.dims[3]);
    const int N  = static_cast<int>(B2.dims[3]);
    cl_mem da = A.mem, db = B2.mem;
    setArg(k, 0, sizeof(da), &da);
    setArg(k, 1, sizeof(db), &db);
    setArg(k, 2, sizeof(dy), &dy);
    setArg(k, 3, sizeof(B0), &B0);
    setArg(k, 4, sizeof(B1), &B1);
    setArg(k, 5, sizeof(M), &M);
    setArg(k, 6, sizeof(K), &K);
    setArg(k, 7, sizeof(N), &N);
    if (kernel == "bmm2")
    {
      dim = 3;
      gws[0] = static_cast<size_t>(N);
      gws[1] = static_cast<size_t>(M);
      gws[2] = static_cast<size_t>(B0) * static_cast<size_t>(B1);
    }
    else if (kernel == "bmm_t")
    {
      const int TM = std::max(1, optInt("-DBMM_TM=", 4));
      const int TN = std::max(1, optInt("-DBMM_TN=", 8));
      dim = 3;
      gws[0] = static_cast<size_t>((N + TN - 1) / TN);
      gws[1] = static_cast<size_t>((M + TM - 1) / TM);
      gws[2] = static_cast<size_t>(B0) * static_cast<size_t>(B1);
    }
    else
    {
      gws[0] = static_cast<size_t>(B0) * static_cast<size_t>(B1) * static_cast<size_t>(M) *
               static_cast<size_t>(N);
    }
    return;
  }
  if (n.op == "softmax_axis")
  {
    int outer = attrInt(n, "outer", 1), axdim = attrInt(n, "axdim", 0),
        inner = attrInt(n, "inner", 1);
    cl_mem dx = inMem(0);
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(outer), &outer);
    setArg(k, 3, sizeof(axdim), &axdim);
    setArg(k, 4, sizeof(inner), &inner);
    if (kernel == "softmax_axis_r")
    {
      const int wgs = std::max(1, optInt("-DSM_WGS=", 128));
      useLws = true;
      dim = 2;
      lws[0] = static_cast<size_t>(wgs);
      lws[1] = 1;
      // one work-group per (outer, inner) row; the group's WGS lanes reduce the axis.
      gws[0] = static_cast<size_t>(outer) * static_cast<size_t>(wgs);
      gws[1] = static_cast<size_t>(inner);
    }
    else
    {
      gws[0] = static_cast<size_t>(outer) * static_cast<size_t>(inner);
    }
    return;
  }
  if (n.op == "gap")
  {
    int C = attrInt(n, "C", 0), HW = attrInt(n, "HW", 1);
    cl_mem dx = inMem(0);
    const int wgs = std::max(1, optInt("-DGAP_WGS=", 128));
    setArg(k, 0, sizeof(dx), &dx);
    setArg(k, 1, sizeof(dy), &dy);
    setArg(k, 2, sizeof(C), &C);
    setArg(k, 3, sizeof(HW), &HW);
    useLws = true;
    dim = 1;
    lws[0] = static_cast<size_t>(wgs);
    gws[0] = static_cast<size_t>(C) * static_cast<size_t>(wgs);
    return;
  }
  throw std::runtime_error("PlanModel: smallLaunch unsupported op: " + n.op);
}

namespace
{
// kernel 函数名 → .cl 源文件名。
std::string sourceOfKernel(const std::string & kernel)
{
  if (kernel == "conv3x3_ov") return "conv_ov";
  if (kernel == "conv3x3_cin3") return "conv_cin3";
  if (kernel == "conv3x3_f16" || kernel == "conv3x3_rt" || kernel == "conv3x3_db") return "conv";
  if (kernel == "conv3x3_sg") return "conv_sg";
  if (kernel == "conv3x3_osv") return "conv_osv";
  if (kernel == "conv3x3_blk" || kernel == "reorder_bfyx_to_fsv16") return "conv_blk";
  if (kernel == "gemm_f16") return "gemm";
  if (kernel == "gemm_sk_f16") return "gemm_sk";
  if (kernel == "conv1x1_gemv_f16" || kernel == "conv1x1_f16") return "conv1x1";
  if (kernel == "conv1x1_blk") return "conv1x1_blk";
  if (kernel == "depthwise_f16" || kernel == "depthwise_v" || kernel == "depthwise_vp" ||
      kernel == "depthwise_pad" || kernel == "conv_general") return "conv_general";
  if (kernel == "depthwise_blk") return "depthwise_blk";
  return "ops";
}

// R33: 把 options 里的 -DRES=<d> 改成期望值；没有则追加。融合残差后调用。
void setResOpt(std::string & opts, bool on)
{
  const std::string key = "-DRES=";
  const auto        p   = opts.find(key);
  const char *      val = on ? "1" : "0";
  if (p == std::string::npos) {
    opts += " -DRES=";
    opts += val;
    return;
  }
  const size_t v = p + key.size();
  if (v < opts.size() && opts[v] >= '0' && opts[v] <= '9') opts[v] = val[0];
}
}  // namespace

void PlanModel::run()
{
  const auto t0 = std::chrono::steady_clock::now();

  // R36：新的帧——清空「本帧已重排」集合（同帧多消费者共享一次 bfyx->fsv16）。
  reordered_frame_.clear();

  // R32 原型：command buffer 录制完成后，整帧一次提交。
  if (cmdbuf_)
  {
    cl_command_queue q = rt_.queue();
    cl_event          ev = nullptr;
    const auto        enq0 = std::chrono::steady_clock::now();
    cl_int err = reinterpret_cast<clEnqueueCommandBufferKHR_fn>(cmdbuf_enqueue_)(
      1, &q, static_cast<cl_command_buffer_khr>(cmdbuf_), 0, nullptr, &ev);
    if (err != CL_SUCCESS)
      throw std::runtime_error("PlanModel: clEnqueueCommandBufferKHR failed: " +
                               std::to_string(err));
    if (profiling_)
    {
      prof_enqueue_ms_ += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - enq0).count();
      const auto w0 = std::chrono::steady_clock::now();
      clWaitForEvents(1, &ev);
      prof_wait_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - w0).count();
      cl_ulong s = 0, e = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
      tprof_["cmdbuf"].first += static_cast<double>(e - s) * 1e-6;
      tprof_["cmdbuf"].second += 1;
    }
    if (ev) clReleaseEvent(ev);
    rt_.finish();
    last_run_ms_ =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return;
  }

  auto timed = [&](const std::string & tag, cl_kernel k, cl_uint dim, const size_t * gws,
                   const size_t * lws) {
    size_t total = 1;
    for (cl_uint i = 0; i < dim; ++i) total *= gws[i];
    // 安全阀：工作项数超上限直接报错，避免假死/打满内存。
    const size_t kMaxItems = 300u * 1000u * 1000u;
    if (total > kMaxItems)
      throw std::runtime_error(
        "PlanModel: gws too large for " + tag + ": " + std::to_string(total) +
        " items (check plan attrs)");
    cl_event ev = nullptr;
    const auto enq0 = std::chrono::steady_clock::now();
    try {
      ev = enqueueCmd(k, dim, gws, lws, tag.c_str(), lws != nullptr);
    } catch (const std::exception & e) {
      throw std::runtime_error("node " + tag + ": " + e.what());
    }
    // P2: host 入队（提交）耗时；即使非 profiling 也可量，但只在 profiling 时统计。
    if (profiling_)
      prof_enqueue_ms_ += std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - enq0).count();
    if (profiling_)
    {
      const auto w0 = std::chrono::steady_clock::now();
      clWaitForEvents(1, &ev);
      prof_wait_ms_ += std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - w0).count();
      cl_ulong s = 0, e = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
      tprof_[tag].first += static_cast<double>(e - s) * 1e-6;
      tprof_[tag].second += 1;
      noteNode(cur_node_, tag, static_cast<double>(e - s) * 1e-6);
    }
    clReleaseEvent(ev);
  };

  if (!captured_)
    node_cmds_.assign(nodes_.size(), {});
  if (node_ms_.size() != nodes_.size())
  {
    node_ms_.assign(nodes_.size(), 0.0);
    node_calls_.assign(nodes_.size(), 0);
    node_tag_.assign(nodes_.size(), std::string());
  }

  for (size_t ni = 0; ni < nodes_.size(); ++ni)
  {
    const auto & n = nodes_[ni];
    cur_node_ = ni;
    if (ni < node_skipped_.size() && node_skipped_[ni]) continue;  // P0: 已被别名取代
    auto & out = ref(n.outs[0]);
    if (n.op == "reshape" || n.op == "flatten")
    {
      out.mem = ref(n.ins[0]).mem;  // 视图别名，零拷贝
      continue;
    }
    if (captured_ && launch_cache_)
    {
      replayNode(ni);  // P2: 首帧已录制，重放（零签名/字符串/setArg）
      continue;
    }

    capturing_ = launch_cache_;
    cap_node_  = ni;
    cap_cmds_.clear();
    cap_args_.clear();

    auto in = [&](size_t i) -> Tensor & { return ref(n.ins[i]); };
    auto out_numel = [&]() -> int64_t {
      auto it = T_.find(n.outs[0]);
      if (it != T_.end()) return it->second.numel();
      return T_.at(n.ins[0]).numel();
    };
    const size_t  g1   = static_cast<size_t>(out_numel());

    // Round 28: 小算子调优查表 —— 命中缓存用其 kernel/options，否则内置默认。
    // 只影响「用哪个变体」，数值由 kernel 语义决定。
    // R45: 统一走 choiceEntry（per-node 覆盖优先，回退签名缓存），与 conv 族一致。
    auto smallKernelFor = [&](const OpSignature & sig, const char * dk, std::string & kernOut,
                              std::string & optsOut) -> cl_kernel {
      kernOut = dk;
      optsOut.clear();
      if (const TuningEntry * e = choiceEntry(cur_node_, sig)) {
        kernOut = e->kernel;
        optsOut = e->options;
      }
      // R48 D4: 消费者全是 blocked 时，布局规划器把本张量标记为 fsv16；生产者直接写 fsv16
      // （目前仅 ew_binary_ch 实现），省掉独立 reorder pass。与 planBlockedLayout 的标记条件一致。
      if (kernOut == "ew_binary_ch" && !n.outs.empty() && !std::getenv("INFVINO_NO_D4")) {
        auto oit = T_.find(n.outs[0]);
        if (oit != T_.end() && oit->second.fsv16 && optsOut.find("-DEWCH_OUT_FSV16=") == std::string::npos)
          optsOut += " -DEWCH_OUT_FSV16=1";
      }
      return getKernel("ops", kernOut, optsOut);
    };

    if (n.op == "conv1x1_cat4")
    {
      // R30c Route A: fused concat4->conv1x1 via gemm_f16 CAT4 B-staging.
      // A = weight [Cout][Cin]; C = out [Cout][HW]; logical B [Cin][HW] is the
      // concat of 4 contiguous sources (b0..b3) with channel counts ca/cb/cc/cd.
      const int act = attrInt(n, "act", 0);
      auto & w = in(0);
      const int Cout = static_cast<int>(w.dims[0]);
      const int Cin  = static_cast<int>(w.dims[1]);
      const int ca = attrInt(n, "cat_ca", 0), cb = attrInt(n, "cat_cb", 0),
                cc = attrInt(n, "cat_cc", 0), cd = attrInt(n, "cat_cd", 0);
      const int o0 = attrInt(n, "cat_o0", 0), o1 = attrInt(n, "cat_o1", 0),
                o2 = attrInt(n, "cat_o2", 0), o3 = attrInt(n, "cat_o3", 0);
      // HW 从**输出**张量取（源可能是父张量、比一段大）。
      const int64_t onumel = out.numel();
      const int HW = (Cout > 0) ? static_cast<int>(onumel / Cout) : 0;
      cl_mem dA = w.mem, dC = out.mem;
      cl_mem dB0 = in(1).mem, dB1 = in(2).mem, dB2 = in(3).mem, dB3 = in(4).mem;
      cl_mem db = (n.ins.size() > 5 && n.ins[5] != "-") ? in(5).mem : nullptr;
      cl_mem dres = (n.ins.size() > 6 && n.ins[6] != "-") ? in(6).mem : nullptr;
      // tile: same heuristic as the plain conv1x1 N>1 path.
      Tiles t;
      const long grid = static_cast<long>((Cout + t.BM - 1) / t.BM) *
                        static_cast<long>((HW + t.BN - 1) / t.BN);
      if (Cout <= 64) t.BM = 64;
      else if (Cin >= 192 && grid >= 64) { t.BK = 32; t.DBUF = 0; }
      else { t.BK = 8; t.DBUF = 0; }
      if (Cin < 32) t.SG = 0;
      t.EPI = 1; t.ACT = act; t.ASYNC = 0; t.PF = 0; t.GN = 0;
      std::string gopts = t.options() + " -DCAT4=1";
      const int coff[4] = {o0, o1, o2, o3};
      const OpSignature sig = OpSignature::conv1x1Cat4(Cout, HW, Cin, ca, cb, cc, cd, coff,
                                                       act, dres ? 1 : 0);
      // R45: 统一走 choiceEntry（per-node 覆盖优先，回退签名缓存）。
      if (const TuningEntry * e = choiceEntry(ni, sig)) gopts = e->options;
      if (dres) setResOpt(gopts, true);  // R33 融合残差
      cl_kernel kg = getKernel("gemm", "gemm_f16", gopts);
      auto optInt = [&](const char * key, int def) {
        const auto p = gopts.find(key);
        return p == std::string::npos ? def : std::atoi(gopts.c_str() + p + std::strlen(key));
      };
      t.BM = optInt("-DBM=", t.BM); t.BN = optInt("-DBN=", t.BN);
      t.TM = optInt("-DTM=", t.TM); t.TN = optInt("-DTN=", t.TN);
      setArg(kg, 0, sizeof(dA), &dA);
      setArg(kg, 1, sizeof(dB0), &dB0);
      setArg(kg, 2, sizeof(dC), &dC);
      setArg(kg, 3, sizeof(Cout), &Cout);
      setArg(kg, 4, sizeof(HW), &HW);
      setArg(kg, 5, sizeof(Cin), &Cin);
      setArg(kg, 6, sizeof(db), &db);
      setArg(kg, 7, sizeof(dres), &dres);
      setArg(kg, 8, sizeof(dB1), &dB1);
      setArg(kg, 9, sizeof(dB2), &dB2);
      setArg(kg, 10, sizeof(dB3), &dB3);
      setArg(kg, 11, sizeof(ca), &ca);
      setArg(kg, 12, sizeof(cb), &cb);
      setArg(kg, 13, sizeof(cc), &cc);
      setArg(kg, 14, sizeof(o0), &o0);
      setArg(kg, 15, sizeof(o1), &o1);
      setArg(kg, 16, sizeof(o2), &o2);
      setArg(kg, 17, sizeof(o3), &o3);
      const size_t lws[2] = {t.localX(), t.localY()};
      const size_t gws[2] = {
        static_cast<size_t>((HW + t.BN - 1) / t.BN) * lws[0],
        static_cast<size_t>((Cout + t.BM - 1) / t.BM) * lws[1]};
      timed("conv1x1cat4@" + std::to_string(Cout) + "x" + std::to_string(HW) + "x" +
              std::to_string(Cin),
            kg, 2, gws, lws);
    }
    else if (n.op == "conv1x1")
    {
      // Round 22: dedicated 1x1 conv / fc path.
      //   * N==1 (HW==1): split-K GEMV (one sub-group per output channel) — the
      //     generic GEMM wastes 63/64 lanes and starves the grid on N=1.
      //   * N>1: tuned GEMM with a fused bias+activation epilogue (no separate
      //     bias_add / ew_unary launches).
      const int act = attrInt(n, "act", 0);
      auto & w = in(0);                        // [Cout][Cin]
      const int Cout = static_cast<int>(w.dims[0]);
      const int Cin  = static_cast<int>(w.dims[1]);
      const int64_t xnumel = in(1).numel();
      const int N = (Cin > 0) ? static_cast<int>(xnumel / Cin) : 0;
      int Hin = 1, Win = 1;
      {
        const auto & xd = in(1).dims;
        if (xd.size() >= 2) { Hin = static_cast<int>(xd[xd.size() - 2]); Win = static_cast<int>(xd.back()); }
        if (Hin * Win != N) { Hin = N; Win = 1; }
      }
      cl_mem dw = w.mem, dx = in(1).mem, dy = out.mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? in(2).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? in(3).mem : nullptr;
      // R51 D5: 融合的逐输入通道 scale（SE Mul 折进 prologue）。
      cl_mem dscale = (n.ins.size() > 4 && n.ins[4] != "-") ? in(4).mem : nullptr;
      if (N == 1) {
        Conv1x1Cfg cfg;
        cfg.ACT = act;
        cfg.RES = dres ? 1 : 0;
        cfg.SG  = 16;
        std::string gopts = cfg.options();
        const OpSignature sig = conv1x1Sig(n, Cout, N, Cin, act);
        // R45 P0#6: 统一走 choiceEntry（per-node 覆盖优先，回退签名缓存）。
        if (const TuningEntry * e = choiceEntry(ni, sig)) gopts = e->options;
        if (dres) setResOpt(gopts, true);  // R33 融合残差
        cl_kernel kg = getKernel("conv1x1", "conv1x1_gemv_f16", gopts);
        setArg(kg, 0, sizeof(dw), &dw);
        setArg(kg, 1, sizeof(dx), &dx);
        setArg(kg, 2, sizeof(db), &db);
        setArg(kg, 3, sizeof(dres), &dres);
        setArg(kg, 4, sizeof(dy), &dy);
        setArg(kg, 5, sizeof(Cin), &Cin);
        setArg(kg, 6, sizeof(Cout), &Cout);
        const size_t lws[1] = {16};
        const size_t gws[1] = {static_cast<size_t>(Cout) * 16};
        timed("conv1x1g@" + std::to_string(Cout) + "x" + std::to_string(Cin), kg, 1, gws, lws);
      } else {
        Tiles t;
        const long grid = static_cast<long>((Cout + t.BM - 1) / t.BM) *
                          static_cast<long>((N + t.BN - 1) / t.BN);
        if (Cout <= 64) {
          t.BM = 64;
        } else if (Cin >= 192 && grid >= 64) {
          t.BK = 32;
          t.DBUF = 0;
        } else {
          t.BK = 8;
          t.DBUF = 0;
        }
        if (Cin < 32) t.SG = 0;
        t.EPI = 1;
        t.ACT = act;
        std::string gopts = t.options();
        const OpSignature sig = conv1x1Sig(n, Cout, N, Cin, act);
        std::string gkernel = "gemm_f16";
        const size_t ni1 = static_cast<size_t>(&n - nodes_.data());
        if (const TuningEntry * e = choiceEntry(ni1, sig)) {
          gopts = e->options;
          if (!e->kernel.empty()) gkernel = e->kernel;
          if (gkernel == "gemm_f16") {
            auto optInt = [&](const char * k, int def) {
              const auto p = gopts.find(k);
              return p == std::string::npos ? def : std::atoi(gopts.c_str() + p + std::strlen(k));
            };
            t.BM = optInt("-DBM=", t.BM);
            t.BN = optInt("-DBN=", t.BN);
            t.TM = optInt("-DTM=", t.TM);
            t.TN = optInt("-DTN=", t.TN);
          }
        }
        if (gkernel == "conv1x1_blk") {
          // blocked 1x1（fsv16 输入 + osv16 权重 + NCHW 输出）。几何从 options 回放。
          auto optInt = [&](const char * k, int def) {
            const auto p = gopts.find(k);
            return p == std::string::npos ? def : std::atoi(gopts.c_str() + p + std::strlen(k));
          };
          const int xb = optInt("-DX_BLOCK=", 4), slm = optInt("-DSLM_DIV=", 1);
          const int yb = optInt("-DY_BLOCK=", 1);   // R48 D1: 输出行 tiling
          if (std::getenv("INFVINO_DEBUG_BLK"))
            std::fprintf(stderr, "[blk1x1] out=%s Cout=%d Cin=%d Hin=%d Win=%d N=%d xfsv16=%d XB=%d YB=%d SLM=%d\n",
                         n.outs[0].c_str(), Cout, Cin, Hin, Win, N, (int)in(1).fsv16, xb, yb, slm);
          cl_mem dxb = blkInput(n.ins[1], in(1), Cin, Hin, Win);
          cl_mem dwb = blk1x1Weight(n.ins[0], w, Cout, Cin);
          std::string kbopts = gopts;
          if (out.fsv16 && kbopts.find("-DOUT_FSV16=") == std::string::npos)
            kbopts += " -DOUT_FSV16=1";
          if (dres) setResOpt(kbopts, true);
          if (dscale) kbopts += " -DMUL_SCALE=1";   // R51 D5
          cl_kernel kb = getKernel("conv1x1_blk", "conv1x1_blk", kbopts);
          setArg(kb, 0, sizeof(dxb), &dxb);
          setArg(kb, 1, sizeof(dwb), &dwb);
          setArg(kb, 2, sizeof(db), &db);
          setArg(kb, 3, sizeof(dy), &dy);
          setArg(kb, 4, sizeof(dres), &dres);
          setArg(kb, 5, sizeof(Cin), &Cin);
          setArg(kb, 6, sizeof(Hin), &Hin);
          setArg(kb, 7, sizeof(Win), &Win);
          setArg(kb, 8, sizeof(Cout), &Cout);
          if (dscale) setArg(kb, 9, sizeof(dscale), &dscale);
          const size_t blws[3] = {1, static_cast<size_t>(16 * slm), 1};
          const size_t ybCount = static_cast<size_t>((Hin + yb - 1) / yb);  // R48 D1
          const size_t bgws[3] = {static_cast<size_t>(((Win + xb - 1) / xb)) * ybCount,
                                  static_cast<size_t>(((Cout + 15) / 16) * blws[1]), 1};
          timed("conv1x1blk@" + std::to_string(Cout) + "x" + std::to_string(N) + "x" +
                  std::to_string(Cin),
                kb, 3, bgws, blws);
        } else {
        // R32: split-K（lane 沿 K）候选走 gemm_sk_f16，几何不同；两者参数表相同。
        const bool sk = (gkernel == "gemm_sk_f16");
        if (dres) setResOpt(gopts, true);  // R33 融合残差（gemm_f16/gemm_sk 都支持 RES）
        if (dscale) gopts += " -DMUL_SCALE=1";   // R51 D5
        cl_kernel kg = getKernel(sk ? "gemm_sk" : "gemm", gkernel, gopts);
        setArg(kg, 0, sizeof(dw), &dw);
        setArg(kg, 1, sizeof(dx), &dx);
        setArg(kg, 2, sizeof(dy), &dy);
        setArg(kg, 3, sizeof(Cout), &Cout);
        setArg(kg, 4, sizeof(N), &N);
        setArg(kg, 5, sizeof(Cin), &Cin);
        setArg(kg, 6, sizeof(db), &db);
        setArg(kg, 7, sizeof(dres), &dres);
        if (dscale) setArg(kg, 8, sizeof(dscale), &dscale);
        size_t lws[2], gws[2];
        if (sk) {
          auto optInt = [&](const char * k, int def) {
            const auto p = gopts.find(k);
            return p == std::string::npos ? def : std::atoi(gopts.c_str() + p + std::strlen(k));
          };
          const int TM = optInt("-DSK_TM=", 8), TN = optInt("-DSK_TN=", 4),
                    SG = optInt("-DSK_SG=", 16);
          lws[0] = static_cast<size_t>(SG);
          lws[1] = 1;
          gws[0] = static_cast<size_t>((N + TN - 1) / TN) * lws[0];
          gws[1] = static_cast<size_t>((Cout + TM - 1) / TM);
        } else {
          lws[0] = t.localX();
          lws[1] = t.localY();
          gws[0] = static_cast<size_t>((N + t.BN - 1) / t.BN) * lws[0];
          gws[1] = static_cast<size_t>((Cout + t.BM - 1) / t.BM) * lws[1];
        }
        timed("conv1x1@" + std::to_string(Cout) + "x" + std::to_string(N) + "x" +
                std::to_string(Cin),
              kg, 2, gws, lws);
        }
      }
    }
    else if (n.op == "conv_general")
    {
      const int K = attrInt(n, "K", 3), S = attrInt(n, "S", 1), P = attrInt(n, "P", 1),
                G = attrInt(n, "G", 1), act = attrInt(n, "act", 0);
      const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
      const auto & id  = in(0).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      int Cin  = static_cast<int>(id[base]), H = static_cast<int>(id[base + 1]);
      int W    = static_cast<int>(id[base + 2]);
      int Cout = static_cast<int>(out.dims[out.dims.size() >= 3 ? out.dims.size() - 3 : 0]);
      cl_mem dx = in(0).mem, dw = in(1).mem, dy = out.mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? in(2).mem : nullptr;
      // Round 16: native depthwise (groups == Cin) -> vectorised kernel.
      if (G == Cin && (K == 3 || K == 5) && Cin == Cout) {
        char dopts[128];
        std::snprintf(dopts, sizeof(dopts),
                      "-DDW_K=%d -DDW_S=%d -DDW_P=%d -DDW_ACT=%d "
                      "-cl-mad-enable -cl-fast-relaxed-math", K, S, P, act);
        std::string dwopts = dopts;
        std::string dkern = "depthwise_f16";
        const OpSignature sig = OpSignature::depthwise(Wout, Hout, S, P, Cin, K, act);
        const size_t ni1 = static_cast<size_t>(&n - nodes_.data());
        if (const TuningEntry * e = choiceEntry(ni1, sig)) {
          if (!e->kernel.empty()) dkern = e->kernel;
          dwopts = e->options;
        }
        if (dkern == "depthwise_blk") {
          // blocked depthwise（fsv16 输入/输出 + [C/16][K][K][16] 权重）。
          auto optInt = [&](const char * key, int def) {
            const auto p = dwopts.find(key);
            return p == std::string::npos ? def : std::atoi(dwopts.c_str() + p + std::strlen(key));
          };
          const int xb = optInt("-DX_BLOCK=", 8);
          const int yb = optInt("-DY_BLOCK=", 1);
          cl_mem dxb = blkInput(n.ins[0], in(0), Cin, H, W);
          cl_mem dwb = blkDwWeight(n.ins[1], in(1), Cin, K);
          std::string bopts = dwopts;
          if (out.fsv16 && bopts.find("-DOUT_FSV16=") == std::string::npos)
            bopts += " -DOUT_FSV16=1";
          cl_kernel kd = getKernel("depthwise_blk", "depthwise_blk", bopts);
          setArg(kd, 0, sizeof(dxb), &dxb);
          setArg(kd, 1, sizeof(dwb), &dwb);
          setArg(kd, 2, sizeof(db), &db);
          setArg(kd, 3, sizeof(dy), &dy);
          setArg(kd, 4, sizeof(Cin), &Cin);
          setArg(kd, 5, sizeof(H), &H);
          setArg(kd, 6, sizeof(W), &W);
          int ho = Hout, wo = Wout;
          setArg(kd, 7, sizeof(ho), &ho);
          setArg(kd, 8, sizeof(wo), &wo);
          const size_t g[3] = {static_cast<size_t>(((Wout + xb - 1) / xb) * ((Hout + yb - 1) / yb)),
                               static_cast<size_t>(((Cin + 15) / 16) * 16), 1};
          const size_t l[3] = {1, 16, 1};
          timed("depthwise", kd, 3, g, l);
        } else if (dkern == "depthwise_vp") {
          // R31: 每帧先把 interior 重排进零边 Xp（约 2×输入字节的一趟），再无边界地卷积。
          int Hp = H + 2 * P, Wpad = 0;
          cl_mem xp = dwPadInput(n.ins[0], Cin, H, W, K, S, P, &Hp, &Wpad);
          cl_kernel kp = getKernel("conv_general", "depthwise_pad", "");
          setArg(kp, 0, sizeof(dx), &dx);
          setArg(kp, 1, sizeof(xp), &xp);
          setArg(kp, 2, sizeof(Cin), &Cin);
          setArg(kp, 3, sizeof(H), &H);
          setArg(kp, 4, sizeof(W), &W);
          setArg(kp, 5, sizeof(Hp), &Hp);
          setArg(kp, 6, sizeof(Wpad), &Wpad);
          setArg(kp, 7, sizeof(P), &P);
          const size_t gp[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                                static_cast<size_t>(Cin)};
          timed("depthwise_pad", kp, 3, gp, nullptr);

          cl_kernel kd = getKernel("conv_general", dkern, dwopts);
          setArg(kd, 0, sizeof(xp), &xp);
          setArg(kd, 1, sizeof(dw), &dw);
          setArg(kd, 2, sizeof(db), &db);
          setArg(kd, 3, sizeof(dy), &dy);
          setArg(kd, 4, sizeof(Cin), &Cin);
          setArg(kd, 5, sizeof(Hp), &Hp);
          setArg(kd, 6, sizeof(Wpad), &Wpad);
          int ho = Hout, wo = Wout;
          setArg(kd, 7, sizeof(ho), &ho);
          setArg(kd, 8, sizeof(wo), &wo);
          auto p = dwopts.find("-DDW_TW=");
          const int tw = p == std::string::npos ? 8 : std::atoi(dwopts.c_str() + p + 8);
          const size_t g[3] = {static_cast<size_t>((Wout + tw - 1) / tw),
                               static_cast<size_t>(Hout), static_cast<size_t>(Cin)};
          timed("depthwise", kd, 3, g, nullptr);
        } else {
        cl_kernel kd = getKernel("conv_general", dkern, dwopts);
        int ho = Hout, wo = Wout;
        setArg(kd, 0, sizeof(dx), &dx);
        setArg(kd, 1, sizeof(dw), &dw);
        setArg(kd, 2, sizeof(db), &db);
        setArg(kd, 3, sizeof(dy), &dy);
        setArg(kd, 4, sizeof(Cin), &Cin);
        setArg(kd, 5, sizeof(H), &H);
        setArg(kd, 6, sizeof(W), &W);
        setArg(kd, 7, sizeof(ho), &ho);
        setArg(kd, 8, sizeof(wo), &wo);
        if (dkern == "depthwise_v") {
          auto p = dwopts.find("-DDW_TW=");
          const int tw = p == std::string::npos ? 4 : std::atoi(dwopts.c_str() + p + 8);
          const size_t g[3] = {static_cast<size_t>((Wout + tw - 1) / tw),
                               static_cast<size_t>(Hout), static_cast<size_t>(Cin)};
          timed("depthwise", kd, 3, g, nullptr);
        } else {
          const size_t gdw[1] = {static_cast<size_t>(Cin) * Hout * Wout};
          timed("depthwise", kd, 1, gdw, nullptr);
        }
        }
      } else {
      setArg(kConvG_, 0, sizeof(dx), &dx);
      setArg(kConvG_, 1, sizeof(dw), &dw);
      setArg(kConvG_, 2, sizeof(db), &db);
      setArg(kConvG_, 3, sizeof(dy), &dy);
      setArg(kConvG_, 4, sizeof(Cin), &Cin);
      setArg(kConvG_, 5, sizeof(H), &H);
      setArg(kConvG_, 6, sizeof(W), &W);
      setArg(kConvG_, 7, sizeof(Cout), &Cout);
      int ho = Hout, wo = Wout;
      setArg(kConvG_, 8, sizeof(ho), &ho);
      setArg(kConvG_, 9, sizeof(wo), &wo);
      int KK = K, SS = S, PP = P, GG = G, aa = act;
      setArg(kConvG_, 10, sizeof(KK), &KK);
      setArg(kConvG_, 11, sizeof(SS), &SS);
      setArg(kConvG_, 12, sizeof(PP), &PP);
      setArg(kConvG_, 13, sizeof(GG), &GG);
      setArg(kConvG_, 14, sizeof(aa), &aa);
      const size_t g = static_cast<size_t>(Cout) * ho * wo;
      timed("conv_general", kConvG_, 1, &g, nullptr);
      }
    }
    else if (n.op == "conv3x3")
    {
      const int stride = attrInt(n, "stride", 1), pad = attrInt(n, "pad", 1),
                act = attrInt(n, "act", 0);
      const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
      const auto & id = in(0).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      int Cin  = static_cast<int>(id[base]), H = static_cast<int>(id[base + 1]);
      int W    = static_cast<int>(id[base + 2]);
      const auto & od = out.dims;
      int Cout = static_cast<int>(od[od.size() >= 3 ? od.size() - 3 : 0]);
      cl_mem dx = in(0).mem, dy = out.mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? in(2).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? in(3).mem : nullptr;

      // 自动调优：按签名查表。命中 → 用缓存里的 kernel/options；未命中 → 内置启发式。
      // 数值与 kernel 语义不变，只改「选哪个 kernel/config」。
      const OpSignature tsig =
        OpSignature::conv3x3(Wout, Hout, stride, pad, Cin, Cout, act);
      // R38: per-node joint-fixpoint choice (blk vs non-blk with layout-exact reorder),
      // falling back to the signature cache.
      const TuningEntry * te = choiceEntry(ni, tsig);
      // R33: 带残差的 conv3x3 只有 conv3x3_ov 支持 RES；native/blk 不支持 → 绕回调优走 ov。
      if (dres && te && te->kernel != "conv3x3_ov") te = nullptr;

      if (te && te->kernel == "conv3x3_f16")
      {
        // 调优命中的 native conv3x3_f16：config 串回放为 Conv3x3Cfg 选项（options 已存）。
        cl_kernel kk = getKernel(sourceOfKernel(te->kernel), te->kernel, te->options);
        cl_mem dw = in(1).mem;
        // 从 options 回放 launch 几何（tuner 存的 options 就是 kernel 的编译宏）。
        auto optInt = [&](const char * k, int def) {
          const auto p = te->options.find(k);
          return p == std::string::npos ? def : std::atoi(te->options.c_str() + p + std::strlen(k));
        };
        Conv3x3Cfg cfg;
        cfg.TX = optInt("-DTX=", (Wout >= 40) ? 40 : (Wout >= 20 ? 20 : 16));
        cfg.TY = optInt("-DTY=", 8);
        cfg.TM = optInt("-DTM=", 1);
        cfg.CB = optInt("-DCB=", 32);
        cfg.STRIDE = stride; cfg.PAD = pad; cfg.ACT = act; cfg.SG = 16; cfg.WC = 1;
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dy), &dy);
        setArg(kk, 4, sizeof(Cin), &Cin);
        setArg(kk, 5, sizeof(H), &H);
        setArg(kk, 6, sizeof(W), &W);
        setArg(kk, 7, sizeof(Cout), &Cout);
        setArg(kk, 8, sizeof(Hout), &Hout);
        setArg(kk, 9, sizeof(Wout), &Wout);
        const size_t lws[3] = {
          static_cast<size_t>(cfg.TX / cfg.TM), static_cast<size_t>(cfg.TY), 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + cfg.TX - 1) / cfg.TX) * lws[0],
          static_cast<size_t>((Hout + cfg.TY - 1) / cfg.TY) * lws[1],
          static_cast<size_t>((Cout + cfg.CB - 1) / cfg.CB)};
        timed("conv3x3@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(cfg.STRIDE) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout) + "(tuned)",
              kk, 3, gws, lws);
      }
      else if (te && te->kernel == "conv3x3_cin3")
      {
        // Cin<=4 专用首层 conv（kernels/conv_cin3.cl）：lane=空间列、每 WI 算全部
        // Cout、权重 SLM [k][c] 广播。无权重重排（直接读 [Cout][Cin][3][3]）。
        cl_kernel kk = getKernel("conv_cin3", "conv3x3_cin3", te->options);
        cl_mem dw = cin3Weight(n.ins[1], in(1), Cout, Cin);
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dy), &dy);
        setArg(kk, 4, sizeof(H), &H);
        setArg(kk, 5, sizeof(W), &W);
        setArg(kk, 6, sizeof(Hout), &Hout);
        setArg(kk, 7, sizeof(Wout), &Wout);
        const size_t lws[2] = {128, 1};
        const size_t gws[2] = {
          (static_cast<size_t>(Wout) + 127) / 128 * 128, static_cast<size_t>(Hout)};
        timed("conv3x3cin3@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(stride) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout) + "(tuned)",
              kk, 2, gws, lws);
      }
      else if ((te && te->kernel == "conv3x3_blk") || (attrInt(n, "blk", 0) != 0 && !te))
      {
        // R25: OpenVINO blocked conv port (kernels/conv_blk.cl). Reorders the
        // input bfyx -> b_fs_yx_fsv16 (cached scratch, unless the producer already
        // emitted fsv16 — R36) then runs the blocked kernel, writing bfyx output
        // (or b_fs_yx_fsv16 when the whole consumer set is blocked — R36).
        int obw = 8, slm = 1;
        std::string oo;
        if (te) {
          oo = te->options;
          auto p = oo.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(oo.c_str() + p + 6);
          auto ps = oo.find("-DSLM_DIV=");
          if (ps != std::string::npos) slm = std::atoi(oo.c_str() + ps + 10);  // len("-DSLM_DIV=")==10
          if (slm < 1) slm = 1;
        } else {
          char buf[192];
          std::snprintf(buf, sizeof(buf),
                        "-DOBW=%d -DSLM_DIV=1 -DSTRIDE=%d -DPAD=%d -DACT=%d -DSG=16 "
                        "-cl-mad-enable -cl-fast-relaxed-math", obw, stride, pad, act);
          oo = buf;
        }
        // R36 (P1-layout)：输出端持久 blocked —— 消费者全是 blocked conv 时直接写
        // b_fs_yx_fsv16，让它们零 reorder 直接读。
        if (out.fsv16 && oo.find("-DOUT_FSV16=") == std::string::npos)
          oo += " -DOUT_FSV16=1";
        cl_kernel kk = getKernel("conv_blk", "conv3x3_blk", oo);
        cl_mem dw = blkWeight(n.ins[1], in(1), Cout, Cin);
        cl_mem dxb = blkInput(n.ins[0], in(0), Cin, H, W);
        setArg(kk, 0, sizeof(dxb), &dxb);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dy), &dy);
        setArg(kk, 4, sizeof(Cin), &Cin);
        setArg(kk, 5, sizeof(H), &H);
        setArg(kk, 6, sizeof(W), &W);
        setArg(kk, 7, sizeof(Cout), &Cout);
        setArg(kk, 8, sizeof(Hout), &Hout);
        setArg(kk, 9, sizeof(Wout), &Wout);
        const size_t lws[3] = {1, static_cast<size_t>(16 * slm), 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + obw - 1) / obw) * static_cast<size_t>(Hout),
          static_cast<size_t>(((Cout + 15) / 16) * 16 * slm), 1};
        timed("conv3x3blk@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(stride) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout) + (te ? "(tuned)" : ""),
              kk, 3, gws, lws);
      }
      else if (te && te->kernel == "conv3x3_ov")
      {
        // 调优命中的 OV osv32：options 里已含 OBW/OBH/STRIDE/PAD/ACT/RES/SG。
        // 从 options 解析 OBW/OBH 以重建 grid（避免再解析 config 串）。
        std::string ovopts = te->options;
        if (dres) setResOpt(ovopts, true);  // R33 融合残差
        int obw = 8, obh = 2, slm = 1;
        {
          auto p = ovopts.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(ovopts.c_str() + p + 6);
          p = ovopts.find("-DOBH=");
          if (p != std::string::npos) obh = std::atoi(ovopts.c_str() + p + 6);
          auto ps = ovopts.find("-DSLM_DIV=");
          if (ps != std::string::npos) slm = std::atoi(ovopts.c_str() + ps + 10);  // len==10
          if (slm < 1) slm = 1;
        }
        cl_kernel kk = getKernel(sourceOfKernel(te->kernel), te->kernel, ovopts);
        cl_mem dw = ovWeight(n.ins[1], in(1), Cout, Cin);
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dres), &dres);
        setArg(kk, 4, sizeof(dy), &dy);
        setArg(kk, 5, sizeof(Cin), &Cin);
        setArg(kk, 6, sizeof(H), &H);
        setArg(kk, 7, sizeof(W), &W);
        setArg(kk, 8, sizeof(Cout), &Cout);
        setArg(kk, 9, sizeof(Hout), &Hout);
        setArg(kk, 10, sizeof(Wout), &Wout);
        const size_t lws[3] = {1, 1, static_cast<size_t>(16 * slm)};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + obw - 1) / obw),
          static_cast<size_t>((Hout + obh - 1) / obh),
          static_cast<size_t>((((Cout + 1) / 2) + 15) / 16) * 16 * static_cast<size_t>(slm)};
        timed("conv3x3ov@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(stride) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout) + "(tuned)",
              kk, 3, gws, lws);
      }
      else if (attrInt(n, "ov", 1))
      {
        int obw = (stride == 2) ? 5 : 8;
        int obh = (stride == 2) ? 4 : 2;
        if (obw > Wout) obw = Wout > 0 ? Wout : obw;
        if (obh > Hout) obh = Hout > 0 ? Hout : obh;
        char oo[192];
        cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? in(3).mem : nullptr;
        std::snprintf(oo, sizeof(oo),
                      "-DOBW=%d -DOBH=%d -DSTRIDE=%d -DPAD=%d -DACT=%d -DRES=%d -DSG=16 "
                      "-cl-mad-enable -cl-fast-relaxed-math",
                      obw, obh, stride, pad, act, dres ? 1 : 0);
        cl_kernel kk = getKernel("conv_ov", "conv3x3_ov", oo);
        cl_mem dw = ovWeight(n.ins[1], in(1), Cout, Cin);
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dres), &dres);
        setArg(kk, 4, sizeof(dy), &dy);
        setArg(kk, 5, sizeof(Cin), &Cin);
        setArg(kk, 6, sizeof(H), &H);
        setArg(kk, 7, sizeof(W), &W);
        setArg(kk, 8, sizeof(Cout), &Cout);
        setArg(kk, 9, sizeof(Hout), &Hout);
        setArg(kk, 10, sizeof(Wout), &Wout);
        const size_t lws[3] = {1, 1, 16};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + obw - 1) / obw),
          static_cast<size_t>((Hout + obh - 1) / obh),
          static_cast<size_t>((((Cout + 1) / 2) + 15) / 16) * 16};
        timed("conv3x3ov@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(stride) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout),
              kk, 3, gws, lws);
      }
      else
      {
        Conv3x3Cfg cfg;
        auto       it = n.attr.find("cfg");
        if (it != n.attr.end()) cfg = parseConv(it->second);
        cfg.STRIDE = stride;
        cfg.PAD    = pad;
        cfg.ACT    = act;
        cfg.SG     = 16;  // Round 15: pin SIMD16 (see Tiles.hpp / docs R15)
        cfg.WC     = 1;   // Round 18: coalesced weight staging (+5% on 3x3)
        // Round 15: adaptive spatial tile.
        cfg.TX = (Wout >= 40) ? 40 : (Wout >= 20 ? 20 : 16);
        if (cfg.TX > Wout) cfg.TX = Wout;
        cfg.TY = 8;
        const int spatial = ((Wout + cfg.TX - 1) / cfg.TX) * ((Hout + cfg.TY - 1) / cfg.TY);
        const int wgs32   = spatial * ((Cout + 31) / 32);
        if (Cout <= 16 || wgs32 < 16) cfg.CB = 16;
        cl_kernel kk = getKernel("conv", "conv3x3_f16", cfg.options());
        cl_mem dw = in(1).mem;
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dy), &dy);
        setArg(kk, 4, sizeof(Cin), &Cin);
        setArg(kk, 5, sizeof(H), &H);
        setArg(kk, 6, sizeof(W), &W);
        setArg(kk, 7, sizeof(Cout), &Cout);
        setArg(kk, 8, sizeof(Hout), &Hout);
        setArg(kk, 9, sizeof(Wout), &Wout);
        const size_t lws[3] = {
          static_cast<size_t>(cfg.TX / cfg.TM), static_cast<size_t>(cfg.TY), 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + cfg.TX - 1) / cfg.TX) * lws[0],
          static_cast<size_t>((Hout + cfg.TY - 1) / cfg.TY) * lws[1],
          static_cast<size_t>((Cout + cfg.CB - 1) / cfg.CB)};
        timed("conv3x3@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(cfg.STRIDE) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout),
              kk, 3, gws, lws);
      }
    }
    else if (n.op == "gemm")
    {
      // C[M,N] = A[M,K] * B[K,N]  (A=weights, B=input)
      const int M = static_cast<int>(in(0).dims[0]);
      const int K = static_cast<int>(in(0).dims[1]);
      const int N = static_cast<int>(in(1).numel() / K);
      cl_mem da = in(0).mem, db = in(1).mem, dc = out.mem;
      // Per-node tile (Round 9; revisited in Round 12).
      //   * M<=64            : BM=64 (half the A tile) + DBUF=1.
      //   * else large grid  : BK=32 DBUF=0 (R12: single-buffer with double the
      //                        k-tile amortises the barrier; SG=16 keeps SIMD16
      //                        so the staging latency hides across workgroups).
      //   * otherwise        : BK=8 DBUF=0 (few k-tiles: pipeline overhead wins).
      // SG=16 is the R12 default (see Tiles); it is dropped for tiny K where the
      // wider SIMD slightly regresses.
      Tiles t;
      const long grid = static_cast<long>((M + t.BM - 1) / t.BM) *
                        static_cast<long>((N + t.BN - 1) / t.BN);
      if (M <= 64) {
        t.BM = 64;
      } else if (K >= 192 && grid >= 64) {
        t.BK = 32;
        t.DBUF = 0;
      } else {
        t.BK = 8;
        t.DBUF = 0;
      }
      if (K < 32) t.SG = 0;
      std::string gopts = t.options();
      // 自动调优：命中则用缓存的 tile（options 已含全部编译宏）。
      // R45: 统一走 choiceEntry（per-node 覆盖优先，回退签名缓存）。
      const OpSignature gsig = OpSignature::gemm(M, N, K, 0);
      if (const TuningEntry * ge = choiceEntry(ni, gsig)) {
        gopts = ge->options;
        auto optInt = [&](const char * k, int def) {
          const auto p = gopts.find(k);
          return p == std::string::npos ? def : std::atoi(gopts.c_str() + p + std::strlen(k));
        };
        t.BM = optInt("-DBM=", t.BM);
        t.BN = optInt("-DBN=", t.BN);
        t.TM = optInt("-DTM=", t.TM);
        t.TN = optInt("-DTN=", t.TN);
      }
      cl_kernel kg = getKernel("gemm", "gemm_f16", gopts);
      setArg(kg, 0, sizeof(da), &da);
      setArg(kg, 1, sizeof(db), &db);
      setArg(kg, 2, sizeof(dc), &dc);
      setArg(kg, 3, sizeof(M), &M);
      setArg(kg, 4, sizeof(N), &N);
      setArg(kg, 5, sizeof(K), &K);
      const size_t lws[2] = {t.localX(), t.localY()};
      const size_t gws[2] = {
        static_cast<size_t>((N + t.BN - 1) / t.BN) * lws[0],
        static_cast<size_t>((M + t.BM - 1) / t.BM) * lws[1]};
      timed("gemm@" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
            kg, 2, gws, lws);
    }
    else if (n.op == "ew_binary")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      // R30: 广播默认走 3-D 网格核（去掉逐元素 div/mod）；通道标量走 ew_binary_ch；
      // 仅当有效秩 >4 时回退通用 rank 核。
      std::string dk = "ew_binary";
      if (sig.op == "ew_binary_bcast")
      {
        const long n  = sig.params.size() > 0 ? sig.params[0] : 0;
        const long C  = sig.params.size() > 3 ? sig.params[3] : 0;
        const int  er = sig.params.size() > 4 ? sig.params[4] : 4;
        if (C > 0 && C < n && n % C == 0) dk = "ew_binary_ch";
        else if (er <= 4)                 dk = "ew_binary_bcast4";
        else                              dk = "ew_binary_bcast";
      }
      cl_kernel   k = smallKernelFor(sig, dk.c_str(), kern, opts);
      cl_uint dim;
      size_t  gws[3], lws[3];
      bool    useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("ew_binary", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "ew_unary")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "ew_unary", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("ew_unary", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "copy_c")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "copy_c", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("copy_c", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "slice_axis")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "slice_axis", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("slice_axis", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "concat4")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "concat4", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("concat4", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "maxpool")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "maxpool", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("maxpool", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "resize_nn")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "resize_nn", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("resize_nn", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "softmax_axis")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "softmax_axis", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("softmax_axis", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "permute_0213")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "permute_0213", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("permute", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "bmm")
    {
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, "bmm", kern, opts);
      cl_uint     dim;
      size_t      gws[3], lws[3];
      bool        useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("bmm", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "gap")
    {
      const int C = attrInt(n, "C", 0), HW = attrInt(n, "HW", 1);
      const OpSignature sig = OpSignature::gap(C, HW);
      std::string kern = "gap_r", opts = "-DGAP_WGS=128";
      if (const TuningEntry * e = tuning_.lookup(sig)) {
        if (!e->kernel.empty()) kern = e->kernel;
        if (!e->options.empty()) opts = e->options;
      }
      // R51 D4+: 输入已持久 fsv16（生产者直写）时，gap 直读 fsv16（免一趟 reorder）。
      if (in(0).fsv16 && opts.find("-DGAP_IN_FSV16=") == std::string::npos)
        opts += " -DGAP_IN_FSV16=1";
      cl_kernel k = getKernel("ops", kern, opts);
      cl_uint   dim;
      size_t    gws[3], lws[3];
      bool      useLws;
      smallLaunch(n, k, kern, opts, dim, gws, lws, useLws);
      timed("gap", k, dim, gws, useLws ? lws : nullptr);
    }
    else if (n.op == "bias_add")
    {
      int HW = attrInt(n, "HW", 1);
      cl_mem dx = in(0).mem, db = in(1).mem, dy = out.mem;
      setArg(kBias_, 0, sizeof(dx), &dx);
      setArg(kBias_, 1, sizeof(db), &db);
      setArg(kBias_, 2, sizeof(dy), &dy);
      setArg(kBias_, 3, sizeof(HW), &HW);
      timed("bias_add", kBias_, 1, &g1, nullptr);
    }
    else
    {
      throw std::runtime_error("PlanModel: unknown op: " + n.op);
    }

    capturing_ = false;
    if (launch_cache_) node_cmds_[ni] = std::move(cap_cmds_);
  }

  captured_ = true;

  // R32 原型：首帧录制完成后，若开启则把整帧 dispatch 录进一个 command buffer。
  buildCommandBuffer();

  rt_.finish();
  last_run_ms_ =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void PlanModel::readOutput(size_t i, void * fp16_host)
{
  const Tensor & t = T_.at(outputs_.at(i));
  rt_.read(t.mem, static_cast<size_t>(t.numel()) * 2, fp16_host);
}

bool PlanModel::readTensor(const std::string & name, void * fp16_host) const
{
  auto it = T_.find(name);
  if (it == T_.end() || it->second.mem == nullptr) return false;
  const Tensor & t = it->second;
  const size_t   n = static_cast<size_t>(t.numel());
  const int64_t  C = t.dims.size() >= 3 ? t.dims[t.dims.size() - 3] : 0;
  if (t.fsv16 && t.dims.size() >= 3 && C > 0)
  {
    // R36/R51: 持久 blocked 张量在设备上是 [C/16][H][W][16]（通道补齐到 16 的倍数，
    // 故 C%16!=0 时缓冲比 numel 大）；诊断读回时转回 NCHW，语义不变。
    const int64_t H = t.dims[t.dims.size() - 2];
    const int64_t W = t.dims[t.dims.size() - 1];
    const int64_t Cpad = (C + 15) / 16 * 16;
    const int64_t nread = (n / C) * Cpad;   // 设备侧元素数（含补齐 lane）
    std::vector<uint16_t> tmp(static_cast<size_t>(nread));
    const_cast<ClRuntime &>(rt_).read(t.mem, tmp.size() * 2, tmp.data());
    uint16_t * dst = static_cast<uint16_t *>(fp16_host);
    for (int64_t p = 0; p < nread; ++p)
    {
      const int64_t l  = p % 16;
      const int64_t q  = p / 16;
      const int64_t x  = q % W;
      const int64_t q2 = q / W;
      const int64_t y  = q2 % H;
      const int64_t cb = q2 / H;
      const int64_t c  = cb * 16 + l;
      if (c < C) dst[(c * H + y) * W + x] = tmp[static_cast<size_t>(p)];
    }
    return true;
  }
  const_cast<ClRuntime &>(rt_).read(t.mem, n * 2, fp16_host);
  return true;
}

size_t PlanModel::tensorNumel(const std::string & name) const
{
  auto it = T_.find(name);
  return it == T_.end() ? 0 : static_cast<size_t>(it->second.numel());
}

// ---------------------------------------------------------------------------
// 离线自动调优
// ---------------------------------------------------------------------------

namespace
{
bool opInList(const std::vector<std::string> & ops, const std::string & op)
{
  if (ops.empty()) return true;
  return std::find(ops.begin(), ops.end(), op) != ops.end();
}
}  // namespace

OpSignature PlanModel::conv1x1Sig(const Node & n, int Cout, int N, int Cin, int act) const
{
  const bool res = n.ins.size() > 3 && n.ins[3] != "-";
  return OpSignature::conv1x1(Cout, N, Cin, act, res ? 1 : 0);
}

OpSignature PlanModel::nodeSignature(const Node & n, bool * ok) const
{
  if (ok) *ok = true;
  if (n.op == "conv3x3")
  {
    const int stride = attrInt(n, "stride", 1), pad = attrInt(n, "pad", 1), act = attrInt(n, "act", 0);
    const auto & id = T_.at(n.ins[0]).dims;
    const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
    const int Cin = (int)id[base];
    const auto & od = T_.at(n.outs[0]).dims;
    const int Cout = (int)od[od.size() >= 3 ? od.size() - 3 : 0];
    return OpSignature::conv3x3(attrInt(n, "Wout", 0), attrInt(n, "Hout", 0), stride, pad, Cin, Cout, act);
  }
  if (n.op == "gemm")
  {
    const auto & ad = T_.at(n.ins[0]).dims;
    const int M = (int)ad[0], K = (int)ad[1];
    const int N = (int)(T_.at(n.ins[1]).numel() / K);
    return OpSignature::gemm(M, N, K, 0);
  }
  if (n.op == "conv1x1_cat4")
  {
    const int act = attrInt(n, "act", 0);
    const auto & wd = T_.at(n.ins[0]).dims;
    const int Cout = (int)wd[0], Cin = (int)wd[1];
    const int ca = attrInt(n, "cat_ca", 0), cb = attrInt(n, "cat_cb", 0),
              cc = attrInt(n, "cat_cc", 0), cd = attrInt(n, "cat_cd", 0);
    const int HW = Cout > 0 ? (int)(T_.at(n.outs[0]).numel() / Cout) : 0;
    const bool res = n.ins.size() > 6 && n.ins[6] != "-";
    const int coff[4] = {attrInt(n, "cat_o0", 0), attrInt(n, "cat_o1", 0),
                         attrInt(n, "cat_o2", 0), attrInt(n, "cat_o3", 0)};
    return OpSignature::conv1x1Cat4(Cout, HW, Cin, ca, cb, cc, cd, coff, act, res ? 1 : 0);
  }
  if (n.op == "conv1x1")
  {
    const int act = attrInt(n, "act", 0);
    const auto & wd = T_.at(n.ins[0]).dims;
    const int Cout = (int)wd[0], Cin = (int)wd[1];
    const int N = Cin > 0 ? (int)(T_.at(n.ins[1]).numel() / Cin) : 0;
    return conv1x1Sig(n, Cout, N, Cin, act);
  }
  if (n.op == "conv_general")
  {
    const int K = attrInt(n, "K", 3), S = attrInt(n, "S", 1), P = attrInt(n, "P", 1),
              G = attrInt(n, "G", 1), act = attrInt(n, "act", 0);
    const auto & id = T_.at(n.ins[0]).dims;
    const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
    const int Cin = (int)id[base];
    const auto & od = T_.at(n.outs[0]).dims;
    const int Cout = (int)od[od.size() >= 3 ? od.size() - 3 : 0];
    if (G == Cin && (K == 3 || K == 5) && Cin == Cout)
      return OpSignature::depthwise(attrInt(n, "Wout", 0), attrInt(n, "Hout", 0), S, P, Cin, K, act);
    if (ok) *ok = false;
    return OpSignature::convGeneral(attrInt(n, "Wout", 0), attrInt(n, "Hout", 0), S, P, Cin, Cout, G, K, act);
  }
  if (n.op == "ew_binary" || n.op == "ew_unary" || n.op == "copy_c" ||
      n.op == "slice_axis" || n.op == "concat4" || n.op == "maxpool" ||
      n.op == "resize_nn" || n.op == "permute_0213" || n.op == "bmm" ||
      n.op == "softmax_axis" || n.op == "gap")
    return smallSig(n);  // Round 28: 小算子签名（与 dispatch/autotune 完全一致）
  if (ok) *ok = false;
  return OpSignature::custom(n.op, {});
}

std::vector<OpSignature> PlanModel::tuningSignatures(const std::vector<std::string> & ops) const
{
  std::vector<OpSignature> out;
  std::map<std::string, int> seen;
  auto add = [&](const OpSignature & s) {
    const std::string k = s.str();
    if (!seen.count(k)) { seen[k] = 1; out.push_back(s); }
  };
  for (const auto & n : nodes_)
  {
    // conv_general 仅当请求里含 "depthwise" 时才可能纳入（与旧语义一致）。
    const std::string wop = n.op == "conv_general" ? "depthwise" : n.op;
    if (!ops.empty() && std::find(ops.begin(), ops.end(), wop) == ops.end()) continue;
    bool ok = true;
    OpSignature s = nodeSignature(n, &ok);
    if (ok) add(s);
  }
  return out;
}

std::vector<std::string> PlanModel::tuningTargets(const std::vector<std::string> & ops) const
{
  std::vector<std::string> out;
  for (const auto & s : tuningSignatures(ops)) out.push_back(s.str());
  return out;
}

int PlanModel::refreshExpected(const std::vector<std::string> & ops)
{
  auto want = [&](const std::string & op) {
    return ops.empty() || std::find(ops.begin(), ops.end(), op) != ops.end();
  };
  int n = 0;
  auto & es = tuning_.entries();
  for (auto & kv : es)
  {
    TuningEntry & e = kv.second;
    const std::string & key = kv.first;
    const size_t bar = key.find('|');
    if (bar == std::string::npos) continue;
    const std::string op = key.substr(0, bar);
    const std::string body = key.substr(bar + 1);
    OpSignature sig;
    bool have = false;
    // 从缓存 key 反解签名（覆盖「当前 plan 里已不存在」的历史条目，否则它们的
    // ratio 会永远停在旧标准上）。只处理 GEMM 族——本轮修正的范围。
    if (op == "gemm" && want("gemm")) {
      int M = 0, N = 0, K = 0, act = 0;
      if (std::sscanf(body.c_str(), "M%dN%dK%d_act%d", &M, &N, &K, &act) >= 3) {
        sig = OpSignature::gemm(M, N, K, act);
        have = true;
      }
    } else if (op == "conv1x1" && want("conv1x1")) {
      int C = 0, N = 0, K = 0;
      if (std::sscanf(body.c_str(), "Cout%d_N%d_Cin%d", &C, &N, &K) == 3) {
        sig = OpSignature::conv1x1(C, N, K, 0, 0);
        have = true;
      }
    } else if (op == "conv1x1_cat4" && want("conv1x1_cat4")) {
      int C = 0, N = 0, K = 0;
      if (std::sscanf(body.c_str(), "Cout%d_N%d_Cin%d", &C, &N, &K) == 3) {
        sig = OpSignature::conv1x1Cat4(C, N, K, 0, 0, 0, 0, nullptr, 0, 0);
        have = true;
      }
    } else if (op == "conv3x3" && want("conv3x3")) {
      // R39: refresh conv3x3 too, using the *winner family* ceiling (consistent with
      // autotuneOp). Previously only gemm/conv1x1 were refreshed, so a ceiling-model
      // fix for conv3x3 could not be applied without a full GPU retune. The key body
      // is "W<w>H<h>s<stride>p<pad>_Cin<cin>_Cout<cout>_act<a>_f16[#...]".
      int w = 0, h = 0, st = 0, pd = 0, cin = 0, cout = 0, act = 0;
      if (std::sscanf(body.c_str(), "W%dH%ds%dp%d_Cin%d_Cout%d_act%d",
                      &w, &h, &st, &pd, &cin, &cout, &act) == 7) {
        sig = OpSignature::conv3x3(w, h, st, pd, cin, cout, act);
        have = true;
      }
    }
    if (!have) continue;
    // Prefer the winner family's own ceiling (matches autotuneOp); fall back to the
    // global middle standard for families without one.
    e.expected = expectedOps(sig, rt_.info());
    const KernelFamily * fam = familyByName(e.kernel);
    if (fam && fam->ceiling)
    {
      const double famCeil = fam->ceiling(sig, rt_.info());
      // R47-L3: conv/gemm 软标尺同时受「族计算上限」与「内存 roofline（含 L3 断崖）」约束。
      const bool memBind = (sig.op == "conv3x3" || sig.op == "gemm" || sig.op == "conv1x1" ||
                            sig.op == "conv1x1_cat4");
      e.expected = memBind ? std::min(e.expected, famCeil) : famCeil;
    }
    e.hard_ceiling = (fam && fam->hardCeiling) ? fam->hardCeiling(sig, rt_.info()) : e.expected;
    if (e.hard_ceiling <= 0.0) e.hard_ceiling = e.expected;
    if (e.expected > 0.0) e.ratio = e.ops / e.expected;
    e.hard_ratio = e.hard_ceiling > 0.0 ? e.ops / e.hard_ceiling : 0.0;
    ++n;
  }
  return n;
}

std::map<std::string, TuningEntry> PlanModel::autotune(
  const std::vector<std::string> & ops, const std::string & onlySubstr, int limit, int iters,
  bool merge, bool verbose, bool retune)
{
  std::map<std::string, TuningEntry> done;
  int n_tuned = 0;
  cand_short_.clear();  // R44: 每次隔离扫描重建短名单
  std::map<std::string, int> sig_seen;  // 同一签名只调一次（plan 里大量层共享签名）
  // R45: 整网 retune 的**分批进度标记**用 per-plan 工件承载：本图已写入 plan_overrides_ 的
  // 节点（即已整网回验过）直接跳过。这样 `--retune --limit N` 的多个独立批进程能自然推进，
  // 而不是每批都从第一个签名重来（`INFVINO_GLOBAL_PROGRESS=1` 由安全驱动设置）。
  const bool planProgress = std::getenv("INFVINO_GLOBAL_PROGRESS") != nullptr;
  // 返回 true 表示该签名还未调过（并登记）；false 表示跳过。
  // 已在本进程调过、或缓存里已有 source=="tuned" 的条目 → 跳过（让分批调用自然推进）。
  // retune=true 时忽略已有 tuned 条目（用于候选集/中间标准更新后重扫）。
  auto shouldTune = [&](const OpSignature & sig, const Node & n) {
    const std::string k = sig.str();
    if (sig_seen.count(k)) return false;
    sig_seen[k] = 1;
    if (!retune)
    {
      if (const TuningEntry * e = tuning_.lookup(sig))
        if (e->source == "tuned") return false;
    }
    else if (planProgress && !n.outs.empty() &&
             plan_overrides_.lookup(planNodeKey(n.outs[0])))
    {
      return false;  // 本图已整网回验过 → 跳过（分批推进）
    }
    return true;
  };

  for (const auto & n : nodes_)
  {
    if (limit > 0 && n_tuned >= limit) break;

    // 每个可调优 op 一条分支；签名/候选/几何与 run() 保持一致。
    if (n.op == "conv3x3" && opInList(ops, "conv3x3"))
    {
      const int stride = attrInt(n, "stride", 1), pad = attrInt(n, "pad", 1),
                act = attrInt(n, "act", 0);
      const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
      const auto & id = ref(n.ins[0]).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      const int Cin = static_cast<int>(id[base]), H = static_cast<int>(id[base + 1]),
                W = static_cast<int>(id[base + 2]);
      const auto & od = ref(n.outs[0]).dims;
      const int Cout = static_cast<int>(od[od.size() >= 3 ? od.size() - 3 : 0]);
      cl_mem dx = ref(n.ins[0]).mem, dy = ref(n.outs[0]).mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? ref(n.ins[2]).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? ref(n.ins[3]).mem : nullptr;
      const std::string tag = "conv3x3" + std::to_string(Wout) + "x" + std::to_string(Hout);
      if (!onlySubstr.empty() && tag.find(onlySubstr) == std::string::npos) continue;

      const OpSignature sig = OpSignature::conv3x3(Wout, Hout, stride, pad, Cin, Cout, act);
      if (!shouldTune(sig, n)) continue;
      const double flops = 2.0 * Cout * static_cast<double>(Hout) * Wout * Cin * 9.0;
      const std::vector<Candidate> cands = candidatesConv3x3(sig);

      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        if (c.kernel == "conv3x3_ov") {
          cl_kernel kk = getKernel("conv_ov", "conv3x3_ov", c.options);
          cl_mem dw = ovWeight(n.ins[1], ref(n.ins[1]), Cout, Cin);
          int obw = 8, obh = 2, slm = 1;
          auto p = c.options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(c.options.c_str() + p + 6);
          p = c.options.find("-DOBH=");
          if (p != std::string::npos) obh = std::atoi(c.options.c_str() + p + 6);
          auto ps = c.options.find("-DSLM_DIV=");
          if (ps != std::string::npos) slm = std::atoi(c.options.c_str() + ps + 10);  // len==10
          if (slm < 1) slm = 1;
          setArg(kk, 0, sizeof(dx), &dx);
          setArg(kk, 1, sizeof(dw), &dw);
          setArg(kk, 2, sizeof(db), &db);
          setArg(kk, 3, sizeof(dres), &dres);
          setArg(kk, 4, sizeof(dy), &dy);
          setArg(kk, 5, sizeof(Cin), &Cin);
          setArg(kk, 6, sizeof(H), &H);
          setArg(kk, 7, sizeof(W), &W);
          setArg(kk, 8, sizeof(Cout), &Cout);
          setArg(kk, 9, sizeof(Hout), &Hout);
          setArg(kk, 10, sizeof(Wout), &Wout);
          const size_t lws[3] = {1, 1, static_cast<size_t>(16 * slm)};
          const size_t gws[3] = {
            static_cast<size_t>((Wout + obw - 1) / obw),
            static_cast<size_t>((Hout + obh - 1) / obh),
            static_cast<size_t>((((Cout + 1) / 2) + 15) / 16) * 16 * static_cast<size_t>(slm)};
          return [this, kk, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
          };
        }
        if (c.kernel == "conv3x3_blk") {
          int obw = 8;
          auto p = c.options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(c.options.c_str() + p + 6);
          int slm = 1;
          auto ps = c.options.find("-DSLM_DIV=");
          if (ps != std::string::npos) slm = std::atoi(c.options.c_str() + ps + 10);  // len==10
          if (slm < 1) slm = 1;
          cl_kernel kk = getKernel("conv_blk", "conv3x3_blk", c.options);
          cl_mem dw = blkWeight(n.ins[1], ref(n.ins[1]), Cout, Cin);
          cl_mem dxb = blkInput(n.ins[0], ref(n.ins[0]), Cin, H, W);
          setArg(kk, 0, sizeof(dxb), &dxb);
          setArg(kk, 1, sizeof(dw), &dw);
          setArg(kk, 2, sizeof(db), &db);
          setArg(kk, 3, sizeof(dy), &dy);
          setArg(kk, 4, sizeof(Cin), &Cin);
          setArg(kk, 5, sizeof(H), &H);
          setArg(kk, 6, sizeof(W), &W);
          setArg(kk, 7, sizeof(Cout), &Cout);
          setArg(kk, 8, sizeof(Hout), &Hout);
          setArg(kk, 9, sizeof(Wout), &Wout);
          const size_t lws[3] = {1, static_cast<size_t>(16 * slm), 1};
          const size_t gws[3] = {
            static_cast<size_t>((Wout + obw - 1) / obw) * static_cast<size_t>(Hout),
            static_cast<size_t>(((Cout + 15) / 16) * 16 * slm), 1};
          return [this, kk, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
          };
        }
        if (c.kernel == "conv3x3_cin3") {
          cl_kernel kk = getKernel("conv_cin3", "conv3x3_cin3", c.options);
          cl_mem dw = cin3Weight(n.ins[1], ref(n.ins[1]), Cout, Cin);
          setArg(kk, 0, sizeof(dx), &dx);
          setArg(kk, 1, sizeof(dw), &dw);
          setArg(kk, 2, sizeof(db), &db);
          setArg(kk, 3, sizeof(dy), &dy);
          setArg(kk, 4, sizeof(H), &H);
          setArg(kk, 5, sizeof(W), &W);
          setArg(kk, 6, sizeof(Hout), &Hout);
          setArg(kk, 7, sizeof(Wout), &Wout);
          const size_t lws[2] = {128, 1};
          const size_t gws[2] = {
            (static_cast<size_t>(Wout) + 127) / 128 * 128, static_cast<size_t>(Hout)};
          return [this, kk, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kk, 2, gws, lws);
          };
        }
        // native
        cl_kernel kk = getKernel("conv", c.kernel, c.options);
        cl_mem dw = ref(n.ins[1]).mem;
        auto optInt = [&](const char * k, int def) {
          const auto p = c.options.find(k);
          return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
        };
        const int TX = optInt("-DTX=", 40), TY = optInt("-DTY=", 8), TM = optInt("-DTM=", 1),
                  CB = optInt("-DCB=", 32);
        setArg(kk, 0, sizeof(dx), &dx);
        setArg(kk, 1, sizeof(dw), &dw);
        setArg(kk, 2, sizeof(db), &db);
        setArg(kk, 3, sizeof(dy), &dy);
        setArg(kk, 4, sizeof(Cin), &Cin);
        setArg(kk, 5, sizeof(H), &H);
        setArg(kk, 6, sizeof(W), &W);
        setArg(kk, 7, sizeof(Cout), &Cout);
        setArg(kk, 8, sizeof(Hout), &Hout);
        setArg(kk, 9, sizeof(Wout), &Wout);
        const size_t lws[3] = {
          static_cast<size_t>(TX / TM), static_cast<size_t>(TY), 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + TX - 1) / TX) * lws[0],
          static_cast<size_t>((Hout + TY - 1) / TY) * lws[1],
          static_cast<size_t>((Cout + CB - 1) / CB)};
        return [this, kk, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
        };
      };

      // R38: bench the blk and non-blk groups separately so the plan-time joint fixpoint
      // (resolveLayoutChoices) has both alternatives plus this input's reorder cost.
      std::vector<Candidate> cBlk, cNon;
      for (const auto & c : cands)
        (c.kernel == "conv3x3_blk" ? cBlk : cNon).push_back(c);
      std::vector<TuningEntry> shortNon, shortBlk;
      TuningEntry en = autotuneOp(rt_, sig, cNon, makeEnqueue, flops, iters, &shortNon);
      TuningEntry eb = cBlk.empty() ? TuningEntry()
                                    : autotuneOp(rt_, sig, cBlk, makeEnqueue, flops, iters, &shortBlk);
      // R44: 保留全部测到的候选（blk + 非 blk），供 globalRetune 的整网 busy 回验。
      {
        auto & cs = cand_short_[sig.str()];
        cs.sig = sig;
        cs.cands = shortNon;
        cs.cands.insert(cs.cands.end(), shortBlk.begin(), shortBlk.end());
      }
      // R47 S1: 统一计费口径。conv3x3 的 base 选择此前用 min(blk, non)（**不计 reorder**），
      // 与 conv1x1/depthwise 的 `blk+reorder` 口径不一致——会把「kernel 快但净负」的 blk
      // 写进 base 条目（尤其生产缓存没有 #blk/#non/#reorder 时，fixpoint 退化、无任何
      // reorder 计费）。现统一为先测本节点输入的一趟 reorder，再按 `blk+reorder` 选，
      // 存 kernel-only ms（与 conv1x1/depthwise 完全一致）。
      double reorderMs = 0.0;
      if (!eb.kernel.empty())
      {
        // Measure one bfyx->fsv16 reorder for this node's input (the cost the blk
        // alternative incurs when the input is NOT persisted fsv16).
        Tensor & xt = ref(n.ins[0]);
        const size_t bytes = static_cast<size_t>((Cin + 15) / 16) * H * W * 16 * 2;
        cl_mem scratch = rt_.alloc(bytes, CL_MEM_READ_WRITE);
        cl_kernel kr = getKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
        cl_mem xm = xt.mem;
        setArg(kr, 0, sizeof(xm), &xm);
        setArg(kr, 1, sizeof(scratch), &scratch);
        setArg(kr, 2, sizeof(Cin), &Cin);
        setArg(kr, 3, sizeof(H), &H);
        setArg(kr, 4, sizeof(W), &W);
        const size_t rg[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                              static_cast<size_t>(Cin)};
        std::function<cl_event()> renq = [this, kr, rg]() {
          return ClRuntime::enqueueND(rt_.queue(), kr, 3, rg, nullptr);
        };
        benchCandidate(rt_, renq, iters, &reorderMs);
        clReleaseMemObject(scratch);
      }
      TuningEntry e = en;
      if (!eb.kernel.empty())
      {
        const double eff = eb.ms + reorderMs;   // billed selection cost
        if (en.kernel.empty() || eff < en.ms) e = eb;   // store kernel-only ms
      }
      if (merge && !eb.kernel.empty())
      {
        tuning_.put(OpSignature::custom(sig.str() + "#blk", {}), eb);
        TuningEntry er;
        er.kernel = "reorder_bfyx_to_fsv16";
        er.ms = reorderMs;
        er.device_id = eb.device_id;
        er.source = "tuned";
        tuning_.put(OpSignature::custom(sig.str() + "#reorder", {}), er);
        if (!en.kernel.empty())
          tuning_.put(OpSignature::custom(sig.str() + "#non", {}), en);
      }
      if (!e.kernel.empty()) {
        done[sig.str()] = e;
        ++n_tuned;
        if (merge) tuning_.put(sig, e);
        if (verbose)
          std::printf("  [tune] %-40s best=%-14s %6.3f ms  ops=%5.2f exp=%5.2f ratio=%.2f (%s)\n",
                      sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.expected, e.ratio,
                      e.config.c_str());
      }
      continue;
    }

    if (n.op == "gemm" && opInList(ops, "gemm"))
    {
      const int M = static_cast<int>(ref(n.ins[0]).dims[0]);
      const int K = static_cast<int>(ref(n.ins[0]).dims[1]);
      const int N = static_cast<int>(ref(n.ins[1]).numel() / K);
      cl_mem da = ref(n.ins[0]).mem, db = ref(n.ins[1]).mem, dc = ref(n.outs[0]).mem;
      const OpSignature sig = OpSignature::gemm(M, N, K, 0);
      if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
      if (!shouldTune(sig, n)) continue;
      const double flops = 2.0 * M * N * static_cast<double>(K);
      const std::vector<Candidate> cands = candidatesGemm(sig);
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        cl_kernel kg = getKernel("gemm", "gemm_f16", c.options);
        auto optInt = [&](const char * k, int def) {
          const auto p = c.options.find(k);
          return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
        };
        const int BM = optInt("-DBM=", 128), BN = optInt("-DBN=", 64), TM = optInt("-DTM=", 8),
                  TN = optInt("-DTN=", 4);
        setArg(kg, 0, sizeof(da), &da);
        setArg(kg, 1, sizeof(db), &db);
        setArg(kg, 2, sizeof(dc), &dc);
        setArg(kg, 3, sizeof(M), &M);
        setArg(kg, 4, sizeof(N), &N);
        setArg(kg, 5, sizeof(K), &K);
        const size_t lws[2] = {static_cast<size_t>(BN / TN), static_cast<size_t>(BM / TM)};
        const size_t gws[2] = {
          static_cast<size_t>((N + BN - 1) / BN) * lws[0],
          static_cast<size_t>((M + BM - 1) / BM) * lws[1]};
        return [this, kg, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kg, 2, gws, lws);
        };
      };
      std::vector<TuningEntry> shortAll;
      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters, &shortAll);
      if (auto & cs = cand_short_[sig.str()]; cs.cands.empty())
      {
        cs.sig = sig;
        cs.cands = shortAll;
      }
      if (!e.kernel.empty()) {
        done[sig.str()] = e;
        ++n_tuned;
        if (merge) tuning_.put(sig, e);
        if (verbose)
          std::printf("  [tune] %-40s best=%-14s %6.3f ms  ops=%5.2f exp=%5.2f ratio=%.2f\n",
                      sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.expected, e.ratio);
      }
      continue;
    }

    if (n.op == "conv1x1_cat4" && opInList(ops, "conv1x1_cat4"))
    {
      const int act = attrInt(n, "act", 0);
      auto & w = ref(n.ins[0]);
      const int Cout = static_cast<int>(w.dims[0]);
      const int Cin  = static_cast<int>(w.dims[1]);
      const int ca = attrInt(n, "cat_ca", 0), cb = attrInt(n, "cat_cb", 0),
                cc = attrInt(n, "cat_cc", 0), cd = attrInt(n, "cat_cd", 0);
      const int o0 = attrInt(n, "cat_o0", 0), o1 = attrInt(n, "cat_o1", 0),
                o2 = attrInt(n, "cat_o2", 0), o3 = attrInt(n, "cat_o3", 0);
      const int64_t onumel = ref(n.outs[0]).numel();
      const int HW = (Cout > 0) ? static_cast<int>(onumel / Cout) : 0;
      cl_mem dA = w.mem, dC = ref(n.outs[0]).mem;
      cl_mem dB0 = ref(n.ins[1]).mem, dB1 = ref(n.ins[2]).mem,
             dB2 = ref(n.ins[3]).mem, dB3 = ref(n.ins[4]).mem;
      cl_mem db = (n.ins.size() > 5 && n.ins[5] != "-") ? ref(n.ins[5]).mem : nullptr;
      cl_mem dres = (n.ins.size() > 6 && n.ins[6] != "-") ? ref(n.ins[6]).mem : nullptr;
      const int coff[4] = {o0, o1, o2, o3};
      const OpSignature sig = OpSignature::conv1x1Cat4(Cout, HW, Cin, ca, cb, cc, cd, coff,
                                                       act, dres ? 1 : 0);
      if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
      if (!shouldTune(sig, n)) continue;
      const double flops = 2.0 * Cout * static_cast<double>(HW) * Cin;
      // R48 §4bis P0: 候选谱改由注册表（gemm_cat4_f16 族）给出——不再在此手工复制
      // gemm 谱并追加 -DCAT4=1（单一真相源）；每个候选已带 -DCAT4=1 -DEPI=1 -DACT=。
      const std::vector<Candidate> cands = candidatesFromRegistry(sig);
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        cl_kernel kg = getKernel("gemm", "gemm_f16", c.options);
        auto optInt = [&](const char * k, int def) {
          const auto p = c.options.find(k);
          return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
        };
        const int BM = optInt("-DBM=", 128), BN = optInt("-DBN=", 64), TM = optInt("-DTM=", 8),
                  TN = optInt("-DTN=", 4);
        setArg(kg, 0, sizeof(dA), &dA);
        setArg(kg, 1, sizeof(dB0), &dB0);
        setArg(kg, 2, sizeof(dC), &dC);
        setArg(kg, 3, sizeof(Cout), &Cout);
        setArg(kg, 4, sizeof(HW), &HW);
        setArg(kg, 5, sizeof(Cin), &Cin);
        setArg(kg, 6, sizeof(db), &db);
        setArg(kg, 7, sizeof(dres), &dres);
        setArg(kg, 8, sizeof(dB1), &dB1);
        setArg(kg, 9, sizeof(dB2), &dB2);
        setArg(kg, 10, sizeof(dB3), &dB3);
        setArg(kg, 11, sizeof(ca), &ca);
        setArg(kg, 12, sizeof(cb), &cb);
        setArg(kg, 13, sizeof(cc), &cc);
        setArg(kg, 14, sizeof(o0), &o0);
        setArg(kg, 15, sizeof(o1), &o1);
        setArg(kg, 16, sizeof(o2), &o2);
        setArg(kg, 17, sizeof(o3), &o3);
        const size_t lws[2] = {static_cast<size_t>(BN / TN), static_cast<size_t>(BM / TM)};
        const size_t gws[2] = {
          static_cast<size_t>((HW + BN - 1) / BN) * lws[0],
          static_cast<size_t>((Cout + BM - 1) / BM) * lws[1]};
        return [this, kg, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kg, 2, gws, lws);
        };
      };
      std::vector<TuningEntry> shortAll;
      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters, &shortAll);
      {
        auto & cs = cand_short_[sig.str()];
        cs.sig = sig;
        cs.cands = shortAll;
      }
      if (!e.kernel.empty()) {
        done[sig.str()] = e;
        ++n_tuned;
        if (merge) tuning_.put(sig, e);
        if (verbose)
          std::printf("  [tune] %-40s best=%-16s %6.3f ms  ops=%5.2f exp=%5.2f ratio=%.2f\n",
                      sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.expected, e.ratio);
      }
      continue;
    }

    if (n.op == "conv1x1" && opInList(ops, "conv1x1"))
    {
      const int act = attrInt(n, "act", 0);
      auto & w = ref(n.ins[0]);
      const int Cout = static_cast<int>(w.dims[0]);
      const int Cin  = static_cast<int>(w.dims[1]);
      const int64_t xnumel = ref(n.ins[1]).numel();
      const int N = (Cin > 0) ? static_cast<int>(xnumel / Cin) : 0;
      int Hin = 1, Win = 1;
      {
        const auto & xd = ref(n.ins[1]).dims;
        const size_t xb = xd.size() >= 3 ? xd.size() - 3 : 0;
        if (xd.size() >= 2) { Hin = static_cast<int>(xd[xd.size() - 2]); Win = static_cast<int>(xd.back()); }
        (void)xb;
        if (Hin * Win != N) { Hin = N; Win = 1; }
      }
      cl_mem dw = w.mem, dx = ref(n.ins[1]).mem, dy = ref(n.outs[0]).mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? ref(n.ins[2]).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? ref(n.ins[3]).mem : nullptr;
      // R51 D5: 融合的逐输入通道 scale（SE Mul 折进 prologue）。
      cl_mem dscale = (n.ins.size() > 4 && n.ins[4] != "-") ? ref(n.ins[4]).mem : nullptr;
      const OpSignature sig = conv1x1Sig(n, Cout, N, Cin, act);
      if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
      if (!shouldTune(sig, n)) continue;
      const double flops = 2.0 * Cout * static_cast<double>(N) * Cin;
      const std::vector<Candidate> cands = candidatesConv1x1(sig);
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        if (N == 1) {
          cl_kernel kg = getKernel("conv1x1", "conv1x1_gemv_f16", c.options);
          setArg(kg, 0, sizeof(dw), &dw);
          setArg(kg, 1, sizeof(dx), &dx);
          setArg(kg, 2, sizeof(db), &db);
          setArg(kg, 3, sizeof(dres), &dres);
          setArg(kg, 4, sizeof(dy), &dy);
          setArg(kg, 5, sizeof(Cin), &Cin);
          setArg(kg, 6, sizeof(Cout), &Cout);
          const size_t lws[1] = {16};
          const size_t gws[1] = {static_cast<size_t>(Cout) * 16};
          return [this, kg, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kg, 1, gws, lws);
          };
        }
        if (c.kernel == "conv1x1_blk") {
          // blocked 1x1（fsv16 输入 + osv16 权重）：几何从 options 回放。
          auto optInt = [&](const char * k, int def) {
            const auto p = c.options.find(k);
            return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
          };
          const int xb = optInt("-DX_BLOCK=", 4), slm = optInt("-DSLM_DIV=", 1);
          const int yb = optInt("-DY_BLOCK=", 1);   // R48 D1: 输出行 tiling
          cl_mem dxb = blkInput(n.ins[1], ref(n.ins[1]), Cin, Hin, Win);
          cl_mem dwb = blk1x1Weight(n.ins[0], w, Cout, Cin);
          std::string bopts = c.options;
          if (dres) setResOpt(bopts, true);
          if (dscale) bopts += " -DMUL_SCALE=1";   // R51 D5
          cl_kernel kb = getKernel("conv1x1_blk", "conv1x1_blk", bopts);
          setArg(kb, 0, sizeof(dxb), &dxb);
          setArg(kb, 1, sizeof(dwb), &dwb);
          setArg(kb, 2, sizeof(db), &db);
          setArg(kb, 3, sizeof(dy), &dy);
          setArg(kb, 4, sizeof(dres), &dres);
          setArg(kb, 5, sizeof(Cin), &Cin);
          setArg(kb, 6, sizeof(Hin), &Hin);
          setArg(kb, 7, sizeof(Win), &Win);
          setArg(kb, 8, sizeof(Cout), &Cout);
          if (dscale) setArg(kb, 9, sizeof(dscale), &dscale);
          const size_t lws[3] = {1, static_cast<size_t>(16 * slm), 1};
          const size_t ybCount = static_cast<size_t>((Hin + yb - 1) / yb);  // R48 D1
          const size_t gws[3] = {static_cast<size_t>(((Win + xb - 1) / xb)) * ybCount,
                                 static_cast<size_t>(((Cout + 15) / 16) * lws[1]), 1};
          return [this, kb, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kb, 3, gws, lws);
          };
        }
        const bool sk = (c.kernel == "gemm_sk_f16");
        std::string gopts = c.options;
        if (dscale) gopts += " -DMUL_SCALE=1";   // R51 D5
        cl_kernel kg = getKernel(sk ? "gemm_sk" : "gemm", c.kernel, gopts);
        auto optInt = [&](const char * k, int def) {
          const auto p = c.options.find(k);
          return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
        };
        setArg(kg, 0, sizeof(dw), &dw);
        setArg(kg, 1, sizeof(dx), &dx);
        setArg(kg, 2, sizeof(dy), &dy);
        setArg(kg, 3, sizeof(Cout), &Cout);
        setArg(kg, 4, sizeof(N), &N);
        setArg(kg, 5, sizeof(Cin), &Cin);
        setArg(kg, 6, sizeof(db), &db);
        setArg(kg, 7, sizeof(dres), &dres);
        if (dscale) setArg(kg, 8, sizeof(dscale), &dscale);
        size_t lws[2], gws[2];
        if (sk) {
          const int TM = optInt("-DSK_TM=", 8), TN = optInt("-DSK_TN=", 4),
                    SG = optInt("-DSK_SG=", 16);
          lws[0] = static_cast<size_t>(SG); lws[1] = 1;
          gws[0] = static_cast<size_t>((N + TN - 1) / TN) * lws[0];
          gws[1] = static_cast<size_t>((Cout + TM - 1) / TM);
        } else {
          const int BM = optInt("-DBM=", 128), BN = optInt("-DBN=", 64), TM = optInt("-DTM=", 8),
                    TN = optInt("-DTN=", 4);
          lws[0] = static_cast<size_t>(BN / TN);
          lws[1] = static_cast<size_t>(BM / TM);
          gws[0] = static_cast<size_t>((N + BN - 1) / BN) * lws[0];
          gws[1] = static_cast<size_t>((Cout + BM - 1) / BM) * lws[1];
        }
        return [this, kg, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kg, 2, gws, lws);
        };
      };
      // blocked 1x1：同样把 bfyx->fsv16 输入重排计入成本（输入已持久化时为 0）。
      std::vector<Candidate> xCandsBlk, xCandsNon;
      for (const auto & c : cands)
        (c.kernel == "conv1x1_blk" ? xCandsBlk : xCandsNon).push_back(c);
      std::vector<TuningEntry> xShortNon, xShortBlk;
      TuningEntry eNon = autotuneOp(rt_, sig, xCandsNon, makeEnqueue, flops, iters, &xShortNon);
      TuningEntry e = eNon;
      if (!xCandsBlk.empty()) {
        TuningEntry eb = autotuneOp(rt_, sig, xCandsBlk, makeEnqueue, flops, iters, &xShortBlk);
        if (!eb.kernel.empty()) {
          // R43: no longer a one-way "conservative billing" baked into the entry.
          // Store the two alternatives + the reorder cost separately (like conv3x3)
          // so the plan-time joint (family, layout) fixpoint can decide with exact
          // layout knowledge (input persisted fsv16 => reorder 0).
          double reorderMs = 0.0;
          {
            Tensor & xt = ref(n.ins[1]);
            const size_t bytes = static_cast<size_t>((Cin + 15) / 16) * Hin * Win * 16 * 2;
            cl_mem scratch = rt_.alloc(bytes, CL_MEM_READ_WRITE);
            cl_kernel kr = getKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
            cl_mem xm = xt.mem;
            setArg(kr, 0, sizeof(xm), &xm);
            setArg(kr, 1, sizeof(scratch), &scratch);
            setArg(kr, 2, sizeof(Cin), &Cin);
            setArg(kr, 3, sizeof(Hin), &Hin);
            setArg(kr, 4, sizeof(Win), &Win);
            const size_t rg[3] = {static_cast<size_t>(Win), static_cast<size_t>(Hin),
                                  static_cast<size_t>(Cin)};
            std::function<cl_event()> renq = [this, kr, rg]() {
              return ClRuntime::enqueueND(rt_.queue(), kr, 3, rg, nullptr);
            };
            benchCandidate(rt_, renq, iters, &reorderMs);
            clReleaseMemObject(scratch);
          }
          // R51 M5: conv1x1_blk 在**输出 fsv16** 时的成本（布局契约 canOutFsv16），与
          // depthwise_blk 的 `#blkfsv16` 对应。此前 conv1x1 一直用 bfyx 输出成本给
          // 「输出 fsv16」定价，系统性高估 blocked 族（fsv16 输出把 16 lane 合并写、通常
          // 更快），使布局规划偏保守/失真。仅当输出缓冲被补齐（Cout%16==0 或 capable）
          // 时才测，避免 OUT_FSV16 写越界。
          TuningEntry ebFsv16;
          const std::string & oname = n.outs[0];
          if ((Cout % 16 == 0 || fsv16_capable_.count(oname)) && !eb.options.empty())
          {
            Candidate c2;
            c2.kernel = eb.kernel;
            c2.config = eb.config;
            c2.options = (eb.options.find("-DOUT_FSV16=") == std::string::npos)
                           ? eb.options + " -DOUT_FSV16=1"
                           : eb.options;
            try
            {
              auto enq2 = makeEnqueue(c2);
              double ms2 = 0, sp2 = 0;
              if (benchCandidate(rt_, enq2, iters, &ms2, &sp2) && ms2 > 0.0)
              {
                ebFsv16 = eb;
                ebFsv16.options = c2.options;
                ebFsv16.ms = ms2;
                ebFsv16.ops = rt_.opsPerEuCycle(flops, ms2);
                ebFsv16.source = "tuned";
                ebFsv16.ratio =
                    ebFsv16.expected > 0 ? ebFsv16.ops / ebFsv16.expected : 0.0;
                ebFsv16.hard_ratio =
                    ebFsv16.hard_ceiling > 0 ? ebFsv16.ops / ebFsv16.hard_ceiling : 0.0;
              }
            }
            catch (const std::exception &) {}
          }
          if (merge) {
            tuning_.put(OpSignature::custom(sig.str() + "#blk", {}), eb);
            TuningEntry er;
            er.kernel = "reorder_bfyx_to_fsv16";
            er.ms = reorderMs;
            er.device_id = eb.device_id;
            er.source = "tuned";
            tuning_.put(OpSignature::custom(sig.str() + "#reorder", {}), er);
            if (!eNon.kernel.empty())
              tuning_.put(OpSignature::custom(sig.str() + "#non", {}), eNon);
            if (!ebFsv16.kernel.empty())
              tuning_.put(OpSignature::custom(sig.str() + "#blkfsv16", {}), ebFsv16);
          }
          // Fallback base entry (used when the joint fixpoint has no alternatives):
          // select on the billed cost, store kernel-only ms.
          const double eff = eb.ms + reorderMs;
          if (eNon.kernel.empty() || eff < eNon.ms) e = eb;
        }
      }
      // R44: 保留 conv1x1 的全部候选（non + blk）供整网 busy 回验。
      {
        auto & cs = cand_short_[sig.str()];
        cs.sig = sig;
        cs.cands = xShortNon;
        cs.cands.insert(cs.cands.end(), xShortBlk.begin(), xShortBlk.end());
      }
      if (!e.kernel.empty()) {
        done[sig.str()] = e;
        ++n_tuned;
        if (merge) tuning_.put(sig, e);
        if (verbose)
          std::printf("  [tune] %-40s best=%-16s %6.3f ms  ops=%5.2f exp=%5.2f ratio=%.2f\n",
                      sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.expected, e.ratio);
      }
      continue;
    }

    if (n.op == "conv_general" && opInList(ops, "depthwise"))
    {
      const int K = attrInt(n, "K", 3), S = attrInt(n, "S", 1), P = attrInt(n, "P", 1),
                G = attrInt(n, "G", 1), act = attrInt(n, "act", 0);
      const int Hout = attrInt(n, "Hout", 0), Wout = attrInt(n, "Wout", 0);
      const auto & id = ref(n.ins[0]).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      const int Cin = static_cast<int>(id[base]), H = static_cast<int>(id[base + 1]),
                W = static_cast<int>(id[base + 2]);
      const auto & od = ref(n.outs[0]).dims;
      const int Cout = static_cast<int>(od[od.size() >= 3 ? od.size() - 3 : 0]);
      if (!(G == Cin && (K == 3 || K == 5) && Cin == Cout)) continue;
      cl_mem dx = ref(n.ins[0]).mem, dw = ref(n.ins[1]).mem, dy = ref(n.outs[0]).mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? ref(n.ins[2]).mem : nullptr;
      const OpSignature sig = OpSignature::depthwise(Wout, Hout, S, P, Cin, K, act);
      if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
      if (!shouldTune(sig, n)) continue;
      const double flops = 2.0 * Cin * static_cast<double>(Hout) * Wout * K * K;
      const std::vector<Candidate> cands = candidatesDepthwise(sig);
      // vp 与其它候选分开计时：vp 的真实每帧成本 = depthwise_vp + pad（pad 单独量）。
      // R31 负结果：pad 受带宽墙限制，省下的边界谓词 ≈ pad 成本，整网基本持平；因此
      // vp 候选默认不进入候选集，只有 INFVINO_DW_PAD 打开时才参与（见 docs/kernel.md）。
      std::vector<Candidate> candsNoPad, candsPad;
      for (const auto & c : cands)
        (c.kernel == "depthwise_vp" ? candsPad : candsNoPad).push_back(c);
      int    Hp = H + 2 * P, Wpad = 0;
      cl_mem xp = nullptr;
      double padMs = 0.0;
      if (!candsPad.empty()) {
        // 为零边输入一次性写零（尺寸按最大 TW），并单独量一趟 pad。
        xp = dwPadInput(n.ins[0], Cin, H, W, K, S, P, &Hp, &Wpad);
        cl_kernel kp = getKernel("conv_general", "depthwise_pad", "");
        setArg(kp, 0, sizeof(dx), &dx);
        setArg(kp, 1, sizeof(xp), &xp);
        setArg(kp, 2, sizeof(Cin), &Cin);
        setArg(kp, 3, sizeof(H), &H);
        setArg(kp, 4, sizeof(W), &W);
        setArg(kp, 5, sizeof(Hp), &Hp);
        setArg(kp, 6, sizeof(Wpad), &Wpad);
        setArg(kp, 7, sizeof(P), &P);
        const size_t gpad[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                                static_cast<size_t>(Cin)};
        std::function<cl_event()> padEnq = [this, kp, gpad]() {
          return ClRuntime::enqueueND(rt_.queue(), kp, 3, gpad, nullptr);
        };
        benchCandidate(rt_, padEnq, iters, &padMs);
      }
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        if (c.kernel == "depthwise_blk") {
          auto optInt = [&](const char * key, int def) {
            const auto p = c.options.find(key);
            return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(key));
          };
          const int xb = optInt("-DX_BLOCK=", 8);
          const int yb = optInt("-DY_BLOCK=", 1);
          cl_mem dxb = blkInput(n.ins[0], ref(n.ins[0]), Cin, H, W);
          cl_mem dwb = blkDwWeight(n.ins[1], ref(n.ins[1]), Cin, K);
          cl_kernel kd = getKernel("depthwise_blk", "depthwise_blk", c.options);
          setArg(kd, 0, sizeof(dxb), &dxb);
          setArg(kd, 1, sizeof(dwb), &dwb);
          setArg(kd, 2, sizeof(db), &db);
          setArg(kd, 3, sizeof(dy), &dy);
          setArg(kd, 4, sizeof(Cin), &Cin);
          setArg(kd, 5, sizeof(H), &H);
          setArg(kd, 6, sizeof(W), &W);
          int ho = Hout, wo = Wout;
          setArg(kd, 7, sizeof(ho), &ho);
          setArg(kd, 8, sizeof(wo), &wo);
          const size_t g[3] = {static_cast<size_t>(((Wout + xb - 1) / xb) * ((Hout + yb - 1) / yb)),
                               static_cast<size_t>(((Cin + 15) / 16) * 16), 1};
          const size_t l[3] = {1, 16, 1};
          return [this, kd, g, l]() {
            return ClRuntime::enqueueND(rt_.queue(), kd, 3, g, l);
          };
        }
        cl_kernel kd = getKernel("conv_general", c.kernel, c.options);
        int ho = Hout, wo = Wout;
        if (c.kernel == "depthwise_vp") {
          setArg(kd, 0, sizeof(xp), &xp);
          setArg(kd, 1, sizeof(dw), &dw);
          setArg(kd, 2, sizeof(db), &db);
          setArg(kd, 3, sizeof(dy), &dy);
          setArg(kd, 4, sizeof(Cin), &Cin);
          setArg(kd, 5, sizeof(Hp), &Hp);
          setArg(kd, 6, sizeof(Wpad), &Wpad);
          setArg(kd, 7, sizeof(ho), &ho);
          setArg(kd, 8, sizeof(wo), &wo);
          auto p = c.options.find("-DDW_TW=");
          const int tw = p == std::string::npos ? 8 : std::atoi(c.options.c_str() + p + 8);
          const size_t g[3] = {static_cast<size_t>((Wout + tw - 1) / tw),
                               static_cast<size_t>(Hout), static_cast<size_t>(Cin)};
          return [this, kd, g]() {
            return ClRuntime::enqueueND(rt_.queue(), kd, 3, g, nullptr);
          };
        }
        setArg(kd, 0, sizeof(dx), &dx);
        setArg(kd, 1, sizeof(dw), &dw);
        setArg(kd, 2, sizeof(db), &db);
        setArg(kd, 3, sizeof(dy), &dy);
        setArg(kd, 4, sizeof(Cin), &Cin);
        setArg(kd, 5, sizeof(H), &H);
        setArg(kd, 6, sizeof(W), &W);
        setArg(kd, 7, sizeof(ho), &ho);
        setArg(kd, 8, sizeof(wo), &wo);
        if (c.kernel == "depthwise_v") {
          auto p = c.options.find("-DDW_TW=");
          const int tw = p == std::string::npos ? 4 : std::atoi(c.options.c_str() + p + 8);
          const size_t g[3] = {static_cast<size_t>((Wout + tw - 1) / tw),
                               static_cast<size_t>(Hout), static_cast<size_t>(Cin)};
          return [this, kd, g]() {
            return ClRuntime::enqueueND(rt_.queue(), kd, 3, g, nullptr);
          };
        }
        const size_t gdw[1] = {static_cast<size_t>(Cin) * Hout * Wout};
        return [this, kd, gdw]() {
          return ClRuntime::enqueueND(rt_.queue(), kd, 1, gdw, nullptr);
        };
      };
      // blocked depthwise：kernel 之外还多一趟 bfyx->fsv16 输入重排。把它计入成本，
      // 避免「kernel 略快但每帧多付 reorder」的 net 负收益（depthwise 的 reorder 相对
      // kernel 很大，与 conv3x3_blk 不同）。若输入已被持久化为 fsv16 则成本为 0。
      std::vector<Candidate> candsBlk, candsNonBlk;
      for (const auto & c : candsNoPad)
        (c.kernel == "depthwise_blk" ? candsBlk : candsNonBlk).push_back(c);
      std::vector<TuningEntry> dShortNon, dShortBlk;
      TuningEntry eNon = autotuneOp(rt_, sig, candsNonBlk, makeEnqueue, flops, iters, &dShortNon);
      TuningEntry e = eNon;
      if (!candsBlk.empty()) {
        TuningEntry eb = autotuneOp(rt_, sig, candsBlk, makeEnqueue, flops, iters, &dShortBlk);
        if (!eb.kernel.empty()) {
          // R43：不再只做单向「保守计费」；与 conv3x3/conv1x1 一样把 blk/non/reorder
          // 分开存，交给联合 (族,布局) 不动点按实际持久化精确判定。
          double reorderMs = 0.0;
          {
            Tensor & xt = ref(n.ins[0]);
            const size_t bytes = static_cast<size_t>((Cin + 15) / 16) * H * W * 16 * 2;
            cl_mem scratch = rt_.alloc(bytes, CL_MEM_READ_WRITE);
            cl_kernel kr = getKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
            cl_mem xm = xt.mem;
            setArg(kr, 0, sizeof(xm), &xm);
            setArg(kr, 1, sizeof(scratch), &scratch);
            setArg(kr, 2, sizeof(Cin), &Cin);
            setArg(kr, 3, sizeof(H), &H);
            setArg(kr, 4, sizeof(W), &W);
            const size_t rg[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                                  static_cast<size_t>(Cin)};
            std::function<cl_event()> renq = [this, kr, rg]() {
              return ClRuntime::enqueueND(rt_.queue(), kr, 3, rg, nullptr);
            };
            benchCandidate(rt_, renq, iters, &reorderMs);
            clReleaseMemObject(scratch);
          }
          // R50: 同一 blk 配置在**输出 fsv16** 时的成本（布局契约 canOutFsv16）。depthwise_blk
          // 的 fsv16 输出把 16 lane（通道）合并为连续写，实测比 bfyx 输出快 ~2×；布局规划据此
          // 给「输出 fsv16」的节点定价（bfyx 成本仍用于输出 NCHW 的情形）。R51：分配池已按
          // 补齐通道分配，故 C%16!=0 也能量/走 fsv16 —— 但仅当输出缓冲确实被补齐（capable），
          // 否则 OUT_FSV16 会越界（输出永远无法被标记 fsv16 时也无需测其成本）。
          const std::string & oname = n.outs[0];
          TuningEntry ebFsv16;
          if ((Cin % 16 == 0 || fsv16_capable_.count(oname)) && !eb.options.empty())
          {
            Candidate c2;
            c2.kernel = eb.kernel;
            c2.config = eb.config;
            c2.options = (eb.options.find("-DOUT_FSV16=") == std::string::npos)
                           ? eb.options + " -DOUT_FSV16=1"
                           : eb.options;
            try
            {
              auto enq2 = makeEnqueue(c2);
              double ms2 = 0, sp2 = 0;
              if (benchCandidate(rt_, enq2, iters, &ms2, &sp2) && ms2 > 0.0)
              {
                ebFsv16 = eb;
                ebFsv16.options = c2.options;
                ebFsv16.ms = ms2;
                ebFsv16.ops = rt_.opsPerEuCycle(flops, ms2);
                ebFsv16.source = "tuned";
                ebFsv16.ratio =
                    ebFsv16.expected > 0 ? ebFsv16.ops / ebFsv16.expected : 0.0;
                ebFsv16.hard_ratio =
                    ebFsv16.hard_ceiling > 0 ? ebFsv16.ops / ebFsv16.hard_ceiling : 0.0;
              }
            }
            catch (const std::exception &) {}
          }
          if (merge) {
            tuning_.put(OpSignature::custom(sig.str() + "#blk", {}), eb);
            TuningEntry er;
            er.kernel = "reorder_bfyx_to_fsv16";
            er.ms = reorderMs;
            er.device_id = eb.device_id;
            er.source = "tuned";
            tuning_.put(OpSignature::custom(sig.str() + "#reorder", {}), er);
            if (!eNon.kernel.empty())
              tuning_.put(OpSignature::custom(sig.str() + "#non", {}), eNon);
            if (!ebFsv16.kernel.empty())
              tuning_.put(OpSignature::custom(sig.str() + "#blkfsv16", {}), ebFsv16);
          }
          const double eff = eb.ms + reorderMs;   // billed selection cost
          if (eNon.kernel.empty() || eff < eNon.ms) e = eb;   // store kernel-only ms
        }
      }
      // R44: 保留 depthwise 全部候选（non + blk）供整网 busy 回验。pad 变体（opt-in）语义
      // 上多一趟 pad dispatch，不直接放进短名单。
      {
        auto & cs = cand_short_[sig.str()];
        cs.sig = sig;
        cs.cands = dShortNon;
        cs.cands.insert(cs.cands.end(), dShortBlk.begin(), dShortBlk.end());
      }
      if (!candsPad.empty()) {
        TuningEntry ev = autotuneOp(rt_, sig, candsPad, makeEnqueue, flops, iters);
        if (!ev.kernel.empty()) {
          const double eff = ev.ms + padMs;  // 加上每帧 pad 的一趟
          if (e.kernel.empty() || eff < e.ms) {
            ev.ms = eff;
            ev.ops = rt_.opsPerEuCycle(flops, eff);
            ev.ratio = ev.expected > 0 ? ev.ops / ev.expected : 0.0;
            e = ev;
          }
        }
      }
      if (!e.kernel.empty()) {
        done[sig.str()] = e;
        ++n_tuned;
        if (merge) tuning_.put(sig, e);
        if (verbose)
          std::printf("  [tune] %-40s best=%-16s %6.3f ms  ops=%5.2f exp=%5.2f ratio=%.2f\n",
                      sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.expected, e.ratio);
      }
      continue;
    }

    // Round 28: 小算子（launch/带宽受限）自动调优。所有候选都是**数值等价**的
    // 变体（逐元素表达式/归约顺序不变），只是每 work-item 处理多少元素或网格维度
    // 不同，所以调优只改性能、不改数值。flops 统一取 2·元素数 作为「越小越快」的
    // 单调代理；autotuneOp 实际按最小 ms 选优。
    {
      const bool smallOp =
        (n.op == "ew_binary" || n.op == "ew_unary" || n.op == "copy_c" ||
         n.op == "slice_axis" || n.op == "concat4" || n.op == "maxpool" ||
         n.op == "resize_nn" || n.op == "permute_0213" || n.op == "bmm" ||
         n.op == "softmax_axis" || n.op == "gap");
      if (smallOp && opInList(ops, n.op))
      {
        const OpSignature sig = smallSig(n);
        if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
        if (!shouldTune(sig, n)) continue;
        const std::vector<Candidate> cands = candidatesSmall(sig);
        if (cands.empty()) continue;
        const double flops = 2.0 * static_cast<double>(ref(n.outs[0]).numel());
        auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
          cl_kernel   k = getKernel("ops", c.kernel, c.options);
          cl_uint     dim;
          size_t      gws[3], lws[3];
          bool        useLws;
          smallLaunch(n, k, c.kernel, c.options, dim, gws, lws, useLws);
          return [this, k, dim, gws, lws, useLws]() {
            return ClRuntime::enqueueND(rt_.queue(), k, dim, gws, useLws ? lws : nullptr);
          };
        };
        std::vector<TuningEntry> sShort;
        TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters, &sShort);
        {
          auto & cs = cand_short_[sig.str()];
          cs.sig = sig;
          cs.cands = sShort;
        }
        if (!e.kernel.empty()) {
          done[sig.str()] = e;
          ++n_tuned;
          if (merge) tuning_.put(sig, e);
          if (verbose)
            std::printf("  [tune] %-40s best=%-18s %6.3f ms  ops=%5.2f ratio=%.2f (%s)\n",
                        sig.str().c_str(), e.kernel.c_str(), e.ms, e.ops, e.ratio,
                        e.config.c_str());
        }
        continue;
      }
    }
  }

  // P2: 调优改变了 kernel/options ⇒ 已录制的 dispatch 失效，下一次 run() 重新录制。
  if (merge && !done.empty()) invalidateCapture();

  // tuning 改变后同步布局（联合选择的一步：布局契约随选中的族变化）。
  if (n_tuned > 0) resolveLayoutChoices();
  return done;
}

// ---------------------------------------------------------------------------
// R44: 整网 busy 坐标下降回验（修正「隔离 min ≠ 全局最优」）
// ---------------------------------------------------------------------------
int PlanModel::globalRetune(const std::vector<std::string> & ops, int iters, int topK, int rounds,
                            int limit, double margin, int budget)
{
  if (!profiling_)
  {
    std::fprintf(stderr, "[global-retune] needs profiling=true; skipped\n");
    return 0;
  }
  if (cand_short_.empty())
  {
    std::fprintf(stderr, "[global-retune] no candidate shortlist; run isolated autotune first\n");
    return 0;
  }
  if (topK < 1) topK = 1;
  if (rounds < 1) rounds = 1;
  const int reps = (iters > 0) ? iters : 3;   // 每个 assignment 的整网测量次数（取 min busy）
  const double kMinGain = 0.01;               // R45 P2#8: 噪声地板 1%（noise_check 典型 ~1.5%）
  // R47 S2: 布局契约门 —— 候选的**可加目标预测净**（含 reorder）不得比现状预测更差超过此值。
  // 只用于**拒绝**：防止「预测净变差、但端到端被噪声抬成改善」的候选被接受（R46 的
  // depthwise_v→blk 即此类：多付 reorder 却过了 1% 门）。不改变「预测更好才考虑」的候选集。
  const double kPredTol = 0.01;

  auto opWanted = [&](const std::string & op) {
    // R45：所有 autotune 处理的族在 run() 里都已统一走 choiceEntry（per-node 覆盖），
    // 因此整网回验对全部可调算子生效，不再限定 conv3x3/conv1x1/depthwise。
    return ops.empty() || std::find(ops.begin(), ops.end(), op) != ops.end();
  };

  // 目标签名 = 短名单里能在本 plan 里找到节点的签名（按隔离 ms 取 top-K）。
  struct Target { CandidateSet cs; std::vector<size_t> nodes; };
  std::vector<Target> targets;
  for (auto & kv : cand_short_)
  {
    CandidateSet & cs = kv.second;
    if (cs.cands.empty() || !opWanted(cs.sig.op)) continue;
    std::vector<size_t> nodes;
    for (size_t ni = 0; ni < nodes_.size(); ++ni)
    {
      bool ok = false;
      const OpSignature s = nodeSignature(nodes_[ni], &ok);
      if (ok && s.str() == kv.first) nodes.push_back(ni);
    }
    if (nodes.empty()) continue;
    std::stable_sort(cs.cands.begin(), cs.cands.end(),
                     [](const TuningEntry & a, const TuningEntry & b) { return a.ms < b.ms; });
    // R45 P2#8: margin 剪枝 —— 隔离期就明显更慢的候选不可能在整网里翻盘（同一布局/耦合），
    // 直接剔除，缩小「组合爆炸」的搜索空间。
    if (margin > 0.0 && !cs.cands.empty())
    {
      const double cut = cs.cands[0].ms * (1.0 + margin);
      size_t keep = 1;
      while (keep < cs.cands.size() && cs.cands[keep].ms <= cut) ++keep;
      cs.cands.resize(keep);
    }
    // R45: 短名单 = 隔离 top-K **∪ 每个 kernel 族的最优**。隔离 top-K 可能整体漏掉某个族
    // （R43：conv1x1_blk 因 reorder 在隔离期落后，若前 K 全被 gemm 变体占据就会被漏掉）；
    // 每族保留一个代表，保证「整网翻盘」的族始终在搜索空间里。额外族数封顶 kExtraFamilies。
    if (!cs.cands.empty())
    {
      const int kExtraFamilies = 4;
      std::vector<TuningEntry> sel;
      std::vector<std::string> fams;
      auto famOf = [](const TuningEntry & e) {
        // 归一化到「族」：去掉配置后缀，只保留 kernel 名（已是族/变体粒度）。
        return e.kernel;
      };
      const int base = std::min<int>(topK, static_cast<int>(cs.cands.size()));
      for (int i = 0; i < base; ++i) { sel.push_back(cs.cands[i]); fams.push_back(famOf(cs.cands[i])); }
      int extra = 0;
      for (const auto & c : cs.cands)
      {
        if (extra >= kExtraFamilies) break;
        const std::string f = famOf(c);
        if (std::find(fams.begin(), fams.end(), f) != fams.end()) continue;
        sel.push_back(c);
        fams.push_back(f);
        ++extra;
      }
      cs.cands = std::move(sel);
    }
    else if (static_cast<int>(cs.cands.size()) > topK)
      cs.cands.resize(static_cast<size_t>(topK));
    Target t;
    t.cs = cs;
    t.nodes = std::move(nodes);
    targets.push_back(std::move(t));
    if (limit > 0 && static_cast<int>(targets.size()) >= limit) break;
  }
  if (targets.empty()) return 0;

  // 整网输入：零填充（卷积耗时对数值不敏感；只为给 run() 一个确定输入）。
  if (!input_name_.empty() && inputNumel() > 0)
  {
    std::vector<uint16_t> zeros(inputNumel(), 0);
    setInput(zeros.data());
  }

  // per-node 赋值：初始为构造期（联合不动点）的实际选择。测量走直接赋值 + planBlockedLayout，
  // 不走 resolveLayoutChoices —— 否则不动点会用隔离 #blk/#non 覆盖我们正在回验的候选。
  std::vector<TuningEntry> assign(nodes_.size());
  for (auto & t : targets)
    for (size_t ni : t.nodes)
    {
      if (const TuningEntry * ce = choiceEntry(ni, t.cs.sig)) assign[ni] = *ce;
      else if (const TuningEntry * b = tuning_.lookup(t.cs.sig)) assign[ni] = *b;
    }
  const std::vector<TuningEntry> baseline = assign;  // R45: 最终验收门的对照

  int evals = 0;   // 整网测量批次数（预算控制；每批 ≈ 一个 assignment 的 reps 次执行）
  const bool groupedOn = std::getenv("INFVINO_NO_GLOBAL_GROUPED") == nullptr;
  // R47 §5.1/§5.2: in-situ **双口径（min + median）+ 交错**，替代 R44 的单一 min。
  //   min    —— 内禀地板（R42 口径）：外部干扰只会加时间，min 稳，但会低估稳态典型值；
  //   median —— 典型口径：与外部稳态 A/B（kernel_run --report）一致；
  //   交错    —— 同一 rep 内先后测 base/candidate，抵消热漂移（R46：同 assignment 前后差 ±5%）。
  // 改善必须在 **median** 口径成立；min 只做地板守卫与报告。
  struct NetStat { double mn = 1e300, med = 1e300, spread = 0.0; bool ok = false; };
  auto statsOf = [](std::vector<double> & v) -> NetStat {
    NetStat s;
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    s.mn = v.front();
    s.med = v[v.size() / 2];
    s.spread = s.mn > 0.0 ? (v.back() - s.mn) / s.mn : 0.0;
    s.ok = (s.mn < 1e299);
    return s;
  };
  // 测一个 assignment（capture 后采样 nreps 次），返回 min/median 双口径。
  auto measureAssign = [&](const std::vector<TuningEntry> & asg, int nreps) -> NetStat {
    node_choice_ = asg;
    invalidateCapture();
    planBlockedLayout();
    evals += nreps + 1;   // R47: 预算按**整网执行次数**计（含 capture），更贴近 GPU 风险口径
    std::vector<double> samples;
    try
    {
      for (int r = 0; r <= nreps; ++r)
      {
        clearProfile();
        run();
        if (r == 0) continue;  // 第 1 次是 capture/热身
        const double b = profile().busy_ms;
        if (b > 0.0) samples.push_back(b);
      }
    }
    catch (const std::exception & ex)
    {
      if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
        std::fprintf(stderr, "[global-retune] measure rejected: %s\n", ex.what());
      return NetStat{};
    }
    return statsOf(samples);
  };
  // 交错测 A/B：同一 rep 内先后测两者（各自先 capture 再采样），抵消热漂移。
  auto measurePair = [&](const std::vector<TuningEntry> & A, const std::vector<TuningEntry> & B,
                         int nreps, NetStat * sa, NetStat * sb) {
    std::vector<double> as, bs;
    auto one = [&](const std::vector<TuningEntry> & asg, std::vector<double> & out) {
      node_choice_ = asg;
      invalidateCapture();
      planBlockedLayout();
      try
      {
        clearProfile();
        run();  // capture/热身（丢弃）
        clearProfile();
        run();  // 采样
        const double b = profile().busy_ms;
        if (b > 0.0) out.push_back(b);
      }
      catch (const std::exception &) { out.clear(); }
    };
    for (int r = 0; r < nreps; ++r) { one(A, as); one(B, bs); evals += 4; }  // 2 次执行/状态/rep
    *sa = statsOf(as);
    *sb = statsOf(bs);
  };

  // R47 step3: **全局 L3 模拟（LRU）** —— 占用会引入强耦合（A 的瞬时工作集逐出 B 的输入），
  // 因此**不能**用逐算子独立项（会二次计费、且破坏可加性），而必须**按节点顺序跑一遍全局缓存
  // 模拟**：维护容量 = L3 的常驻张量集合（LRU），每个节点先按其**瞬时占用**（并发 WG × 每 WG
  // tile 字节）冲刷 LRU，再读输入（未命中 → 记一次 DRAM 往返）/写输出。耦合由模拟本身处理。
  // 折算：miss_bytes × (1/BW_DRAM − 1/BW_L3)。这是对 OA 计数器不可用的替代估计。
  constexpr double kL3 = 3.75e6;
  // R49: miss 折算系数已抽到 Tuning.cpp 的 `kL3SpillPerByteMs`（单一真相源，mincut 共用）。
  auto optIntOf = [](const std::string & opts, const char * key, int def) -> int {
    const std::string k(key);
    auto p = opts.find(k);
    if (p == std::string::npos) return def;
    return std::atoi(opts.c_str() + p + k.size());
  };
  // R47 标定：饱和阈值是 **8192 个线程**（= 102.4 线程/EU），与 WG 大小、寄存器用量均无关
  // （见 docs/round47-l3-model.md §9.2；此前写死「128 WG」只对 WG=64 成立）。按线程算并发。
  // R49：占用压力已抽成单一真相源 `infvino::occupancyPressure`（Tuning.cpp），
  // mincut 的节点代价与这里的 L3 模拟共用，避免两处口径漂移。
  // R47 step4: 占用感知的**有效 L3 容量**（footprint 实测，§9.1）：驻留足迹 ≈1MB 即达
  // ~21GB/s 平台，之后断崖（组相联/流式占位放大有效占用）。把 LRU 容量按节点占用缩放：
  // 占用越大，留给「其他张量复用」的有效容量越小。以 footprint 平台 1MB 为容量锚点。
  constexpr double kL3Resident1Mb = 1.0e6;
  auto effCapacity = [&](double press) -> double {
    return std::max(kL3Resident1Mb, kL3 - 0.5 * press);
  };
  // R47 补充：**小算子（GEMV/ew/gap/pool/...）的整网外溢**。小算子是流式（读输入+写输出，
  // 保留工作集小），其代价 = launch 地板 + 流式 bytes/BW；且在流式期间会**冲刷 L3**（逐出
  // 常驻张量）。此前的 `predictNet` 只累加了 conv/gemm 的占用，漏掉小算子 —— 这正是 §4.7 里
  // fc 头（GEMV）回归未被预见的原因。这里以「launch + 流式」估计其外溢，并让它进入 L3 模拟。
  auto isSmallOp = [](const std::string & op) {
    return op == "ew_binary" || op == "ew_binary_bcast" || op == "ew_unary" || op == "copy_c" ||
           op == "slice_axis" || op == "concat4" || op == "maxpool" || op == "resize_nn" ||
           op == "permute_0213" || op == "bmm" || op == "softmax_axis" || op == "gap";
  };
  auto smallOpCostMs = [&](size_t ni) -> double {
    const Node & n = nodes_[ni];
    if (!isSmallOp(n.op)) return -1.0;   // -1 = 非小算子
    double bytes = 0.0;
    for (const auto & in : n.ins)
    {
      if (in == "-") continue;
      auto it = T_.find(in);
      if (it == T_.end()) continue;
      bytes += static_cast<double>(it->second.numel()) * 2.0;
    }
    if (!n.outs.empty())
    {
      auto it = T_.find(n.outs[0]);
      if (it != T_.end()) bytes += static_cast<double>(it->second.numel()) * 2.0;
    }
    if (bytes <= 0.0) return 0.0;
    const double bw = copyBwGbps(bytes) * 1e9;
    return kSmallLaunchUs * 1e-3 + bytes / bw * 1e3;   // ms
  };
  auto spillBytes = [&]() -> double {
    std::vector<std::pair<std::string, double>> res;   // MRU 在尾部
    double resBytes = 0.0, miss = 0.0, currentCap = kL3;
    auto touch = [&](const std::string & name, double bytes, bool write) {
      for (size_t k = 0; k < res.size(); ++k)
        if (res[k].first == name)
        {
          auto pv = res[k];
          res.erase(res.begin() + static_cast<long>(k));
          res.push_back(pv);
          return;
        }
      if (!write) miss += bytes;
      res.push_back({name, bytes});
      resBytes += bytes;
      while (resBytes > currentCap && !res.empty())
      {
        resBytes -= res.front().second;
        res.erase(res.begin());
      }
    };
    for (size_t i = 0; i < nodes_.size(); ++i)
    {
      bool ok = false;
      const OpSignature s = nodeSignature(nodes_[i], &ok);
      const TuningEntry * e = ok ? choiceEntry(i, s) : nullptr;
      // 占用压力：按占用算有效容量并冲刷 LRU（跨算子耦合在此体现）。
      double press = (ok && e) ? occupancyPressure(*e, s) : 0.0;
      // R47 补充：小算子流式读+写会冲刷 L3（其流式足迹即有效压力）。
      if (isSmallOp(nodes_[i].op))
      {
        double fb = 0.0;
        for (const auto & in : nodes_[i].ins)
        {
          if (in == "-") continue;
          auto it = T_.find(in);
          if (it != T_.end()) fb += static_cast<double>(it->second.numel()) * 2.0;
        }
        if (!nodes_[i].outs.empty())
        {
          auto it = T_.find(nodes_[i].outs[0]);
          if (it != T_.end()) fb += static_cast<double>(it->second.numel()) * 2.0;
        }
        press = std::max(press, fb);
      }
      currentCap = effCapacity(press);
      while (resBytes > currentCap && !res.empty())
      {
        resBytes -= res.front().second;
        res.erase(res.begin());
      }
      const Node & n = nodes_[i];
      for (const auto & in : n.ins)
      {
        if (in == "-") continue;
        auto it = T_.find(in);
        if (it == T_.end()) continue;   // 权重/非激活 → 不计
        touch(in, static_cast<double>(it->second.numel()) * 2.0, false);
      }
      if (!n.outs.empty())
      {
        auto it = T_.find(n.outs[0]);
        if (it != T_.end()) touch(n.outs[0], static_cast<double>(it->second.numel()) * 2.0, true);
      }
    }
    return miss;
  };
  auto spillMs = [&]() -> double { return spillBytes() * kL3SpillPerByteMs; };

  // R47 §3-P0: **可加目标**（GPU-free 代理整网）—— Σ 逐节点 kernel ms + 未持久化 blk 输入的
  // reorder 成本 + R47 step2 的 L3 溢出估计。对应 TVM meta_schedule 的「Σ weight × 单算子 ms」，
  // 并把内存层级显式建模。用于排序 / 剪枝 / 分组，**不用于最终裁决**。
  auto predictNet = [&](const std::vector<TuningEntry> & asg) -> double {
    node_choice_ = asg;
    planBlockedLayout();  // 只更新 fsv16 标志，不触碰 dispatch
    double net = 0.0;
    for (size_t ni = 0; ni < nodes_.size(); ++ni)
    {
      // R47 fix: asg 只填 target 节点；非 target 节点必须按 choiceEntry 解析其**实际选择**，
      // 否则「整网代理」只统计到被调优的少数节点（reorder 占比也会误算成 0）。
      bool ok = false;
      const OpSignature s = nodeSignature(nodes_[ni], &ok);
      const TuningEntry * e = ok ? choiceEntry(ni, s) : nullptr;
      if (!e && ni < asg.size() && !asg[ni].kernel.empty()) e = &asg[ni];
      if (!e) continue;
      if (!e->kernel.empty()) net += e->ms;
      if (e->kernel.find("_blk") == std::string::npos) continue;
      const Node & n = nodes_[ni];
      const size_t inIdx = (n.op == "conv1x1" || n.op == "conv1x1_cat4") ? 1 : 0;
      if (n.ins.size() <= inIdx) continue;
      auto xit = T_.find(n.ins[inIdx]);
      if (xit == T_.end() || xit->second.fsv16) continue;
      if (!ok) continue;
      if (const TuningEntry * r = tuning_.lookup(OpSignature::custom(s.str() + "#reorder", {})))
        net += r->ms;
    }
    // R47 补充：小算子的 launch + 流式成本（隔离 ms 已含 kernel 时间，但 GEMV/ew 的
    // 「跨 op 外溢」——冲刷 L3 后再被消费者读取——需显式计入，否则 fc 头回归不可见）。
    for (size_t ni = 0; ni < nodes_.size(); ++ni)
    {
      const double c = smallOpCostMs(ni);
      if (c > 0.0) net += c;
    }
    net += spillMs();   // R47 step2: L3 溢出（复用距离 + reorder 缓冲 + 小算子流式）
    return net;
  };
  // R47 S3: 只算可加目标里的 **reorder 分量**（供 chain move 门控：reorder 占比小的模型跳过）。
  auto predReorder = [&](const std::vector<TuningEntry> & asg) -> double {
    node_choice_ = asg;
    planBlockedLayout();
    double r = 0.0;
    for (size_t ni = 0; ni < nodes_.size(); ++ni)
    {
      bool ok = false;
      const OpSignature s = nodeSignature(nodes_[ni], &ok);
      if (!ok) continue;
      const TuningEntry * e = choiceEntry(ni, s);
      if (!e && ni < asg.size() && !asg[ni].kernel.empty()) e = &asg[ni];
      if (!e || e->kernel.find("_blk") == std::string::npos) continue;
      const Node & n = nodes_[ni];
      const size_t inIdx = (n.op == "conv1x1" || n.op == "conv1x1_cat4") ? 1 : 0;
      if (n.ins.size() <= inIdx) continue;
      auto xit = T_.find(n.ins[inIdx]);
      if (xit == T_.end() || xit->second.fsv16) continue;
      if (const TuningEntry * rr = tuning_.lookup(OpSignature::custom(s.str() + "#reorder", {})))
        r += rr->ms;
    }
    return r;
  };

  // R47 §3-P0: **布局耦合连通分量**。共享「可能被重排的输入张量」、或「blk 生产者→消费者」
  // 的节点，其选择互相影响（改一个会改变另一个的 reorder / 持久化）——这些才是必须联合
  // 端到端回验的「不可加残差」；其余签名按可加目标 per-node 决定即可。
  std::vector<int> comp(nodes_.size(), -1);
  {
    std::vector<int> par(nodes_.size());
    for (size_t i = 0; i < nodes_.size(); ++i) par[i] = static_cast<int>(i);
    auto find = [&](int x) { while (par[x] != x) { par[x] = par[par[x]]; x = par[x]; } return x; };
    auto uni = [&](int a, int b) { a = find(a); b = find(b); if (a != b) par[a] = b; };
    std::unordered_map<std::string, std::vector<size_t>> cons;
    for (size_t i = 0; i < nodes_.size(); ++i)
      for (const auto & t : nodes_[i].ins)
        if (t != "-") cons[t].push_back(i);
    // (a) 共享输入张量（同帧 reorder 去重的受益者）→ 同分量。
    for (auto & kv : cons)
      for (size_t k = 1; k < kv.second.size(); ++k) uni(static_cast<int>(kv.second[0]),
                                                        static_cast<int>(kv.second[k]));
    // (b) 生产者输出 → 消费者（fsv16 持久化契约）。
    for (size_t i = 0; i < nodes_.size(); ++i)
      if (!nodes_[i].outs.empty())
      {
        auto it = cons.find(nodes_[i].outs[0]);
        if (it != cons.end())
          for (size_t j : it->second) uni(static_cast<int>(i), static_cast<int>(j));
      }
    for (size_t i = 0; i < nodes_.size(); ++i) comp[i] = find(static_cast<int>(i));
  }
  std::unordered_map<int, int> compSize;
  for (int c : comp) ++compSize[c];
  auto coupled = [&](size_t ni) { return compSize[comp[ni]] > 1; };

  // R47: 目标优先级 = 可加目标的预测 headroom 降序（TVM gradient 的静态近似）：
  // 先投给「最可能有收益」的签名。headroom = predictNet(base) − min_c predictNet(base,c)。
  struct TgtMeta { double headroom = 0.0; bool coupled = false; };
  std::vector<TgtMeta> tmeta(targets.size());
  {
    const double basePred = predictNet(baseline);
    for (size_t ti = 0; ti < targets.size(); ++ti)
    {
      tmeta[ti].coupled = !targets[ti].nodes.empty() && coupled(targets[ti].nodes[0]);
      std::vector<TuningEntry> tmp = baseline;
      double best = basePred;
      for (const auto & c : targets[ti].cs.cands)
      {
        tmp = baseline;
        for (size_t ni : targets[ti].nodes) tmp[ni] = c;
        const double p = predictNet(tmp);
        if (p < best) best = p;
      }
      tmeta[ti].headroom = basePred - best;
    }
    std::vector<size_t> order(targets.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](size_t a, size_t b) { return tmeta[a].headroom > tmeta[b].headroom; });
    std::vector<Target> sorted;
    std::vector<TgtMeta> smeta;
    sorted.reserve(order.size());
    smeta.reserve(order.size());
    for (size_t i : order) { sorted.push_back(targets[i]); smeta.push_back(tmeta[i]); }
    targets.swap(sorted);
    tmeta.swap(smeta);
  }
  assign = baseline;  // predictNet 只改 node_choice_/fsv16，这里确保赋值向量回到基线

  // R47: 先量基线（双口径），最终验收门再与它交错对照。
  const NetStat baseStat = measureAssign(assign, reps);
  const double baseNet = baseStat.med;

  if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
  {
    node_choice_ = baseline;
    planBlockedLayout();
    const double baseSpill = spillMs();
    std::fprintf(stderr,
                 "[global-retune] %zu signatures, topK=%d reps=%d rounds=%d, metric=min+median, "
                 "accept=median, interleave=on, additive-priority=on, grouped=%s, base(med)=%.4f, "
                 "base(L3 spill est)=%.4f ms\n",
                 targets.size(), topK, reps, rounds, groupedOn ? "on" : "off", baseNet, baseSpill);
  }

  // 落盘：base 签名写回 winner；同时把**同族**的 #blk / #non 备选更新为 winner，使运行时
  // 的联合布局不动点在本 plan 上复现出同一个选择（跨 plan 的 #blk/#non 唯一分解限制见审计）。
  auto persist = [&](const CandidateSet & cs, const TuningEntry & e) {
    tuning_.put(cs.sig, e);
    const bool isBlk = e.kernel.find("_blk") != std::string::npos;
    const OpSignature altKey =
      OpSignature::custom(cs.sig.str() + (isBlk ? "#blk" : "#non"), {});
    if (tuning_.lookup(altKey)) tuning_.put(altKey, e);
  };

  std::vector<char> won(targets.size(), 0);

  // R47「第二步」：**模型驱动选择**（`INFVINO_GLOBAL_MODEL=1`）。让 L3 模型真正**主导选择**，
  // 而不是逐节点端到端回验（后者被 ±5% 噪声主导、非组合）。流程：
  //   (1) 逐 target 用 `predictNet` argmin 选候选（唯一随预测改善才接受）→ 得 proposal；
  //   (2) 整网**预测**改善门（零 GPU）；
  //   (3) 可选**外部稳态确认门**（`INFVINO_GLOBAL_MODEL_CONFIRM=1`）——一次 committed vs
  //       proposal 的 median 对照，只要不显著变差就放行（否则整体回退）。
  // 与旧的端到端坐标下降互斥（此模式不逐候选测量）。
  if (std::getenv("INFVINO_GLOBAL_MODEL"))
  {
    std::vector<TuningEntry> proposal = baseline;
    const double basePred0 = predictNet(baseline);
    int proposed = 0;
    // 一轮模型坐标下降：逐 target 用 predictNet argmin 选候选（唯一随预测改善才接受）。
    auto modelRound = [&](std::vector<TuningEntry> & prop) {
      int ch = 0;
      for (size_t ti = 0; ti < targets.size(); ++ti)
      {
        auto & t = targets[ti];
        if (t.nodes.empty()) continue;
        const TuningEntry cur = prop[t.nodes[0]];
        TuningEntry bestE = cur;
        std::vector<TuningEntry> bestTmp = prop;
        double bestPred = predictNet(prop);
        for (const auto & c : t.cs.cands)
        {
          if (c.kernel == cur.kernel && c.options == cur.options) continue;
          std::vector<TuningEntry> tmp = prop;
          for (size_t ni : t.nodes) tmp[ni] = c;
          const double p = predictNet(tmp);
          if (p < bestPred * (1.0 - kMinGain)) { bestPred = p; bestE = c; bestTmp = tmp; }
        }
        if (bestE.kernel != cur.kernel || bestE.options != cur.options)
        {
          prop = bestTmp;
          ++ch;
          if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
            std::fprintf(stderr, "[global-retune] model propose %-38s %s -> %s\n",
                         t.cs.sig.str().c_str(), cur.kernel.c_str(), bestE.kernel.c_str());
        }
      }
      return ch;
    };
    // 从 baseline 收敛。
    proposed += modelRound(proposal);
    // **扰动重启**：坐标下降会卡在局部最优（尤其布局耦合的“反协同”谷底）。对每个 target 试
    // 用**次优候选**（按 predictNet）扰动，再收敛一次；若得到更低的预测则接受。取最好解。
    double best = predictNet(proposal);
    for (size_t ti = 0; ti < targets.size(); ++ti)
    {
      auto & t = targets[ti];
      if (t.nodes.empty()) continue;
      const TuningEntry cur = proposal[t.nodes[0]];
      // 选一个**与当前不同**、预测次优的候选作扰动起点。
      TuningEntry perturbE = cur;
      double second = 1e300;
      for (const auto & c : t.cs.cands)
      {
        if (c.kernel == cur.kernel && c.options == cur.options) continue;
        std::vector<TuningEntry> tmp = proposal;
        for (size_t ni : t.nodes) tmp[ni] = c;
        const double p = predictNet(tmp);
        if (p < second) { second = p; perturbE = c; }
      }
      if (perturbE.kernel == cur.kernel && perturbE.options == cur.options) continue;
      std::vector<TuningEntry> alt = proposal;
      for (size_t ni : t.nodes) alt[ni] = perturbE;
      modelRound(alt);
      const double ap = predictNet(alt);
      if (ap < best * (1.0 - kMinGain)) { best = ap; proposal = std::move(alt); }
    }
    const double basePred = basePred0;
    const double propPred = predictNet(proposal);
    proposed = 0;
    for (size_t ti = 0; ti < targets.size(); ++ti)
    {
      auto & t = targets[ti];
      if (t.nodes.empty()) continue;
      if (proposal[t.nodes[0]].kernel != baseline[t.nodes[0]].kernel ||
          proposal[t.nodes[0]].options != baseline[t.nodes[0]].options)
      { won[ti] = 1; ++proposed; }
    }
    const bool predBetter = propPred < basePred * (1.0 - kMinGain);
    if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
      std::fprintf(stderr,
                   "[global-retune] model-driven: %d target(s), predictNet %.4f -> %.4f ms (%+.1f%%) %s\n",
                   proposed, basePred, propPred,
                   basePred > 0.0 ? (propPred - basePred) / basePred * 100.0 : 0.0,
                   predBetter ? "ACCEPT" : "REJECT(pred)");
    bool commit = predBetter && proposed > 0;
    if (commit && std::getenv("INFVINO_GLOBAL_MODEL_CONFIRM"))
    {
      NetStat bsW, psW;
      measurePair(baseline, proposal, std::max(reps, 3), &bsW, &psW);
      const bool ok = psW.ok && bsW.ok && psW.med < bsW.med * (1.0 + kMinGain);  // 不显著变差即放行
      if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
        std::fprintf(stderr, "[global-retune] model confirm: net(median) %.4f -> %.4f ms %s\n",
                     bsW.med, psW.med, ok ? "PASS" : "FAIL(revert)");
      commit = ok;
    }
    if (commit)
    {
      assign = proposal;
      for (size_t ti = 0; ti < targets.size(); ++ti)
      {
        auto & t = targets[ti];
        if (!t.nodes.empty() && !assign[t.nodes[0]].kernel.empty()) persist(t.cs, assign[t.nodes[0]]);
      }
    }
    plan_overrides_.setEnabled(true);
    plan_overrides_.setDeviceId(tuning_.deviceId());
    for (auto & t : targets)
      for (size_t ni : t.nodes)
        if (ni < assign.size() && ni < nodes_.size() && !nodes_[ni].outs.empty() &&
            !assign[ni].kernel.empty())
        {
          TuningEntry e = assign[ni];
          if (e.device_id.empty()) e.device_id = tuning_.deviceId();
          plan_overrides_.put(planNodeKey(nodes_[ni].outs[0]), e);
        }
    invalidateCapture();
    resolveLayoutChoices();
    return commit ? proposed : 0;
  }

  bool budgetHit = false;
  for (int round = 0; round < rounds; ++round)
  {
    if (budget > 0 && evals >= budget) { budgetHit = true; break; }
    // R47 前置①: **整赋值验收** —— 本轮所有逐节点决定都相对**同一个固定上下文** `committed`
    // 评估（而不是随提交演进的 assign），汇总成一个整赋值候选 `proposal`；只有 `proposal`
    // 相对 `committed` 的**整网稳态**（median、多 rep）确实改善时才提交整轮。这堵住
    // 「逐节点小改善不组合、累积成净回归」的非组合性（实测 +0.8% 漏网 / +2.9% 才回退）。
    const std::vector<TuningEntry> committed = assign;
    int proposed = 0;
    std::vector<TuningEntry> proposal = committed;
    for (size_t ti = 0; ti < targets.size(); ++ti)
    {
      if (budget > 0 && evals >= budget) { budgetHit = true; break; }
      auto & t = targets[ti];
      if (t.nodes.empty()) continue;
      const TuningEntry base = committed[t.nodes[0]];
      std::vector<TuningEntry> baseAssign = committed;   // 固定上下文（不随本轮提交演进）
      TuningEntry bestE = base;
      NetStat bestStat = measureAssign(baseAssign, reps);
      if (!bestStat.ok) continue;
      const double baseMed = bestStat.med;
      const double basePred = predictNet(baseAssign);   // R47 S2: 布局契约门基线
      if (bestStat.spread > 0.05 && std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
        std::fprintf(stderr, "[global-retune] WARN %s base noisy (spread %.0f%%)\n",
                     t.cs.sig.str().c_str(), bestStat.spread * 100.0);
      for (const auto & c : t.cs.cands)
      {
        if (c.kernel == bestE.kernel && c.options == bestE.options) continue;
        std::vector<TuningEntry> candAssign = baseAssign;
        for (size_t ni : t.nodes) candAssign[ni] = c;
        // R47 S2: 布局契约门 —— 预测净（含 reorder）明显变差的候选直接拒绝，不做端到端测量。
        const double candPred = predictNet(candAssign);
        if (candPred > basePred * (1.0 + kPredTol))
        {
          if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
            std::fprintf(stderr,
                         "[global-retune] gate %s candidate %s pred %.4f > base %.4f (+%.1f%%)\n",
                         t.cs.sig.str().c_str(), c.kernel.c_str(), candPred, basePred,
                         basePred > 0.0 ? (candPred - basePred) / basePred * 100.0 : 0.0);
          continue;
        }
        NetStat bs, cs;
        measurePair(baseAssign, candAssign, reps, &bs, &cs);   // R47: 交错，抵消热漂移
        if (!cs.ok) continue;
        // R47: 改善必须在 **median**（典型口径）成立；min 仅做地板守卫。
        const bool medBetter = cs.med < bestStat.med * (1.0 - kMinGain);
        const bool floorOk = (bs.mn >= 1e299) || (cs.mn <= bs.mn * (1.0 + 0.02));
        if (medBetter && floorOk) { bestStat = cs; bestE = c; }
      }
      if (bestE.kernel != base.kernel || bestE.options != base.options)
      {
        for (size_t ni : t.nodes) proposal[ni] = bestE;
        ++proposed;
        if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
          std::fprintf(stderr,
                       "[global-retune] round%d propose %-38s %s -> %s  (target med %.4f -> %.4f)\n",
                       round, t.cs.sig.str().c_str(), base.kernel.c_str(), bestE.kernel.c_str(),
                       baseMed, bestStat.med);
      }
    }
    if (proposed == 0) break;
    // 整赋值验收：committed vs proposal 的**整网稳态**（median、多 rep）——通过才提交整轮，
    // 否则整轮拒绝（不回退、不部分提交）。
    NetStat bsW, psW;
    measurePair(committed, proposal, std::max(reps, 3), &bsW, &psW);
    const bool wholeBetter = psW.ok && bsW.ok && psW.med < bsW.med * (1.0 - kMinGain);
    if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
      std::fprintf(stderr, "[global-retune] round%d whole-net(median) %.4f -> %.4f ms (%+.1f%%) %s\n",
                   round, bsW.med, psW.med,
                   bsW.med > 0.0 ? (psW.med - bsW.med) / bsW.med * 100.0 : 0.0,
                   wholeBetter ? "COMMIT" : "REJECT (no commit)");
    if (!wholeBetter) break;
    for (size_t ti = 0; ti < targets.size(); ++ti)
    {
      auto & t = targets[ti];
      if (t.nodes.empty()) continue;
      if (proposal[t.nodes[0]].kernel != committed[t.nodes[0]].kernel ||
          proposal[t.nodes[0]].options != committed[t.nodes[0]].options)
        won[ti] = 1;
    }
    assign = proposal;
    if (budgetHit) break;
  }

  // R47 S3: **决策级 chain move** —— 对每个布局耦合连通分量，显式尝试
  //   (a) 逐节点贪心最优、(b) 整链一起切 blk、(c) 整链一起切 non
  // 三种**整分量联合赋值**，各作为一个 move 端到端回验（先过 S2 契约门），取 median 最优者。
  // 这是「blocked chain 的决策级近似」：不改 kernel/数据流，只把「全链是否一起进 blocked
  // 布局」当成一个联合决定——命中「整链一起切 blk → reorder=0」的联合最优，而逐节点坐标下降
  // 做不到（反协同）。门控：预测 reorder 占比 < kChainReorderShare 的模型跳过（如 yolo）。
  if (groupedOn)
  {
    const double kChainReorderShare = 0.02;
    const double baseNetPred = predictNet(baseline);
    const double reorderShare = baseNetPred > 0.0 ? predReorder(baseline) / baseNetPred : 0.0;
    if (reorderShare < kChainReorderShare)
    {
      if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
        std::fprintf(stderr,
                     "[global-retune] chain moves skipped (reorder share %.1f%% < %.1f%%)\n",
                     reorderShare * 100.0, kChainReorderShare * 100.0);
    }
    else
    {
      std::unordered_map<int, std::vector<size_t>> compTargets;
      for (size_t ti = 0; ti < targets.size(); ++ti)
        if (!targets[ti].nodes.empty()) compTargets[comp[targets[ti].nodes[0]]].push_back(ti);
      // 按族选候选：mode 1=仅 blk、2=仅 non。**按 kernel ms 选**（不看 reorder）——reorder 是
      // 整链的联合属性，必须在整链赋值上评估；逐节点用 predictNet 会被 reorder 卡住而无法翻转
      // （顺序贪心陷阱，正是 chain move 要解决的）。
      auto pickFamily = [&](size_t ti, int mode) -> TuningEntry {
        auto & t = targets[ti];
        TuningEntry bestE = assign[t.nodes[0]];
        double bestMs = 1e300;
        bool found = false;
        for (const auto & c : t.cs.cands)
        {
          const bool isBlk = c.kernel.find("_blk") != std::string::npos;
          if (mode == 1 && !isBlk) continue;
          if (mode == 2 && isBlk) continue;
          if (c.ms < bestMs) { bestMs = c.ms; bestE = c; found = true; }
        }
        return found ? bestE : assign[t.nodes[0]];
      };
      if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
      {
        int coupled = 0;
        for (auto & kv : compTargets) if (kv.second.size() >= 2) ++coupled;
        std::fprintf(stderr, "[global-retune] chain-diag: %zu component(s) hold targets, %d coupled\n",
                     compTargets.size(), coupled);
      }
      for (auto & kv : compTargets)
      {
        if (kv.second.size() < 2) continue;   // 非耦合分量无需分组（逐节点已最优）
        if (budget > 0 && evals >= budget) { budgetHit = true; break; }
        std::vector<TuningEntry> baseAssign = assign;
        std::vector<size_t> order = kv.second;
        std::stable_sort(order.begin(), order.end(),
                         [&](size_t a, size_t b) { return tmeta[a].headroom > tmeta[b].headroom; });
        std::vector<std::vector<TuningEntry>> candJoints;
        for (int mode = 1; mode <= 2; ++mode)   // 1=整链切 blk, 2=整链切 non
        {
          std::vector<TuningEntry> j = assign;
          bool changed = false;
          for (size_t ti : order)
          {
            const TuningEntry cur = j[targets[ti].nodes[0]];
            const TuningEntry e = pickFamily(ti, mode);
            if (e.kernel != cur.kernel || e.options != cur.options) changed = true;
            for (size_t ni : targets[ti].nodes) j[ni] = e;
          }
          if (!changed) continue;
          if (predictNet(j) > baseNetPred * (1.0 + kPredTol)) continue;   // S2 契约门
          bool dup = false;   // 去重（整链 blk 与整链 non 可能得到同一赋值）
          for (auto & pv : candJoints)
          {
            bool same = true;
            for (size_t ti : order)
              for (size_t ni : targets[ti].nodes)
                if (pv[ni].kernel != j[ni].kernel || pv[ni].options != j[ni].options)
                {
                  same = false;
                  break;
                }
            if (same) { dup = true; break; }
          }
          if (!dup) candJoints.push_back(std::move(j));
        }
        if (candJoints.empty())
        {
          if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
            std::fprintf(stderr,
                         "[global-retune] chain-diag: comp=%d %zu targets -> no joint candidate "
                         "(no-change or contract-gated)\n",
                         kv.first, kv.second.size());
          continue;
        }
        double bestMed = 1e300;
        NetStat bestBs;
        std::vector<TuningEntry> bestJoint;
        bool haveBest = false;
        for (auto & j : candJoints)
        {
          NetStat bs, js;
          measurePair(baseAssign, j, reps, &bs, &js);
          if (!js.ok || !bs.ok) continue;
          if (js.med < bestMed) { bestMed = js.med; bestBs = bs; bestJoint = j; haveBest = true; }
        }
        if (haveBest && bestMed < bestBs.med * (1.0 - kMinGain))
        {
          assign = bestJoint;
          for (size_t ti : kv.second) won[ti] = 1;
          if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
            std::fprintf(stderr,
                         "[global-retune] chain move (comp=%d, %zu targets): med %.4f -> %.4f ms "
                         "(%.1f%%)\n",
                         kv.first, kv.second.size(), bestBs.med, bestMed,
                         bestBs.med > 0.0 ? (bestBs.med - bestMed) / bestBs.med * 100.0 : 0.0);
        }
      }
    }
  }
  if (budgetHit && std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
    std::fprintf(stderr, "[global-retune] budget hit (%d evals); stopped early\n", evals);

  // R47 最终验收门：交错测「初始基线」vs「最终组合」，用 **median**（典型口径）裁决：只有最终
  // 组合不慢于基线才落盘，否则整轮回退。（R45 门用单一 min；R46 证明其与稳态口径错配。）
  {
    std::vector<TuningEntry> finalAssign = assign;
    NetStat bls, fs;
    measurePair(baseline, finalAssign, std::max(reps, 3), &bls, &fs);
    const bool revert = (bls.ok && fs.ok && fs.med > bls.med * (1.0 + kMinGain));
    if (std::getenv("INFVINO_GLOBAL_RETUNE_REPORT"))
      std::fprintf(stderr, "[global-retune] net(median) %.4f -> %.4f ms (%+.1f%%)%s\n", bls.med,
                   fs.med, bls.med > 0.0 ? (fs.med - bls.med) / bls.med * 100.0 : 0.0,
                   revert ? "  REVERT (final worse than baseline)" : "");
    if (revert)
    {
      assign = baseline;
      std::fill(won.begin(), won.end(), 0);
    }
    else
    {
      // 通过验收：只把**真正改变**的 target 写回（base 签名 + 同族备选）；未改变的不动共享缓存。
      for (size_t ti = 0; ti < targets.size(); ++ti)
      {
        auto & t = targets[ti];
        if (!won[ti] || t.nodes.empty()) continue;
        if (!assign[t.nodes[0]].kernel.empty()) persist(t.cs, assign[t.nodes[0]]);
      }
    }
  }
  int changed_total = 0;
  for (char w : won) if (w) ++changed_total;

  // R45 P0#4: 把本图的位置相关选择写进 per-plan 覆盖（node 输出名为键），与跨模型共享的
  // tuning_ 分离——避免「在 y8 上做的全局选择覆盖掉 y11 需要的那份」。
  plan_overrides_.setEnabled(true);
  plan_overrides_.setDeviceId(tuning_.deviceId());
  for (auto & t : targets)
    for (size_t ni : t.nodes)
      if (ni < assign.size() && ni < nodes_.size() && !nodes_[ni].outs.empty() &&
          !assign[ni].kernel.empty())
      {
        TuningEntry e = assign[ni];
        if (e.device_id.empty()) e.device_id = tuning_.deviceId();
        plan_overrides_.put(planNodeKey(nodes_[ni].outs[0]), e);
      }

  invalidateCapture();
  resolveLayoutChoices();
  return changed_total;
}

int PlanModel::onlineTuneMissing(int budget, int iters, const std::vector<std::string> & ops)
{
  if (budget <= 0 || iters <= 0) return 0;
  try
  {
    auto done = autotune(ops, "", budget, iters, /*merge=*/true, /*verbose=*/false,
                         /*retune=*/false);
    return static_cast<int>(done.size());
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "[online-tune] skipped: %s\n", e.what());
    return 0;
  }
}

}  // namespace infvino
