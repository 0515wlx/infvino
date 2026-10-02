// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/PlanModel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "infvino/Half.hpp"
#include "infvino/Tiles.hpp"

namespace gk
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
}  // namespace

PlanModel::PlanModel(
  const std::string & plan_path, const std::string & kernel_dir, int platform, int device_index,
  bool profiling)
: rt_(kernel_dir, platform, device_index, profiling),
  plan_path_(plan_path),
  plan_dir_(dirName(plan_path)),
  profiling_(profiling)
{
  parse();
  buildKernels();
}

PlanModel::~PlanModel()
{
  releaseKernels();
  for (cl_mem m : owned_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_ov_)
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

  auto alloc = [&](const std::string & name, const std::vector<int64_t> & d) -> Tensor & {
    Tensor t;
    t.dims = d;
    t.mem  = rt_.alloc(static_cast<size_t>(t.numel()) * 2, CL_MEM_READ_WRITE);
    owned_.push_back(t.mem);
    return T_[name] = t;
  };

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
      alloc(name, d);
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

  // 预取权重到设备后无需再保留主机侧数据；rt_.write 为阻塞写。
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

void PlanModel::run()
{
  const auto t0 = std::chrono::steady_clock::now();

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
    cl_event ev = ClRuntime::enqueueND(rt_.queue(), k, dim, gws, lws);
    if (profiling_)
    {
      clWaitForEvents(1, &ev);
      cl_ulong s = 0, e = 0;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(e), &e, nullptr);
      tprof_[tag].first += static_cast<double>(e - s) * 1e-6;
      tprof_[tag].second += 1;
    }
    clReleaseEvent(ev);
  };

  for (const auto & n : nodes_)
  {
    auto & out = ref(n.outs[0]);
    if (n.op == "reshape" || n.op == "flatten")
    {
      out.mem = ref(n.ins[0]).mem;  // 视图别名，零拷贝
      continue;
    }

    auto in = [&](size_t i) -> Tensor & { return ref(n.ins[i]); };
    auto out_numel = [&]() -> int64_t {
      auto it = T_.find(n.outs[0]);
      if (it != T_.end()) return it->second.numel();
      return T_.at(n.ins[0]).numel();
    };
    const int64_t nout = out_numel();
    const size_t  g1   = static_cast<size_t>(out_numel());

    if (n.op == "conv1x1")
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
      cl_mem dw = w.mem, dx = in(1).mem, dy = out.mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? in(2).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? in(3).mem : nullptr;
      if (N == 1) {
        Conv1x1Cfg cfg;
        cfg.ACT = act;
        cfg.RES = dres ? 1 : 0;
        cfg.SG  = 16;
        cl_kernel kg = getKernel("conv1x1", "conv1x1_gemv_f16", cfg.options());
        clSetKernelArg(kg, 0, sizeof(dw), &dw);
        clSetKernelArg(kg, 1, sizeof(dx), &dx);
        clSetKernelArg(kg, 2, sizeof(db), &db);
        clSetKernelArg(kg, 3, sizeof(dres), &dres);
        clSetKernelArg(kg, 4, sizeof(dy), &dy);
        clSetKernelArg(kg, 5, sizeof(Cin), &Cin);
        clSetKernelArg(kg, 6, sizeof(Cout), &Cout);
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
        cl_kernel kg = rt_.buildKernel("gemm", "gemm_f16", t.options());
        clSetKernelArg(kg, 0, sizeof(dw), &dw);
        clSetKernelArg(kg, 1, sizeof(dx), &dx);
        clSetKernelArg(kg, 2, sizeof(dy), &dy);
        clSetKernelArg(kg, 3, sizeof(Cout), &Cout);
        clSetKernelArg(kg, 4, sizeof(N), &N);
        clSetKernelArg(kg, 5, sizeof(Cin), &Cin);
        clSetKernelArg(kg, 6, sizeof(db), &db);
        clSetKernelArg(kg, 7, sizeof(dres), &dres);
        const size_t lws[2] = {t.localX(), t.localY()};
        const size_t gws[2] = {
          static_cast<size_t>((N + t.BN - 1) / t.BN) * lws[0],
          static_cast<size_t>((Cout + t.BM - 1) / t.BM) * lws[1]};
        timed("conv1x1@" + std::to_string(Cout) + "x" + std::to_string(N) + "x" +
                std::to_string(Cin),
              kg, 2, gws, lws);
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
        cl_kernel kd = getKernel("conv_general", "depthwise_f16", dopts);
        int ho = Hout, wo = Wout;
        clSetKernelArg(kd, 0, sizeof(dx), &dx);
        clSetKernelArg(kd, 1, sizeof(dw), &dw);
        clSetKernelArg(kd, 2, sizeof(db), &db);
        clSetKernelArg(kd, 3, sizeof(dy), &dy);
        clSetKernelArg(kd, 4, sizeof(Cin), &Cin);
        clSetKernelArg(kd, 5, sizeof(H), &H);
        clSetKernelArg(kd, 6, sizeof(W), &W);
        clSetKernelArg(kd, 7, sizeof(ho), &ho);
        clSetKernelArg(kd, 8, sizeof(wo), &wo);
        const size_t gdw[1] = {static_cast<size_t>(Cin) * Hout * Wout};
        timed("depthwise", kd, 1, gdw, nullptr);
      } else {
      clSetKernelArg(kConvG_, 0, sizeof(dx), &dx);
      clSetKernelArg(kConvG_, 1, sizeof(dw), &dw);
      clSetKernelArg(kConvG_, 2, sizeof(db), &db);
      clSetKernelArg(kConvG_, 3, sizeof(dy), &dy);
      clSetKernelArg(kConvG_, 4, sizeof(Cin), &Cin);
      clSetKernelArg(kConvG_, 5, sizeof(H), &H);
      clSetKernelArg(kConvG_, 6, sizeof(W), &W);
      clSetKernelArg(kConvG_, 7, sizeof(Cout), &Cout);
      int ho = Hout, wo = Wout;
      clSetKernelArg(kConvG_, 8, sizeof(ho), &ho);
      clSetKernelArg(kConvG_, 9, sizeof(wo), &wo);
      int KK = K, SS = S, PP = P, GG = G, aa = act;
      clSetKernelArg(kConvG_, 10, sizeof(KK), &KK);
      clSetKernelArg(kConvG_, 11, sizeof(SS), &SS);
      clSetKernelArg(kConvG_, 12, sizeof(PP), &PP);
      clSetKernelArg(kConvG_, 13, sizeof(GG), &GG);
      clSetKernelArg(kConvG_, 14, sizeof(aa), &aa);
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

      // Round 22: OpenVINO os_iyx_osv32 port (kernels/conv_ov.cl) is the default
      // 3x3 groups=1 path. Measured faster than the native direct conv on every
      // model shape (e.g. 64->64@80 13.6 vs 10.3 ops/EU/cyc; s2 10.7 vs 7.1),
      // because lane=channel + block-read weights removes the SLM staging cost.
      // Set `ov=0` on the node to fall back to the native conv3x3_f16.
      if (attrInt(n, "ov", 1))
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
        clSetKernelArg(kk, 0, sizeof(dx), &dx);
        clSetKernelArg(kk, 1, sizeof(dw), &dw);
        clSetKernelArg(kk, 2, sizeof(db), &db);
        clSetKernelArg(kk, 3, sizeof(dres), &dres);
        clSetKernelArg(kk, 4, sizeof(dy), &dy);
        clSetKernelArg(kk, 5, sizeof(Cin), &Cin);
        clSetKernelArg(kk, 6, sizeof(H), &H);
        clSetKernelArg(kk, 7, sizeof(W), &W);
        clSetKernelArg(kk, 8, sizeof(Cout), &Cout);
        clSetKernelArg(kk, 9, sizeof(Hout), &Hout);
        clSetKernelArg(kk, 10, sizeof(Wout), &Wout);
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
        clSetKernelArg(kk, 0, sizeof(dx), &dx);
        clSetKernelArg(kk, 1, sizeof(dw), &dw);
        clSetKernelArg(kk, 2, sizeof(db), &db);
        clSetKernelArg(kk, 3, sizeof(dy), &dy);
        clSetKernelArg(kk, 4, sizeof(Cin), &Cin);
        clSetKernelArg(kk, 5, sizeof(H), &H);
        clSetKernelArg(kk, 6, sizeof(W), &W);
        clSetKernelArg(kk, 7, sizeof(Cout), &Cout);
        clSetKernelArg(kk, 8, sizeof(Hout), &Hout);
        clSetKernelArg(kk, 9, sizeof(Wout), &Wout);
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
      cl_kernel kg = getKernel("gemm", "gemm_f16", t.options());
      clSetKernelArg(kg, 0, sizeof(da), &da);
      clSetKernelArg(kg, 1, sizeof(db), &db);
      clSetKernelArg(kg, 2, sizeof(dc), &dc);
      clSetKernelArg(kg, 3, sizeof(M), &M);
      clSetKernelArg(kg, 4, sizeof(N), &N);
      clSetKernelArg(kg, 5, sizeof(K), &K);
      const size_t lws[2] = {t.localX(), t.localY()};
      const size_t gws[2] = {
        static_cast<size_t>((N + t.BN - 1) / t.BN) * lws[0],
        static_cast<size_t>((M + t.BM - 1) / t.BM) * lws[1]};
      timed("gemm@" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K),
            kg, 2, gws, lws);
    }
    else if (n.op == "ew_binary")
    {
      int nn = static_cast<int>(nout), op = attrInt(n, "op", 0), bs = attrInt(n, "b_scalar", 0);
      cl_mem da = in(0).mem, db = in(1).mem, dy = out.mem;
      auto   bit = n.attr.find("bdims");
      if (bit != n.attr.end())
      {
        std::vector<int> all;
        {
          std::stringstream ss(bit->second);
          std::string       tk;
          while (std::getline(ss, tk, ',')) all.push_back(std::atoi(tk.c_str()));
        }
        const int rank = static_cast<int>(all.size() / 3);
        cl_mem    dm   = rt_.alloc(all.size() * 4, CL_MEM_READ_ONLY);
        rt_.write(dm, all.size() * 4, all.data());
        clSetKernelArg(kBinB_, 0, sizeof(da), &da);
        clSetKernelArg(kBinB_, 1, sizeof(db), &db);
        clSetKernelArg(kBinB_, 2, sizeof(dy), &dy);
        clSetKernelArg(kBinB_, 3, sizeof(nn), &nn);
        clSetKernelArg(kBinB_, 4, sizeof(op), &op);
        clSetKernelArg(kBinB_, 5, sizeof(rank), &rank);
        clSetKernelArg(kBinB_, 6, sizeof(dm), &dm);
        timed("ew_binary", kBinB_, 1, &g1, nullptr);
        clReleaseMemObject(dm);
      }
      else
      {
        clSetKernelArg(kBin_, 0, sizeof(da), &da);
        clSetKernelArg(kBin_, 1, sizeof(db), &db);
        clSetKernelArg(kBin_, 2, sizeof(dy), &dy);
        clSetKernelArg(kBin_, 3, sizeof(nn), &nn);
        clSetKernelArg(kBin_, 4, sizeof(op), &op);
        clSetKernelArg(kBin_, 5, sizeof(bs), &bs);
        timed("ew_binary", kBin_, 1, &g1, nullptr);
      }
    }
    else if (n.op == "ew_unary")
    {
      int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kUn_, 0, sizeof(dx), &dx);
      clSetKernelArg(kUn_, 1, sizeof(dy), &dy);
      clSetKernelArg(kUn_, 2, sizeof(nn), &nn);
      clSetKernelArg(kUn_, 3, sizeof(op), &op);
      timed("ew_unary", kUn_, 1, &g1, nullptr);
    }
    else if (n.op == "copy_c")
    {
      int HW = attrInt(n, "HW", 1), c0 = attrInt(n, "c0", 0), cnt = attrInt(n, "cnt", 0),
          dst = attrInt(n, "dst_off", 0);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kCopy_, 0, sizeof(dx), &dx);
      clSetKernelArg(kCopy_, 1, sizeof(dy), &dy);
      clSetKernelArg(kCopy_, 2, sizeof(HW), &HW);
      clSetKernelArg(kCopy_, 3, sizeof(c0), &c0);
      clSetKernelArg(kCopy_, 4, sizeof(cnt), &cnt);
      clSetKernelArg(kCopy_, 5, sizeof(dst), &dst);
      const size_t g = static_cast<size_t>(cnt) * HW;
      timed("copy_c", kCopy_, 1, &g, nullptr);
    }
    else if (n.op == "slice_axis")
    {
      int outer = attrInt(n, "outer", 1), axdim = attrInt(n, "axdim", 0), inner = attrInt(n, "inner", 1),
          start = attrInt(n, "start", 0), len = attrInt(n, "len", 0);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kSlice_, 0, sizeof(dx), &dx);
      clSetKernelArg(kSlice_, 1, sizeof(dy), &dy);
      clSetKernelArg(kSlice_, 2, sizeof(outer), &outer);
      clSetKernelArg(kSlice_, 3, sizeof(axdim), &axdim);
      clSetKernelArg(kSlice_, 4, sizeof(inner), &inner);
      clSetKernelArg(kSlice_, 5, sizeof(start), &start);
      clSetKernelArg(kSlice_, 6, sizeof(len), &len);
      const size_t g = static_cast<size_t>(outer) * len * inner;
      timed("slice_axis", kSlice_, 1, &g, nullptr);
    }
    else if (n.op == "concat4")
    {
      int ca = attrInt(n, "ca", 0), cb = attrInt(n, "cb", 0), cc = attrInt(n, "cc", 0),
          cd = attrInt(n, "cd", 0);
      int outer = attrInt(n, "outer", 1), inner = attrInt(n, "inner", 1);
      auto inOr = [&](size_t i) -> cl_mem {
        return (i < n.ins.size() && n.ins[i] != "-" && T_.count(n.ins[i])) ? ref(n.ins[i]).mem
                                                                          : nullptr;
      };
      cl_mem ia = ca ? inOr(0) : nullptr;
      cl_mem ib = cb ? inOr(1) : nullptr;
      cl_mem ic = cc ? inOr(2) : nullptr;
      cl_mem id = cd ? inOr(3) : nullptr;
      cl_mem dy = out.mem;
      clSetKernelArg(kConcat_, 0, sizeof(ia), &ia);
      clSetKernelArg(kConcat_, 1, sizeof(ca), &ca);
      clSetKernelArg(kConcat_, 2, sizeof(ib), &ib);
      clSetKernelArg(kConcat_, 3, sizeof(cb), &cb);
      clSetKernelArg(kConcat_, 4, sizeof(ic), &ic);
      clSetKernelArg(kConcat_, 5, sizeof(cc), &cc);
      clSetKernelArg(kConcat_, 6, sizeof(id), &id);
      clSetKernelArg(kConcat_, 7, sizeof(cd), &cd);
      clSetKernelArg(kConcat_, 8, sizeof(dy), &dy);
      clSetKernelArg(kConcat_, 9, sizeof(outer), &outer);
      clSetKernelArg(kConcat_, 10, sizeof(inner), &inner);
      const size_t g = static_cast<size_t>(outer) * (ca + cb + cc + cd) * inner;
      timed("concat4", kConcat_, 1, &g, nullptr);
    }
    else if (n.op == "maxpool")
    {
      int C = attrInt(n, "C", 0), H = attrInt(n, "H", 0), W = attrInt(n, "W", 0),
          ho = attrInt(n, "Hout", 0), wo = attrInt(n, "Wout", 0), K = attrInt(n, "K", 5),
          S = attrInt(n, "S", 1), P = attrInt(n, "P", 2);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kPool_, 0, sizeof(dx), &dx);
      clSetKernelArg(kPool_, 1, sizeof(dy), &dy);
      clSetKernelArg(kPool_, 2, sizeof(C), &C);
      clSetKernelArg(kPool_, 3, sizeof(H), &H);
      clSetKernelArg(kPool_, 4, sizeof(W), &W);
      clSetKernelArg(kPool_, 5, sizeof(ho), &ho);
      clSetKernelArg(kPool_, 6, sizeof(wo), &wo);
      clSetKernelArg(kPool_, 7, sizeof(K), &K);
      clSetKernelArg(kPool_, 8, sizeof(S), &S);
      clSetKernelArg(kPool_, 9, sizeof(P), &P);
      const size_t g = static_cast<size_t>(C) * ho * wo;
      timed("maxpool", kPool_, 1, &g, nullptr);
    }
    else if (n.op == "resize_nn")
    {
      int C = attrInt(n, "C", 0), H = attrInt(n, "H", 0), W = attrInt(n, "W", 0), S = attrInt(n, "S", 2);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kResize_, 0, sizeof(dx), &dx);
      clSetKernelArg(kResize_, 1, sizeof(dy), &dy);
      clSetKernelArg(kResize_, 2, sizeof(C), &C);
      clSetKernelArg(kResize_, 3, sizeof(H), &H);
      clSetKernelArg(kResize_, 4, sizeof(W), &W);
      clSetKernelArg(kResize_, 5, sizeof(S), &S);
      const size_t g = static_cast<size_t>(C) * H * S * W * S;
      timed("resize_nn", kResize_, 1, &g, nullptr);
    }
    else if (n.op == "softmax_axis")
    {
      int outer = attrInt(n, "outer", 1), axdim = attrInt(n, "axdim", 0), inner = attrInt(n, "inner", 1);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kSoftmax_, 0, sizeof(dx), &dx);
      clSetKernelArg(kSoftmax_, 1, sizeof(dy), &dy);
      clSetKernelArg(kSoftmax_, 2, sizeof(outer), &outer);
      clSetKernelArg(kSoftmax_, 3, sizeof(axdim), &axdim);
      clSetKernelArg(kSoftmax_, 4, sizeof(inner), &inner);
      const size_t g = static_cast<size_t>(outer) * inner;
      timed("softmax_axis", kSoftmax_, 1, &g, nullptr);
    }
    else if (n.op == "permute_0213")
    {
      int D1 = attrInt(n, "D1", 1), D2 = attrInt(n, "D2", 1), I = attrInt(n, "I", 1),
          mode = attrInt(n, "mode", 0);
      cl_mem dx = in(0).mem, dy = out.mem;
      clSetKernelArg(kPerm_, 0, sizeof(dx), &dx);
      clSetKernelArg(kPerm_, 1, sizeof(dy), &dy);
      clSetKernelArg(kPerm_, 2, sizeof(D1), &D1);
      clSetKernelArg(kPerm_, 3, sizeof(D2), &D2);
      clSetKernelArg(kPerm_, 4, sizeof(I), &I);
      clSetKernelArg(kPerm_, 5, sizeof(mode), &mode);
      const size_t g = static_cast<size_t>(D1) * D2 * I;
      timed("permute", kPerm_, 1, &g, nullptr);
    }
    else if (n.op == "bmm")
    {
      auto & A  = in(0);
      auto & B2 = in(1);
      const int B0 = static_cast<int>(A.dims[0]);
      const int B1 = static_cast<int>(A.dims[1]);
      const int M  = static_cast<int>(A.dims[2]);
      const int K  = static_cast<int>(A.dims[3]);
      const int N  = static_cast<int>(B2.dims[3]);
      cl_mem da = A.mem, db = B2.mem, dy = out.mem;
      clSetKernelArg(kBmm_, 0, sizeof(da), &da);
      clSetKernelArg(kBmm_, 1, sizeof(db), &db);
      clSetKernelArg(kBmm_, 2, sizeof(dy), &dy);
      clSetKernelArg(kBmm_, 3, sizeof(B0), &B0);
      clSetKernelArg(kBmm_, 4, sizeof(B1), &B1);
      clSetKernelArg(kBmm_, 5, sizeof(M), &M);
      clSetKernelArg(kBmm_, 6, sizeof(K), &K);
      clSetKernelArg(kBmm_, 7, sizeof(N), &N);
      const size_t g = static_cast<size_t>(B0) * B1 * M * N;
      timed("bmm", kBmm_, 1, &g, nullptr);
    }
    else if (n.op == "gap")
    {
      int C = attrInt(n, "C", 0), HW = attrInt(n, "HW", 1);
      cl_mem dx = in(0).mem, dy = out.mem;
      const int WGS = 128;
      cl_kernel k = getKernel("ops", "gap_r", "-DGAP_WGS=" + std::to_string(WGS));
      clSetKernelArg(k, 0, sizeof(dx), &dx);
      clSetKernelArg(k, 1, sizeof(dy), &dy);
      clSetKernelArg(k, 2, sizeof(C), &C);
      clSetKernelArg(k, 3, sizeof(HW), &HW);
      const size_t lws[1] = {static_cast<size_t>(WGS)};
      const size_t gws[1] = {static_cast<size_t>(C) * WGS};
      timed("gap", k, 1, gws, lws);
    }
    else if (n.op == "bias_add")
    {
      int HW = attrInt(n, "HW", 1);
      cl_mem dx = in(0).mem, db = in(1).mem, dy = out.mem;
      clSetKernelArg(kBias_, 0, sizeof(dx), &dx);
      clSetKernelArg(kBias_, 1, sizeof(db), &db);
      clSetKernelArg(kBias_, 2, sizeof(dy), &dy);
      clSetKernelArg(kBias_, 3, sizeof(HW), &HW);
      timed("bias_add", kBias_, 1, &g1, nullptr);
    }
    else
    {
      throw std::runtime_error("PlanModel: unknown op: " + n.op);
    }
  }

  rt_.finish();
  last_run_ms_ =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void PlanModel::readOutput(size_t i, void * fp16_host)
{
  const Tensor & t = T_.at(outputs_.at(i));
  rt_.read(t.mem, static_cast<size_t>(t.numel()) * 2, fp16_host);
}

}  // namespace gk
