// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.

#include "infvino/PlanModel.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "infvino/Half.hpp"
#include "infvino/Tiles.hpp"
#include "infvino/Autotuner.hpp"

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
  tuning_ = TuningCache::loadDefault();
  tuning_.setDeviceId(TuningCache::deviceKey(rt_.info()));

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
  releaseKernels();
  for (cl_mem m : owned_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_ov_)
    if (m) clReleaseMemObject(m);
  for (cl_mem m : owned_blk_)
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

cl_mem PlanModel::blkInput(const std::string & name, Tensor & x, int Cin, int H, int W)
{
  auto it = blk_in_.find(name);
  if (it != blk_in_.end()) return it->second;
  const size_t bytes = static_cast<size_t>((Cin + 15) / 16) * H * W * 16 * 2;
  cl_mem m = rt_.alloc(bytes, CL_MEM_READ_WRITE);
  blk_in_[name] = m;
  owned_blk_.push_back(m);
  cl_kernel k = getKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
  clSetKernelArg(k, 0, sizeof(x.mem), &x.mem);
  clSetKernelArg(k, 1, sizeof(m), &m);
  clSetKernelArg(k, 2, sizeof(Cin), &Cin);
  clSetKernelArg(k, 3, sizeof(H), &H);
  clSetKernelArg(k, 4, sizeof(W), &W);
  const size_t gws[3] = {static_cast<size_t>(W), static_cast<size_t>(H),
                         static_cast<size_t>(Cin)};
  ClRuntime::enqueueND(rt_.queue(), k, 3, gws, nullptr);
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
    if (n.attr.count("bdims"))
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
      return OpSignature::custom("ew_binary_bcast", {nn, op, bs, C});
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

  if (n.op == "ew_binary" && !n.attr.count("bdims"))
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
    cl_mem da = inMem(0), db = inMem(1);
    clSetKernelArg(k, 0, sizeof(da), &da);
    clSetKernelArg(k, 1, sizeof(db), &db);
    clSetKernelArg(k, 2, sizeof(dy), &dy);
    clSetKernelArg(k, 3, sizeof(nn), &nn);
    clSetKernelArg(k, 4, sizeof(op), &op);
    const int bs = attrInt(n, "b_scalar", 0);
    clSetKernelArg(k, 5, sizeof(bs), &bs);
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
      clSetKernelArg(k, 0, sizeof(da), &da);
      clSetKernelArg(k, 1, sizeof(db), &db);
      clSetKernelArg(k, 2, sizeof(dy), &dy);
      clSetKernelArg(k, 3, sizeof(HW), &HW);
      clSetKernelArg(k, 4, sizeof(C), &C);
      clSetKernelArg(k, 5, sizeof(op), &op);
      clSetKernelArg(k, 6, sizeof(a_ch), &a_ch);
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
    cl_mem dm = bcastDims(spec);
    clSetKernelArg(k, 0, sizeof(da), &da);
    clSetKernelArg(k, 1, sizeof(db), &db);
    clSetKernelArg(k, 2, sizeof(dy), &dy);
    clSetKernelArg(k, 3, sizeof(nn), &nn);
    clSetKernelArg(k, 4, sizeof(op), &op);
    clSetKernelArg(k, 5, sizeof(rank), &rank);
    clSetKernelArg(k, 6, sizeof(dm), &dm);
    gws[0] = static_cast<size_t>(nn);
    return;
  }
  if (n.op == "ew_unary")
  {
    const int nn = static_cast<int>(nout), op = attrInt(n, "op", 0);
    cl_mem dx = inMem(0);
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(nn), &nn);
    clSetKernelArg(k, 3, sizeof(op), &op);
    gws[0] = isVec ? vecN(nn) : static_cast<size_t>(nn);
    return;
  }
  if (n.op == "copy_c")
  {
    int HW = attrInt(n, "HW", 1), c0 = attrInt(n, "c0", 0), cnt = attrInt(n, "cnt", 0),
        dst = attrInt(n, "dst_off", 0);
    cl_mem dx = inMem(0);
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(HW), &HW);
    clSetKernelArg(k, 3, sizeof(c0), &c0);
    clSetKernelArg(k, 4, sizeof(cnt), &cnt);
    clSetKernelArg(k, 5, sizeof(dst), &dst);
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
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(outer), &outer);
    clSetKernelArg(k, 3, sizeof(axdim), &axdim);
    clSetKernelArg(k, 4, sizeof(inner), &inner);
    clSetKernelArg(k, 5, sizeof(start), &start);
    clSetKernelArg(k, 6, sizeof(len), &len);
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
    clSetKernelArg(k, 0, sizeof(ia), &ia);
    clSetKernelArg(k, 1, sizeof(ca), &ca);
    clSetKernelArg(k, 2, sizeof(ib), &ib);
    clSetKernelArg(k, 3, sizeof(cb), &cb);
    clSetKernelArg(k, 4, sizeof(ic), &ic);
    clSetKernelArg(k, 5, sizeof(cc), &cc);
    clSetKernelArg(k, 6, sizeof(id), &id);
    clSetKernelArg(k, 7, sizeof(cd), &cd);
    clSetKernelArg(k, 8, sizeof(dy), &dy);
    clSetKernelArg(k, 9, sizeof(outer), &outer);
    clSetKernelArg(k, 10, sizeof(inner), &inner);
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
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(C), &C);
    clSetKernelArg(k, 3, sizeof(H), &H);
    clSetKernelArg(k, 4, sizeof(W), &W);
    clSetKernelArg(k, 5, sizeof(ho), &ho);
    clSetKernelArg(k, 6, sizeof(wo), &wo);
    clSetKernelArg(k, 7, sizeof(K), &K);
    clSetKernelArg(k, 8, sizeof(S), &S);
    clSetKernelArg(k, 9, sizeof(P), &P);
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
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(C), &C);
    clSetKernelArg(k, 3, sizeof(H), &H);
    clSetKernelArg(k, 4, sizeof(W), &W);
    clSetKernelArg(k, 5, sizeof(S), &S);
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
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(D1), &D1);
    clSetKernelArg(k, 3, sizeof(D2), &D2);
    clSetKernelArg(k, 4, sizeof(I), &I);
    clSetKernelArg(k, 5, sizeof(mode), &mode);
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
    clSetKernelArg(k, 0, sizeof(da), &da);
    clSetKernelArg(k, 1, sizeof(db), &db);
    clSetKernelArg(k, 2, sizeof(dy), &dy);
    clSetKernelArg(k, 3, sizeof(B0), &B0);
    clSetKernelArg(k, 4, sizeof(B1), &B1);
    clSetKernelArg(k, 5, sizeof(M), &M);
    clSetKernelArg(k, 6, sizeof(K), &K);
    clSetKernelArg(k, 7, sizeof(N), &N);
    if (kernel == "bmm2")
    {
      dim = 3;
      gws[0] = static_cast<size_t>(N);
      gws[1] = static_cast<size_t>(M);
      gws[2] = static_cast<size_t>(B0) * static_cast<size_t>(B1);
    }
    else
    {
      gws[0] = static_cast<size_t>(B0) * static_cast<size_t>(B1) * static_cast<size_t>(M) *
               static_cast<size_t>(N);
    }
    return;
  }
  if (n.op == "gap")
  {
    int C = attrInt(n, "C", 0), HW = attrInt(n, "HW", 1);
    cl_mem dx = inMem(0);
    const int wgs = std::max(1, optInt("-DGAP_WGS=", 128));
    clSetKernelArg(k, 0, sizeof(dx), &dx);
    clSetKernelArg(k, 1, sizeof(dy), &dy);
    clSetKernelArg(k, 2, sizeof(C), &C);
    clSetKernelArg(k, 3, sizeof(HW), &HW);
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
  if (kernel == "conv3x3_f16" || kernel == "conv3x3_rt" || kernel == "conv3x3_db") return "conv";
  if (kernel == "conv3x3_sg") return "conv_sg";
  if (kernel == "conv3x3_osv") return "conv_osv";
  if (kernel == "gemm_f16") return "gemm";
  if (kernel == "conv1x1_gemv_f16" || kernel == "conv1x1_f16") return "conv1x1";
  if (kernel == "depthwise_f16" || kernel == "conv_general") return "conv_general";
  return "ops";
}
}  // namespace

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
    cl_event ev = nullptr;
    try {
      ev = ClRuntime::enqueueND(rt_.queue(), k, dim, gws, lws);
    } catch (const std::exception & e) {
      throw std::runtime_error("node " + tag + ": " + e.what());
    }
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
    const size_t  g1   = static_cast<size_t>(out_numel());

    // Round 28: 小算子调优查表 —— 命中缓存用其 kernel/options，否则内置默认。
    // 只影响「用哪个变体」，数值由 kernel 语义决定。
    auto smallKernelFor = [&](const OpSignature & sig, const char * dk, std::string & kernOut,
                              std::string & optsOut) -> cl_kernel {
      kernOut = dk;
      optsOut.clear();
      if (const TuningEntry * e = tuning_.lookup(sig)) {
        kernOut = e->kernel;
        optsOut = e->options;
      }
      return getKernel("ops", kernOut, optsOut);
    };

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
        std::string gopts = cfg.options();
        const OpSignature sig = OpSignature::conv1x1(Cout, N, Cin, act, dres ? 1 : 0);
        if (const TuningEntry * e = tuning_.lookup(sig)) gopts = e->options;
        cl_kernel kg = getKernel("conv1x1", "conv1x1_gemv_f16", gopts);
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
        std::string gopts = t.options();
        const OpSignature sig = OpSignature::conv1x1(Cout, N, Cin, act, dres ? 1 : 0);
        if (const TuningEntry * e = tuning_.lookup(sig)) {
          gopts = e->options;
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
        std::string dwopts = dopts;
        const OpSignature sig = OpSignature::depthwise(Wout, Hout, S, P, Cin, K, act);
        if (const TuningEntry * e = tuning_.lookup(sig)) dwopts = e->options;
        cl_kernel kd = getKernel("conv_general", "depthwise_f16", dwopts);
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
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? in(3).mem : nullptr;

      // 自动调优：按签名查表。命中 → 用缓存里的 kernel/options；未命中 → 内置启发式。
      // 数值与 kernel 语义不变，只改「选哪个 kernel/config」。
      const OpSignature tsig =
        OpSignature::conv3x3(Wout, Hout, stride, pad, Cin, Cout, act);
      const TuningEntry * te = tuning_.lookup(tsig);

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
                std::to_string(Cout) + "(tuned)",
              kk, 3, gws, lws);
      }
      else if ((te && te->kernel == "conv3x3_blk") || (attrInt(n, "blk", 0) != 0 && !te))
      {
        // R25: OpenVINO blocked conv port (kernels/conv_blk.cl). Reorders the
        // input bfyx -> b_fs_yx_fsv16 (cached scratch) then runs the blocked
        // kernel, writing plain bfyx output.
        int obw = 8;
        char oo[192];
        if (te) {
          auto p = te->options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(te->options.c_str() + p + 6);
          std::snprintf(oo, sizeof(oo), "%s", te->options.c_str());
        } else {
          std::snprintf(oo, sizeof(oo),
                        "-DOBW=%d -DSTRIDE=%d -DPAD=%d -DACT=%d -DSG=16 "
                        "-cl-mad-enable -cl-fast-relaxed-math", obw, stride, pad, act);
        }
        cl_kernel kk = getKernel("conv_blk", "conv3x3_blk", oo);
        cl_mem dw = blkWeight(n.ins[1], in(1), Cout, Cin);
        cl_mem dxb = blkInput(n.ins[0], in(0), Cin, H, W);
        clSetKernelArg(kk, 0, sizeof(dxb), &dxb);
        clSetKernelArg(kk, 1, sizeof(dw), &dw);
        clSetKernelArg(kk, 2, sizeof(db), &db);
        clSetKernelArg(kk, 3, sizeof(dy), &dy);
        clSetKernelArg(kk, 4, sizeof(Cin), &Cin);
        clSetKernelArg(kk, 5, sizeof(H), &H);
        clSetKernelArg(kk, 6, sizeof(W), &W);
        clSetKernelArg(kk, 7, sizeof(Cout), &Cout);
        clSetKernelArg(kk, 8, sizeof(Hout), &Hout);
        clSetKernelArg(kk, 9, sizeof(Wout), &Wout);
        const size_t lws[3] = {1, 16, 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + obw - 1) / obw) * static_cast<size_t>(Hout),
          static_cast<size_t>(((Cout + 15) / 16) * 16), 1};
        timed("conv3x3blk@" + std::to_string(Wout) + "x" + std::to_string(Hout) + "s" +
                std::to_string(stride) + "_Cin" + std::to_string(Cin) + "_Cout" +
                std::to_string(Cout) + (te ? "(tuned)" : ""),
              kk, 3, gws, lws);
      }
      else if (te && te->kernel == "conv3x3_ov")
      {
        // 调优命中的 OV osv32：options 里已含 OBW/OBH/STRIDE/PAD/ACT/RES/SG。
        // 从 options 解析 OBW/OBH 以重建 grid（避免再解析 config 串）。
        int obw = 8, obh = 2;
        {
          auto p = te->options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(te->options.c_str() + p + 6);
          p = te->options.find("-DOBH=");
          if (p != std::string::npos) obh = std::atoi(te->options.c_str() + p + 6);
        }
        cl_kernel kk = getKernel(sourceOfKernel(te->kernel), te->kernel, te->options);
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
      std::string gopts = t.options();
      // 自动调优：命中则用缓存的 tile（options 已含全部编译宏）。
      const OpSignature gsig = OpSignature::gemm(M, N, K, 0);
      if (const TuningEntry * ge = tuning_.lookup(gsig)) {
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
      const OpSignature sig = smallSig(n);
      std::string kern, opts;
      cl_kernel   k = smallKernelFor(sig, sig.op == "ew_binary_bcast" ? "ew_binary_bcast"
                                                                     : "ew_binary",
                                     kern, opts);
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

std::vector<std::string> PlanModel::tuningTargets(const std::vector<std::string> & ops) const
{
  std::vector<std::string> out;
  std::map<std::string, int> seen;
  auto add = [&](const OpSignature & s) {
    const std::string k = s.str();
    if (!seen.count(k)) { seen[k] = 1; out.push_back(k); }
  };
  auto want = [&](const std::string & op) {
    return ops.empty() || std::find(ops.begin(), ops.end(), op) != ops.end();
  };
  for (const auto & n : nodes_)
  {
    if (n.op == "conv3x3" && want("conv3x3")) {
      const int stride = attrInt(n, "stride", 1), pad = attrInt(n, "pad", 1), act = attrInt(n, "act", 0);
      const auto & id = T_.at(n.ins[0]).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      const int Cin = (int)id[base];
      const auto & od = T_.at(n.outs[0]).dims;
      const int Cout = (int)od[od.size() >= 3 ? od.size() - 3 : 0];
      add(OpSignature::conv3x3(attrInt(n, "Wout", 0), attrInt(n, "Hout", 0), stride, pad, Cin, Cout, act));
    } else if (n.op == "gemm" && want("gemm")) {
      const auto & ad = T_.at(n.ins[0]).dims;
      const int M = (int)ad[0], K = (int)ad[1];
      const int N = (int)(T_.at(n.ins[1]).numel() / K);
      add(OpSignature::gemm(M, N, K, 0));
    } else if (n.op == "conv1x1" && want("conv1x1")) {
      const int act = attrInt(n, "act", 0);
      const auto & wd = T_.at(n.ins[0]).dims;
      const int Cout = (int)wd[0], Cin = (int)wd[1];
      const int N = Cin > 0 ? (int)(T_.at(n.ins[1]).numel() / Cin) : 0;
      const bool res = n.ins.size() > 3 && n.ins[3] != "-";
      add(OpSignature::conv1x1(Cout, N, Cin, act, res ? 1 : 0));
    } else if (n.op == "conv_general" && want("depthwise")) {
      const int K = attrInt(n, "K", 3), S = attrInt(n, "S", 1), P = attrInt(n, "P", 1),
                G = attrInt(n, "G", 1), act = attrInt(n, "act", 0);
      const auto & id = T_.at(n.ins[0]).dims;
      const size_t base = id.size() >= 3 ? id.size() - 3 : 0;
      const int Cin = (int)id[base];
      const auto & od = T_.at(n.outs[0]).dims;
      const int Cout = (int)od[od.size() >= 3 ? od.size() - 3 : 0];
      if (G == Cin && (K == 3 || K == 5) && Cin == Cout)
        add(OpSignature::depthwise(attrInt(n, "Wout", 0), attrInt(n, "Hout", 0), S, P, Cin, K, act));
    } else if (want(n.op) &&
               (n.op == "ew_binary" || n.op == "ew_unary" || n.op == "copy_c" ||
                n.op == "slice_axis" || n.op == "concat4" || n.op == "maxpool" ||
                n.op == "resize_nn" || n.op == "permute_0213" || n.op == "bmm" ||
                n.op == "gap")) {
      // Round 28: 小算子签名（与 dispatch/autotune 完全一致）。
      add(smallSig(n));
    }
  }
  return out;
}

std::map<std::string, TuningEntry> PlanModel::autotune(
  const std::vector<std::string> & ops, const std::string & onlySubstr, int limit, int iters,
  bool merge, bool verbose, bool retune)
{
  std::map<std::string, TuningEntry> done;
  int n_tuned = 0;
  std::map<std::string, int> sig_seen;  // 同一签名只调一次（plan 里大量层共享签名）
  // 返回 true 表示该签名还未调过（并登记）；false 表示跳过。
  // 已在本进程调过、或缓存里已有 source=="tuned" 的条目 → 跳过（让分批调用自然推进）。
  // retune=true 时忽略已有 tuned 条目（用于候选集/中间标准更新后重扫）。
  auto shouldTune = [&](const OpSignature & sig) {
    const std::string k = sig.str();
    if (sig_seen.count(k)) return false;
    sig_seen[k] = 1;
    if (!retune)
      if (const TuningEntry * e = tuning_.lookup(sig))
        if (e->source == "tuned") return false;
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
      if (!shouldTune(sig)) continue;
      const double flops = 2.0 * Cout * static_cast<double>(Hout) * Wout * Cin * 9.0;
      const std::vector<Candidate> cands = candidatesConv3x3(sig);

      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        if (c.kernel == "conv3x3_ov") {
          cl_kernel kk = getKernel("conv_ov", "conv3x3_ov", c.options);
          cl_mem dw = ovWeight(n.ins[1], ref(n.ins[1]), Cout, Cin);
          int obw = 8, obh = 2;
          auto p = c.options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(c.options.c_str() + p + 6);
          p = c.options.find("-DOBH=");
          if (p != std::string::npos) obh = std::atoi(c.options.c_str() + p + 6);
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
          return [this, kk, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
          };
        }
        if (c.kernel == "conv3x3_blk") {
          int obw = 8;
          auto p = c.options.find("-DOBW=");
          if (p != std::string::npos) obw = std::atoi(c.options.c_str() + p + 6);
          cl_kernel kk = getKernel("conv_blk", "conv3x3_blk", c.options);
          cl_mem dw = blkWeight(n.ins[1], ref(n.ins[1]), Cout, Cin);
          cl_mem dxb = blkInput(n.ins[0], ref(n.ins[0]), Cin, H, W);
          clSetKernelArg(kk, 0, sizeof(dxb), &dxb);
          clSetKernelArg(kk, 1, sizeof(dw), &dw);
          clSetKernelArg(kk, 2, sizeof(db), &db);
          clSetKernelArg(kk, 3, sizeof(dy), &dy);
          clSetKernelArg(kk, 4, sizeof(Cin), &Cin);
          clSetKernelArg(kk, 5, sizeof(H), &H);
          clSetKernelArg(kk, 6, sizeof(W), &W);
          clSetKernelArg(kk, 7, sizeof(Cout), &Cout);
          clSetKernelArg(kk, 8, sizeof(Hout), &Hout);
          clSetKernelArg(kk, 9, sizeof(Wout), &Wout);
          const size_t lws[3] = {1, 16, 1};
          const size_t gws[3] = {
            static_cast<size_t>((Wout + obw - 1) / obw) * static_cast<size_t>(Hout),
            static_cast<size_t>(((Cout + 15) / 16) * 16), 1};
          return [this, kk, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
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
          static_cast<size_t>(TX / TM), static_cast<size_t>(TY), 1};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + TX - 1) / TX) * lws[0],
          static_cast<size_t>((Hout + TY - 1) / TY) * lws[1],
          static_cast<size_t>((Cout + CB - 1) / CB)};
        return [this, kk, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kk, 3, gws, lws);
        };
      };

      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters);
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
      if (!shouldTune(sig)) continue;
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
        clSetKernelArg(kg, 0, sizeof(da), &da);
        clSetKernelArg(kg, 1, sizeof(db), &db);
        clSetKernelArg(kg, 2, sizeof(dc), &dc);
        clSetKernelArg(kg, 3, sizeof(M), &M);
        clSetKernelArg(kg, 4, sizeof(N), &N);
        clSetKernelArg(kg, 5, sizeof(K), &K);
        const size_t lws[2] = {static_cast<size_t>(BN / TN), static_cast<size_t>(BM / TM)};
        const size_t gws[2] = {
          static_cast<size_t>((N + BN - 1) / BN) * lws[0],
          static_cast<size_t>((M + BM - 1) / BM) * lws[1]};
        return [this, kg, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kg, 2, gws, lws);
        };
      };
      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters);
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

    if (n.op == "conv1x1" && opInList(ops, "conv1x1"))
    {
      const int act = attrInt(n, "act", 0);
      auto & w = ref(n.ins[0]);
      const int Cout = static_cast<int>(w.dims[0]);
      const int Cin  = static_cast<int>(w.dims[1]);
      const int64_t xnumel = ref(n.ins[1]).numel();
      const int N = (Cin > 0) ? static_cast<int>(xnumel / Cin) : 0;
      cl_mem dw = w.mem, dx = ref(n.ins[1]).mem, dy = ref(n.outs[0]).mem;
      cl_mem db = (n.ins.size() > 2 && n.ins[2] != "-") ? ref(n.ins[2]).mem : nullptr;
      cl_mem dres = (n.ins.size() > 3 && n.ins[3] != "-") ? ref(n.ins[3]).mem : nullptr;
      const OpSignature sig = OpSignature::conv1x1(Cout, N, Cin, act, dres ? 1 : 0);
      if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
      if (!shouldTune(sig)) continue;
      const double flops = 2.0 * Cout * static_cast<double>(N) * Cin;
      const std::vector<Candidate> cands = candidatesConv1x1(sig);
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        if (N == 1) {
          cl_kernel kg = getKernel("conv1x1", "conv1x1_gemv_f16", c.options);
          clSetKernelArg(kg, 0, sizeof(dw), &dw);
          clSetKernelArg(kg, 1, sizeof(dx), &dx);
          clSetKernelArg(kg, 2, sizeof(db), &db);
          clSetKernelArg(kg, 3, sizeof(dres), &dres);
          clSetKernelArg(kg, 4, sizeof(dy), &dy);
          clSetKernelArg(kg, 5, sizeof(Cin), &Cin);
          clSetKernelArg(kg, 6, sizeof(Cout), &Cout);
          const size_t lws[1] = {16};
          const size_t gws[1] = {static_cast<size_t>(Cout) * 16};
          return [this, kg, gws, lws]() {
            return ClRuntime::enqueueND(rt_.queue(), kg, 1, gws, lws);
          };
        }
        cl_kernel kg = getKernel("gemm", "gemm_f16", c.options);
        auto optInt = [&](const char * k, int def) {
          const auto p = c.options.find(k);
          return p == std::string::npos ? def : std::atoi(c.options.c_str() + p + std::strlen(k));
        };
        const int BM = optInt("-DBM=", 128), BN = optInt("-DBN=", 64), TM = optInt("-DTM=", 8),
                  TN = optInt("-DTN=", 4);
        clSetKernelArg(kg, 0, sizeof(dw), &dw);
        clSetKernelArg(kg, 1, sizeof(dx), &dx);
        clSetKernelArg(kg, 2, sizeof(dy), &dy);
        clSetKernelArg(kg, 3, sizeof(Cout), &Cout);
        clSetKernelArg(kg, 4, sizeof(N), &N);
        clSetKernelArg(kg, 5, sizeof(Cin), &Cin);
        clSetKernelArg(kg, 6, sizeof(db), &db);
        clSetKernelArg(kg, 7, sizeof(dres), &dres);
        const size_t lws[2] = {static_cast<size_t>(BN / TN), static_cast<size_t>(BM / TM)};
        const size_t gws[2] = {
          static_cast<size_t>((N + BN - 1) / BN) * lws[0],
          static_cast<size_t>((Cout + BM - 1) / BM) * lws[1]};
        return [this, kg, gws, lws]() {
          return ClRuntime::enqueueND(rt_.queue(), kg, 2, gws, lws);
        };
      };
      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters);
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
      if (!shouldTune(sig)) continue;
      const double flops = 2.0 * Cin * static_cast<double>(Hout) * Wout * K * K;
      const std::vector<Candidate> cands = candidatesDepthwise(sig);
      auto makeEnqueue = [&](const Candidate & c) -> std::function<cl_event()> {
        cl_kernel kd = getKernel("conv_general", "depthwise_f16", c.options);
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
        return [this, kd, gdw]() {
          return ClRuntime::enqueueND(rt_.queue(), kd, 1, gdw, nullptr);
        };
      };
      TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters);
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
         n.op == "gap");
      if (smallOp && opInList(ops, n.op))
      {
        const OpSignature sig = smallSig(n);
        if (!onlySubstr.empty() && sig.str().find(onlySubstr) == std::string::npos) continue;
        if (!shouldTune(sig)) continue;
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
        TuningEntry e = autotuneOp(rt_, sig, cands, makeEnqueue, flops, iters);
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

  return done;
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

}  // namespace gk
