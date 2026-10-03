// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Autotuner 实现：候选枚举 + 计时。
#include "infvino/Autotuner.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <stdexcept>

namespace gk
{

namespace
{
std::string ovConfig(int obw, int obh, int stride, int pad, int act, int res)
{
  std::ostringstream o;
  o << "OBW=" << obw << ",OBH=" << obh << ",STRIDE=" << stride << ",PAD=" << pad
    << ",ACT=" << act << ",RES=" << res;
  return o.str();
}
std::string ovOptions(int obw, int obh, int stride, int pad, int act, int res)
{
  std::ostringstream o;
  o << "-DOBW=" << obw << " -DOBH=" << obh << " -DSTRIDE=" << stride << " -DPAD=" << pad
    << " -DACT=" << act << " -DRES=" << res
    << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
  return o.str();
}
}  // namespace

std::vector<Candidate> candidatesConv3x3(const OpSignature & sig)
{
  std::vector<Candidate> out;
  // OV osv32 的 block 谱系（R23 扫描过 8x2/5x2/6x2/4x4；再加 s2 的候选）。
  // R24：ISA 反汇编显示 OV 内循环是 288 packed mad / 453 指令（mad 占 63.6%），
  // `sub_group_broadcast` 已折进 mad 操作数——上限 ≈ 20，缺口在延迟/流水（见
  // docs/kernel.md Round 24）。block 仍是主要可调维度。
  std::vector<std::pair<int, int>> blocks;
  if (sig.stride == 2) {
    blocks = {{5, 4}, {4, 4}, {5, 2}, {6, 2}, {8, 2}, {3, 4}};
  } else {
    blocks = {{8, 2}, {5, 2}, {6, 2}, {4, 4}, {8, 1}, {4, 2}, {10, 2}, {8, 4}};
  }
  for (auto [obw, obh] : blocks) {
    if (sig.W > 0 && obw > sig.W) continue;   // 超过输出宽度无意义
    if (sig.H > 0 && obh > sig.H) continue;
    // Round 28 (P2)：形状整除时编译期去掉输出谓词（数值不变，IGC 直接折叠）。
    std::string extra;
    if (sig.W > 0 && sig.H > 0 && sig.W % obw == 0 && sig.H % obh == 0)
      extra += " -DFIT_WH=1";
    if (sig.Cout > 0 && sig.Cout % 32 == 0) extra += " -DFIT_COUT=1";
    Candidate c;
    c.kernel = "conv3x3_ov";
    c.source = "conv_ov";
    c.options = ovOptions(obw, obh, sig.stride, sig.pad, sig.act, sig.groups == 2 ? 1 : 0) + extra;
    c.config = ovConfig(obw, obh, sig.stride, sig.pad, sig.act, sig.groups == 2 ? 1 : 0) +
               (extra.empty() ? "" : " fit");
    out.push_back(std::move(c));
  }
  // R25: OpenVINO blocked conv port (kernels/conv_blk.cl).  Lane=output channel,
  // OBW consecutive output columns per lane, blocked input + vector mads; grid is
  // 16 channels x 1 row per WG (2-4x more WGs than osv32).  Wins on small-spatial
  // / large-channel s1 layers; the tuner picks per size.
  for (int obw : {2, 4, 8}) {
    if (sig.W > 0 && obw > sig.W) continue;
    Candidate c;
    c.kernel = "conv3x3_blk";
    c.source = "conv_blk";
    {
      std::ostringstream o;
      o << "-DOBW=" << obw << " -DSTRIDE=" << sig.stride << " -DPAD=" << sig.pad
        << " -DACT=" << sig.act << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
      if (sig.W > 0 && sig.W % obw == 0) o << " -DFIT_WH=1";
      if (sig.Cout > 0 && sig.Cout % 16 == 0) o << " -DFIT_COUT=1";
      if (sig.Cin > 0 && sig.Cin % 16 == 0) o << " -DFIT_CIN=1";
      c.options = o.str();
      std::ostringstream cc;
      cc << "OBW=" << obw << ",STRIDE=" << sig.stride << ",PAD=" << sig.pad
         << ",ACT=" << sig.act;
      c.config = cc.str();
    }
    out.push_back(std::move(c));
  }
  // native conv3x3_f16（R18 自适应 tile 谱系）作为第二条通路候选——所有 shape 都枚举
  //（含 stride=2），让调优器按 size 在「lane=通道 OV」与「lane=空间 SLM native」之间选。
  // R22 起 OV 通常赢，但小通道 shape 上 native 偶尔更优（如 40×40 Cin32 Cout64、
  // 160×160 Cin16 Cout16）；R23 记录 stride-2 native 用 TX=40 可 build。
  for (int tx : {40, 20}) {
    if (sig.W > 0 && tx > sig.W) continue;
    for (int cb : {32, 16}) {
      Conv3x3Cfg cfg;
      cfg.TX = tx; cfg.TY = 8; cfg.TM = 1; cfg.CB = cb; cfg.CINC = 16;
      cfg.STRIDE = sig.stride; cfg.PAD = sig.pad; cfg.ACT = sig.act;
      cfg.SG = 16; cfg.WC = 1;
      std::string opts = cfg.options();
      if (sig.W > 0 && sig.H > 0 && sig.W % tx == 0 && sig.H % 8 == 0) opts += " -DFIT_WH=1";
      if (sig.Cin > 0 && sig.Cin % 16 == 0) opts += " -DFIT_CIN=1";
      if (sig.Cout > 0 && sig.Cout % cb == 0) opts += " -DFIT_CB=1";
      Candidate c;
      c.kernel = "conv3x3_f16";
      c.source = "conv";
      c.options = opts;
      c.config = cfg.label();
      out.push_back(std::move(c));
    }
  }
  return out;
}

std::vector<Candidate> candidatesGemm(const OpSignature & sig)
{
  std::vector<Candidate> out;
  // R9/R12 谱系：BK=32 DBUF=0（大网格）、BK=16 DBUF=1（双缓冲）、BK=8 DBUF=0（小网格/小 K）。
  struct Opt { int BM, BN, BK, TM, TN, DBUF; };
  const Opt opts[] = {
    {128, 64, 32, 8, 4, 0},
    {128, 64, 16, 8, 4, 1},
    {128, 64, 8,  8, 4, 0},
    {64,  64, 16, 8, 4, 1},
    {64,  64, 8,  8, 4, 0},
    // Round 28 (P1): 小 M/N 形状（conv1x1 的 expansion/projection 常见）的更小块，
    // 减少尾部浪费/增加 work-group 数；大块网格饥饿时由调优器选。
    {64,  32, 32, 8, 4, 0},
    {32,  64, 32, 8, 4, 0},
  };
  for (const auto & o : opts) {
    Tiles t;
    t.BM = o.BM; t.BN = o.BN; t.BK = o.BK; t.TM = o.TM; t.TN = o.TN; t.DBUF = o.DBUF;
    if (sig.K < 32) t.SG = 0;  // R12: 极小 K 上 SG=16 略负
    Candidate c;
    c.kernel = "gemm_f16";
    c.source = "gemm";
    c.options = t.options();
    c.config = t.label();
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<Candidate> candidatesConv1x1(const OpSignature & sig)
{
  std::vector<Candidate> out;
  if (sig.N == 1) {
    // split-K GEMV：只有 SG 可调（R22 固定 16）。
    Conv1x1Cfg cfg;
    cfg.ACT = sig.act;
    cfg.RES = (sig.groups == 2) ? 1 : 0;
    cfg.SG = 16;
    Candidate c;
    c.kernel = "conv1x1_gemv_f16";
    c.source = "conv1x1";
    c.options = cfg.options();
    c.config = cfg.label();
    out.push_back(std::move(c));
    return out;
  }
  // N>1 复用 gemm_f16 + 融合 epilogue（R22），候选即 gemm 的 tile 谱系。
  // 用 groups==2 表示带残差；这里参数逐一对齐 PlanModel 的 conv1x1 N>1 分支。
  std::vector<Candidate> g = candidatesGemm(OpSignature::gemm(sig.Cout, sig.N, sig.Cin, sig.act));
  for (auto & c : g) {
    // 在 options 上追加融合标记（PlanModel 分支按 EPI/ACT 设置；此处保持一致）。
    c.options += " -DEPI=1 -DACT=" + std::to_string(sig.act);
    c.config += " epi=1 act=" + std::to_string(sig.act);
    out.push_back(std::move(c));
  }
  return out;
}

std::vector<Candidate> candidatesDepthwise(const OpSignature & sig)
{
  std::vector<Candidate> out;
  if (sig.K != 3 && sig.K != 5) return out;
  // R16 的 coalesced 标量版（compile-time K/S/P/ACT）。
  char opts[160];
  std::snprintf(opts, sizeof(opts),
                "-DDW_K=%d -DDW_S=%d -DDW_P=%d -DDW_ACT=%d -cl-mad-enable -cl-fast-relaxed-math",
                sig.K, sig.stride, sig.pad, sig.act);
  char cfg[96];
  std::snprintf(cfg, sizeof(cfg), "K=%d,S=%d,P=%d,ACT=%d", sig.K, sig.stride, sig.pad, sig.act);
  Candidate c;
  c.kernel = "depthwise_f16";
  c.source = "conv_general";
  c.options = opts;
  c.config = cfg;
  out.push_back(std::move(c));
  // R29: register-blocked sliding window (DW_TW outputs/WI, strip reused).
  for (int tw : {2, 4, 8}) {
    if (sig.W > 0 && tw > sig.W) continue;
    char vo[192], vc[128];
    std::snprintf(vo, sizeof(vo),
                  "-DDW_K=%d -DDW_S=%d -DDW_P=%d -DDW_ACT=%d -DDW_TW=%d "
                  "-cl-mad-enable -cl-fast-relaxed-math",
                  sig.K, sig.stride, sig.pad, sig.act, tw);
    std::snprintf(vc, sizeof(vc), "K=%d,S=%d,P=%d,ACT=%d,TW=%d", sig.K, sig.stride, sig.pad,
                  sig.act, tw);
    Candidate v;
    v.kernel = "depthwise_v";
    v.source = "conv_general";
    v.options = vo;
    v.config = vc;
    out.push_back(std::move(v));
  }
  // R31: padded (boundary-free) variant.  The input is pre-padded once per frame by
  // `depthwise_pad`; that extra pass is measured separately in PlanModel::autotune and
  // added to this candidate's cost, so the selection accounts for it.  **Opt-in only**:
  // the measured whole-net result is a wash (pad is bandwidth-bound and cancels the
  // boundary-predicate saving), so vp is not enumerated by default.  See docs/kernel.md.
  if (std::getenv("INFVINO_DW_PAD"))
  for (int tw : {2, 4, 8}) {
    if (sig.W > 0 && tw > sig.W) continue;
    char vo[192], vc[128];
    std::snprintf(vo, sizeof(vo),
                  "-DDW_K=%d -DDW_S=%d -DDW_P=%d -DDW_ACT=%d -DDW_TW=%d "
                  "-cl-mad-enable -cl-fast-relaxed-math",
                  sig.K, sig.stride, sig.pad, sig.act, tw);
    std::snprintf(vc, sizeof(vc), "K=%d,S=%d,P=%d,ACT=%d,TW=%d,PAD", sig.K, sig.stride, sig.pad,
                  sig.act, tw);
    Candidate v;
    v.kernel = "depthwise_vp";
    v.source = "conv_general";
    v.options = vo;
    v.config = vc;
    out.push_back(std::move(v));
  }
  return out;
}

namespace
{
void addSmall(std::vector<Candidate> & out, const char * kernel, const char * src,
              const std::string & opts, const std::string & cfg)
{
  Candidate c;
  c.kernel = kernel;
  c.source = src;
  c.options = opts;
  c.config = cfg;
  out.push_back(std::move(c));
}
std::string vecOpts(const char * macro, int v)
{
  std::ostringstream o;
  o << "-D" << macro << "=" << v << " -cl-mad-enable -cl-fast-relaxed-math";
  return o.str();
}
}  // namespace

std::vector<Candidate> candidatesSmall(const OpSignature & sig)
{
  std::vector<Candidate> out;
  const std::string op = sig.op;
  if (op == "ew_binary" || op == "ew_unary")
  {
    const bool binary = (op == "ew_binary");
    addSmall(out, binary ? "ew_binary" : "ew_unary", "ops", "", "scalar");
    for (int v : {2, 4, 8})
      addSmall(out, binary ? "ew_binary_v" : "ew_unary_v", "ops", vecOpts("EW_VEC", v),
               "VEC=" + std::to_string(v));
    return out;
  }
  if (op == "ew_binary_bcast")
  {
    addSmall(out, "ew_binary_bcast", "ops", "", "bcast");
    // R30: 3-D 网格 + 运行期 stride（去掉逐元素 div/mod）；有效秩 ≤4 时可用。
    const int effRank = sig.params.size() > 4 ? sig.params[4] : 4;
    if (effRank <= 4)
      addSmall(out, "ew_binary_bcast4", "ops", "", "grid3");
    // 仅当其中一个是「每通道一个标量」时才加入通道特化变体。
    const int n = sig.params.size() > 0 ? sig.params[0] : 0;
    const int C = sig.params.size() > 3 ? sig.params[3] : 0;
    if (C > 0 && C < n && n % C == 0)
      addSmall(out, "ew_binary_ch", "ops", "", "channel");
    return out;
  }
  if (op == "concat4")
  {
    addSmall(out, "concat4", "ops", "", "scalar");
    for (int v : {2, 4, 8})
      addSmall(out, "concat4_v", "ops", vecOpts("EW_VEC", v), "VEC=" + std::to_string(v));
    return out;
  }
  if (op == "copy_c")
  {
    addSmall(out, "copy_c", "ops", "", "grid1");
    addSmall(out, "copy_c2", "ops", "", "grid2");
    return out;
  }
  if (op == "slice_axis")
  {
    addSmall(out, "slice_axis", "ops", "", "grid1");
    addSmall(out, "slice_axis3", "ops", "", "grid3");
    return out;
  }
  if (op == "maxpool")
  {
    addSmall(out, "maxpool", "ops", "", "grid1");
    addSmall(out, "maxpool3", "ops", "", "grid3");
    return out;
  }
  if (op == "resize_nn")
  {
    addSmall(out, "resize_nn", "ops", "", "grid1");
    addSmall(out, "resize_nn3", "ops", "", "grid3");
    return out;
  }
  if (op == "permute_0213")
  {
    addSmall(out, "permute_0213", "ops", "", "grid1");
    addSmall(out, "permute_0213_3d", "ops", "", "grid3");
    return out;
  }
  if (op == "bmm")
  {
    addSmall(out, "bmm", "ops", "", "grid1");
    addSmall(out, "bmm2", "ops", "", "grid3");
    // Register-tiled variants (explicit x reuse).  BMM_TM x BMM_TN outputs/WI.
    for (auto [tm, tn] : {std::pair<int,int>{4,8}, {8,8}, {8,4}, {4,4}}) {
      for (int uk : {4, 8}) {
        std::ostringstream o;
        o << "-DBMM_TM=" << tm << " -DBMM_TN=" << tn << " -DBMM_UK=" << uk
          << " -cl-mad-enable -cl-fast-relaxed-math";
        addSmall(out, "bmm_t", "ops", o.str(),
                 "TM" + std::to_string(tm) + "x" + std::to_string(tn) + "u" + std::to_string(uk));
      }
    }
    return out;
  }
  if (op == "softmax_axis")
  {
    addSmall(out, "softmax_axis", "ops", "", "serial");
    for (int w : {32, 64, 128, 256})
      addSmall(out, "softmax_axis_r", "ops", "-DSM_WGS=" + std::to_string(w),
               "WGS=" + std::to_string(w));
    return out;
  }
  if (op == "gap")
  {
    for (int w : {64, 128, 256})
      addSmall(out, "gap_r", "ops", "-DGAP_WGS=" + std::to_string(w),
               "WGS=" + std::to_string(w));
    return out;
  }
  return out;
}

bool benchCandidate(
  ClRuntime & rt, const std::function<cl_event()> & enqueue, int iters, double * ms)
{
  try {
    *ms = rt.timeMs(enqueue, 3, iters);
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

TuningEntry autotuneOp(
  ClRuntime & rt, const OpSignature & sig, const std::vector<Candidate> & cands,
  const std::function<std::function<cl_event()>(const Candidate &)> & makeEnqueue,
  double flops, int iters)
{
  TuningEntry best;
  best.expected = expectedOps(sig, rt.info());
  best.device_id = TuningCache::deviceKey(rt.info());
  best.iters = iters;
  best.source = "tuned";

  for (const auto & c : cands) {
    std::function<cl_event()> enq;
    try {
      enq = makeEnqueue(c);
    } catch (const std::exception &) {
      continue;  // build 失败（资源/编译）→ 跳过
    }
    double ms = 0;
    if (!benchCandidate(rt, enq, iters, &ms)) continue;
    const double ops = rt.opsPerEuCycle(flops, ms);
    if (best.kernel.empty() || ops > best.ops) {
      best.kernel = c.kernel;
      best.config = c.config;
      best.options = c.options;
      best.ms = ms;
      best.ops = ops;
    }
  }
  if (!best.kernel.empty()) best.ratio = best.expected > 0 ? best.ops / best.expected : 0.0;
  return best;
}

}  // namespace gk
