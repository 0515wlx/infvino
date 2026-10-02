// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Autotuner 实现：候选枚举 + 计时。
#include "infvino/Autotuner.hpp"

#include <algorithm>
#include <cstdio>
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
    << " -DACT=" << act << " -DRES=" << res << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
  return o.str();
}
}  // namespace

std::vector<Candidate> candidatesConv3x3(const OpSignature & sig)
{
  std::vector<Candidate> out;
  // OV osv32 的 block 谱系（R23 扫描过 8x2/5x2/6x2/4x4；再加 s2 的候选）。
  std::vector<std::pair<int, int>> blocks;
  if (sig.stride == 2) {
    blocks = {{5, 4}, {4, 4}, {5, 2}, {6, 2}, {8, 2}, {3, 4}};
  } else {
    blocks = {{8, 2}, {5, 2}, {6, 2}, {4, 4}, {8, 1}, {4, 2}, {10, 2}, {8, 4}};
  }
  for (auto [obw, obh] : blocks) {
    if (sig.W > 0 && obw > sig.W) continue;   // 超过输出宽度无意义
    if (sig.H > 0 && obh > sig.H) continue;
    Candidate c;
    c.kernel = "conv3x3_ov";
    c.source = "conv_ov";
    c.options = ovOptions(obw, obh, sig.stride, sig.pad, sig.act, sig.groups == 2 ? 1 : 0);
    c.config = ovConfig(obw, obh, sig.stride, sig.pad, sig.act, sig.groups == 2 ? 1 : 0);
    out.push_back(std::move(c));
  }
  // native conv3x3_f16 作为回退候选（R18 自适应 tile 谱系），只在非 2-stride 时。
  if (sig.stride != 2) {
    for (int tx : {40, 20}) {
      if (sig.W > 0 && tx > sig.W) continue;
      for (int cb : {32, 16}) {
        Conv3x3Cfg cfg;
        cfg.TX = tx; cfg.TY = 8; cfg.TM = 1; cfg.CB = cb; cfg.CINC = 16;
        cfg.STRIDE = sig.stride; cfg.PAD = sig.pad; cfg.ACT = sig.act;
        cfg.SG = 16; cfg.WC = 1;
        Candidate c;
        c.kernel = "conv3x3_f16";
        c.source = "conv";
        c.options = cfg.options();
        c.config = cfg.label();
        out.push_back(std::move(c));
      }
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
