// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// KernelFamilies —— 算子族注册表的实现（见 include/infvino/KernelFamily.hpp
// 与 docs/kernel-families.md）。
//
// 这是「候选 config + 布局契约 + 物理上限」的**单一真相源**：Autotuner 的
// candidatesXxx 现在只是 `candidatesFromRegistry(sig)` 的薄包装。加一个族 = 在
// 这里加一条声明 + 一个自包含 .cl，不再改 Autotuner / dispatch / 布局规划 / expectedOps。
#include "infvino/KernelFamily.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <sstream>

namespace infvino
{

namespace
{
double gridFactor(long n_wg, int eu)
{
  const double target = std::max(1.0, static_cast<double>(eu) * 2.0);
  return std::min(1.0, static_cast<double>(n_wg) / target);
}

double convAmort(int Cin, int obw, int obh)
{
  const double work = static_cast<double>(Cin) * 9.0 * obw * obh;
  const double fixed = 24.0 + 6.0 * obw * obh;
  return work / (work + fixed);
}

long wgCount(int W, int H, int Cout, int obw, int obh, int chPerWg)
{
  return static_cast<long>((W + obw - 1) / obw) * static_cast<long>((H + obh - 1) / obh) *
         static_cast<long>((Cout + chPerWg - 1) / chPerWg);
}

std::string opt(const char * k, int v)
{
  std::ostringstream o;
  o << "-D" << k << "=" << v;
  return o.str();
}
std::string ovOptions(int obw, int obh, int stride, int pad, int act, int res, int slm,
                      const std::string & extra)
{
  std::ostringstream o;
  o << "-DOBW=" << obw << " -DOBH=" << obh << " -DSLM_DIV=" << slm << " -DSTRIDE=" << stride
    << " -DPAD=" << pad << " -DACT=" << act << " -DRES=" << res
    << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math" << extra;
  return o.str();
}
std::string ovConfig(int obw, int obh, int stride, int pad, int act, int res, int slm, bool fit)
{
  std::ostringstream o;
  o << "OBW=" << obw << ",OBH=" << obh << ",SLM=" << slm << ",STRIDE=" << stride << ",PAD=" << pad
    << ",ACT=" << act << ",RES=" << res << (fit ? " fit" : "");
  return o.str();
}
Candidate mk(const char * kernel, const char * source, const std::string & options,
             const std::string & config)
{
  Candidate c;
  c.kernel = kernel;
  c.source = source;
  c.options = options;
  c.config = config;
  return c;
}

// GEMM 族的**单一**候选配置谱（gemm_f16 / gemm_cat4_f16 共用，避免两处漂移）。
// `cat4=true` 时编译 `-DCAT4=1`：B 的逻辑 K=Cin 行按 ca/cb/cc/cd 重定向到 4 个源张量
// （见 kernels/gemm.cl 与 PlanModel::fuseConcatConv1x1），从而让 concat4→conv1x1 融合
// 复用同一 tile/流水谱。此前的 ad-hoc 路径在 PlanModel 里手工复制这份列表并追加宏，
// 属「单一真相源」缺口（R48 §4bis P0）。
std::vector<Candidate> gemmSpectrum(const OpSignature & s, bool cat4)
{
  struct Opt { int BM, BN, BK, TM, TN, DBUF; };
  const Opt opts[] = {
    {128, 64, 32, 8, 4, 0}, {128, 64, 16, 8, 4, 1}, {128, 64, 8, 8, 4, 0},
    {64, 64, 16, 8, 4, 1}, {64, 64, 8, 8, 4, 0}, {64, 32, 32, 8, 4, 0},
    {32, 64, 32, 8, 4, 0},
  };
  const int K = (s.op == "gemm") ? s.K : s.Cin;
  std::vector<Candidate> out;
  for (const auto & o : opts)
  {
    Tiles t;
    t.BM = o.BM; t.BN = o.BN; t.BK = o.BK; t.TM = o.TM; t.TN = o.TN; t.DBUF = o.DBUF;
    if (K < 32) t.SG = 0;
    std::string optsS = t.options();
    std::string cfgS = t.label();
    const std::string actS = std::to_string(s.act);
    if (cat4)
    {
      optsS += " -DEPI=1 -DACT=" + actS + " -DCAT4=1";
      cfgS += " epi=1 act=" + actS + " cat4";
      out.push_back(mk("gemm_cat4_f16", "gemm", optsS, cfgS));
    }
    else
    {
      if (s.op == "conv1x1")
      {
        optsS += " -DEPI=1 -DACT=" + actS;
        cfgS += " epi=1 act=" + actS;
      }
      out.push_back(mk("gemm_f16", "gemm", optsS, cfgS));
    }
  }
  return out;
}

// GEMM 族的软上限模型（gemm_f16 / gemm_cat4_f16 / conv1x1 N>1 共用）。
// R32 修正：grid 用实际 BM=64/BN=64；短 K 摊薄不足；上界 13.7（R14 整核）。
double gemmCeiling(const OpSignature & s)
{
  const int M = (s.op == "gemm") ? s.M : s.Cout;
  const int N = s.N;
  const int K = (s.op == "gemm") ? s.K : s.Cin;
  const long grid = ((M + 63) / 64) * ((N + 63) / 64);
  double e = 13.7;
  e *= std::min(1.0, static_cast<double>(grid) / 64.0);
  e *= static_cast<double>(K) / (static_cast<double>(K) + 64.0);
  return std::max(0.5, e);
}

// 小算子候选（launch/带宽受限，无数值语义差异）。原在 Autotuner.cpp，现为注册表真相源。
std::vector<Candidate> smallCandidates(const OpSignature & sig)
{
  std::vector<Candidate> out;
  auto add = [&](const char * kernel, const char * src, const std::string & opts,
                 const std::string & cfg) {
    out.push_back(mk(kernel, src, opts, cfg));
  };
  auto vecOpts = [](const char * macro, int v) {
    std::ostringstream o;
    o << "-D" << macro << "=" << v << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  };
  const std::string & op = sig.op;
  if (op == "ew_binary" || op == "ew_unary") {
    const bool binary = (op == "ew_binary");
    add(binary ? "ew_binary" : "ew_unary", "ops", "", "scalar");
    for (int v : {2, 4, 8})
      add(binary ? "ew_binary_v" : "ew_unary_v", "ops", vecOpts("EW_VEC", v), "VEC=" + std::to_string(v));
    return out;
  }
  if (op == "ew_binary_bcast") {
    add("ew_binary_bcast", "ops", "", "bcast");
    const int effRank = sig.params.size() > 4 ? sig.params[4] : 4;
    if (effRank <= 4) add("ew_binary_bcast4", "ops", "", "grid3");
    const int n = sig.params.size() > 0 ? sig.params[0] : 0;
    const int C = sig.params.size() > 3 ? sig.params[3] : 0;
    if (C > 0 && C < n && n % C == 0) {
      add("ew_binary_ch", "ops", "", "channel");
      // R48 D4: 通道广播（SE Mul）可 -DEWCH_OUT_FSV16 直接产出 fsv16（生产者直写）。
      out.back().canOutFsv16 = true;
    }
    return out;
  }
  if (op == "concat4") {
    add("concat4", "ops", "", "scalar");
    for (int v : {2, 4, 8}) add("concat4_v", "ops", vecOpts("EW_VEC", v), "VEC=" + std::to_string(v));
    return out;
  }
  if (op == "copy_c") { add("copy_c", "ops", "", "grid1"); add("copy_c2", "ops", "", "grid2"); return out; }
  if (op == "slice_axis") { add("slice_axis", "ops", "", "grid1"); add("slice_axis3", "ops", "", "grid3"); return out; }
  if (op == "maxpool") { add("maxpool", "ops", "", "grid1"); add("maxpool3", "ops", "", "grid3"); return out; }
  if (op == "resize_nn") { add("resize_nn", "ops", "", "grid1"); add("resize_nn3", "ops", "", "grid3"); return out; }
  if (op == "permute_0213") { add("permute_0213", "ops", "", "grid1"); add("permute_0213_3d", "ops", "", "grid3"); return out; }
  if (op == "bmm") {
    add("bmm", "ops", "", "grid1");
    add("bmm2", "ops", "", "grid3");
    for (auto [tm, tn] : {std::pair<int, int>{4, 8}, {8, 8}, {8, 4}, {4, 4}}) {
      for (int uk : {4, 8}) {
        std::ostringstream o;
        o << "-DBMM_TM=" << tm << " -DBMM_TN=" << tn << " -DBMM_UK=" << uk
          << " -cl-mad-enable -cl-fast-relaxed-math";
        add("bmm_t", "ops", o.str(),
            "TM" + std::to_string(tm) + "x" + std::to_string(tn) + "u" + std::to_string(uk));
      }
    }
    return out;
  }
  if (op == "softmax_axis") {
    add("softmax_axis", "ops", "", "serial");
    for (int w : {32, 64, 128, 256})
      add("softmax_axis_r", "ops", "-DSM_WGS=" + std::to_string(w), "WGS=" + std::to_string(w));
    return out;
  }
  if (op == "gap") {
    for (int w : {64, 128, 256})
      add("gap_r", "ops", "-DGAP_WGS=" + std::to_string(w), "WGS=" + std::to_string(w));
    return out;
  }
  return out;
}

// R48 §3.1: 确定性候选预算。仅当超出时才重排/截断（否则原样返回，保证零行为变化）。
std::vector<Candidate> evenlyPick(const std::vector<Candidate> & v, int q)
{
  if (q <= 0 || static_cast<int>(v.size()) <= q) return v;
  std::vector<Candidate> o;
  o.reserve(static_cast<size_t>(q));
  for (int i = 0; i < q; ++i)
  {
    const size_t idx = (q == 1)
                         ? 0
                         : static_cast<size_t>(
                             std::llround(static_cast<double>(i) * (v.size() - 1) / (q - 1)));
    o.push_back(v[idx]);
  }
  return o;
}

std::vector<Candidate> applyCandidateBudget(std::vector<Candidate> in, const OpSignature & sig)
{
  const char * qe = std::getenv("INFVINO_FAMILY_QUOTA");
  const char * ce = std::getenv("INFVINO_SIG_CAP");
  const int quota = qe ? std::atoi(qe) : kFamilyCandidateQuota;
  const int cap = ce ? std::atoi(ce) : kSigCandidateCap;
  const int before = static_cast<int>(in.size());
  if (in.empty() || (quota <= 0 && cap <= 0)) return in;

  // 按族（kernel 名前缀）分组，保持首次出现顺序。
  std::vector<std::string> order;
  std::map<std::string, std::vector<Candidate>> groups;
  for (auto & c : in)
  {
    if (!groups.count(c.kernel)) order.push_back(c.kernel);
    groups[c.kernel].push_back(c);
  }
  bool truncated = false;
  for (auto & kv : groups)
    if (quota > 0 && static_cast<int>(kv.second.size()) > quota)
    {
      kv.second = evenlyPick(kv.second, quota);
      truncated = true;
    }
  if (!truncated && (cap <= 0 || before <= cap)) return in;  // 未超预算，零行为变化

  // 轮转交错以保证跨族/跨瓶颈多样性，再取全局上限。
  std::vector<Candidate> out;
  out.reserve(static_cast<size_t>(before));
  for (size_t i = 0;; ++i)
  {
    bool more = false;
    for (const auto & k : order)
    {
      const auto & g = groups[k];
      if (i < g.size()) { out.push_back(g[i]); more = true; }
    }
    if (!more) break;
  }
  if (cap > 0 && static_cast<int>(out.size()) > cap)
  {
    out.resize(static_cast<size_t>(cap));
    truncated = true;
  }
  if (truncated && std::getenv("INFVINO_CAND_STATS"))
    std::fprintf(stderr, "[cand-budget] %s: %d -> %zu candidates (family quota=%d, sig cap=%d)\n",
                 sig.str().c_str(), before, out.size(), quota, cap);
  return out;
}
}  // namespace

const std::vector<KernelFamily> & kernelFamilies()
{
  static const std::vector<KernelFamily> fams = [] {
    std::vector<KernelFamily> v;
    const double peak = kPeakOpsPerEuCycle;

    // =========================================================================
    // conv3x3
    // =========================================================================
    {
      KernelFamily f;
      f.name = "conv3x3_ov";
      f.op = "conv3x3";
      f.source = "conv_ov";
      f.layout = {Layout::NCHW, Layout::NCHW, false, true};
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) { return s.op == "conv3x3" && s.Cin > 0; };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        std::vector<std::pair<int, int>> blocks =
          (s.stride == 2) ? std::vector<std::pair<int, int>>{{5, 4}, {4, 4}, {5, 2}, {6, 2}, {7, 2}, {8, 2}, {3, 4}}
                          : std::vector<std::pair<int, int>>{{8, 2}, {5, 2}, {6, 2}, {4, 4}, {8, 1}, {4, 2}, {10, 2}, {8, 4}};
        // R37/R39: conv_ov SLM_DIV splits the kd loop across SLM_DIV sub-groups in
        // one work-group (upstream SLM_DIV_FACTOR). The R37 numerical bug was a
        // work-group-geometry error in the reduction (fixed in kernels/conv_ov.cl;
        // R39), verified PASS by kernel_bench --verify and model_check. Enumerate
        // powers of two bounded by Cin (the kd split is per-channel, so a
        // non-divisor split is still correct and balanced to within one channel).
        std::vector<int> slms = {1};
        for (int d = 2; d <= 8; d *= 2)
          if (s.Cin >= d) slms.push_back(d);
        for (auto [obw, obh] : blocks) {
          if (s.W > 0 && obw > s.W) continue;
          if (s.H > 0 && obh > s.H) continue;
          std::string extra;
          bool fit = false;
          if (s.W > 0 && s.H > 0 && s.W % obw == 0 && s.H % obh == 0) { extra += " -DFIT_WH=1"; fit = true; }
          if (s.Cout > 0 && s.Cout % 32 == 0) extra += " -DFIT_COUT=1";
          const int res = (s.groups == 2) ? 1 : 0;
          for (int slm : slms)
            out.push_back(mk("conv3x3_ov", "conv_ov",
                             ovOptions(obw, obh, s.stride, s.pad, s.act, res, slm, extra),
                             ovConfig(obw, obh, s.stride, s.pad, s.act, res, slm, fit)));
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const int obw = (s.stride == 2) ? 5 : 8, obh = (s.stride == 2) ? 4 : 2;
        // R39 fix #1: an osv32 sub-group owns OSV=32 output channels (2/lane), so the
        // channel factor is ceil(Cout/32), not ceil(Cout/16). The old value
        // double-counted WGs and inflated gridFactor (inconsistent with expectedOps
        // and with the actual gws).
        // R39 fix #2: SLM_DIV packs up to SLM_DIV independent sub-groups into one
        // work-group, so the occupancy unit is the sub-group, not the work-group.
        // Now that SLM_DIV>1 is enumerated again, count sub-groups for the
        // wave-quantization factor so the ceiling stays an upper bound (without this
        // the measured ops of small grids exceed the modeled ceiling, e.g. ratio>1).
        int slmMax = 1;
        for (int d = 2; d <= 8; d *= 2) if (s.Cin >= d) slmMax = d;
        const long n = wgCount(s.W, s.H, s.Cout, obw, obh, 32) * slmMax;
        double e = peak * kConvOvMadFraction * convAmort(s.Cin, obw, obh);
        e *= std::max(0.25, gridFactor(n, eu));
        return std::max(1.0, std::min(e, kConvOvIssueCeiling));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "conv3x3_cin3";
      f.op = "conv3x3";
      f.source = "conv_cin3";
      f.layout = {Layout::NCHW, Layout::NCHW, false, true};
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) {
        return s.op == "conv3x3" && s.Cin > 0 && s.Cin <= 4 && s.Cout > 0 && s.Cout % 2 == 0;
      };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        std::ostringstream o;
        o << "-DCIN=" << s.Cin << " -DCOUT=" << s.Cout << " -DSTRIDE=" << s.stride
          << " -DPAD=" << s.pad << " -DACT=" << s.act
          << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
        out.push_back(mk("conv3x3_cin3", "conv_cin3", o.str(),
                         "CIN" + std::to_string(s.Cin) + " COUT" + std::to_string(s.Cout)));
        return out;
      };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo &) {
        return std::max(1.0, std::min(kConvOvIssueCeiling * convAmort(s.Cin, 8, 2), 12.0));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "conv3x3_blk";
      f.op = "conv3x3";
      f.source = "conv_blk";
      f.layout = {Layout::FSV16, Layout::NCHW, true, true};
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) { return s.op == "conv3x3" && s.Cin > 0; };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        // R37: SLM_DIV splits the input-channel loop across that many sub-groups in
        // one work-group (upstream SLM_DIV_FACTOR), hiding the global-read latency
        // that dominates the occupancy-limited shapes. Enumerate exact divisors of
        // the input-channel block count (bounded, balanced partition).
        const int icb = (s.Cin > 0) ? ((s.Cin + 15) / 16) : 1;
        std::vector<int> slms = {1};
        for (int d = 2; d <= icb && d <= 8; ++d)
          if (icb % d == 0) slms.push_back(d);
        for (int obw : {2, 4, 8}) {
          if (s.W > 0 && obw > s.W) continue;
          for (int slm : slms) {
            std::ostringstream o;
            o << "-DOBW=" << obw << " -DSLM_DIV=" << slm << " -DSTRIDE=" << s.stride
              << " -DPAD=" << s.pad << " -DACT=" << s.act
              << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
            if (s.W > 0 && s.W % obw == 0) o << " -DFIT_WH=1";
            if (s.Cout > 0 && s.Cout % 16 == 0) o << " -DFIT_COUT=1";
            if (s.Cin > 0 && s.Cin % 16 == 0) o << " -DFIT_CIN=1";
            std::ostringstream cc;
            cc << "OBW=" << obw << ",SLM=" << slm << ",STRIDE=" << s.stride
               << ",PAD=" << s.pad << ",ACT=" << s.act;
            out.push_back(mk("conv3x3_blk", "conv_blk", o.str(), cc.str()));
          }
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = wgCount(s.W, s.H, s.Cout, 8, 1, 16);
        double e = peak * 0.529 * convAmort(s.Cin, 8, 1);
        e *= std::max(0.25, gridFactor(n, eu));
        return std::max(1.0, std::min(e, 16.9));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "conv3x3_f16";
      f.op = "conv3x3";
      f.source = "conv";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) { return s.op == "conv3x3" && s.Cin > 0; };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        std::vector<int> cincs = {16};
        if (s.Cin > 0) {
          if (s.Cin % 16 != 0) cincs.push_back(8);
          if (s.Cin <= 16) cincs.push_back(s.Cin);
          if (s.Cin <= 8 && s.Cin != 4) cincs.push_back(4);
          std::sort(cincs.begin(), cincs.end());
          cincs.erase(std::unique(cincs.begin(), cincs.end()), cincs.end());
        }
        // R40: CB=8 for very narrow outputs. With Cout=8 a CB=16 work-item leaves
        // half its accumulator lanes addressing out-of-range channels; CB=8 measured
        // +21% (5.03 vs 4.16 ops) on 160x160 16->8. Only offered when Cout<=8 so the
        // common Cout%16==0 shapes keep their existing candidate set.
        std::vector<int> cbs = {32, 16};
        if (s.Cout > 0 && s.Cout <= 8) cbs.insert(cbs.begin(), 8);
        for (int tx : {40, 20}) {
          if (s.W > 0 && tx > s.W) continue;
          for (int cb : cbs) {
            for (int cinc : cincs) {
              Conv3x3Cfg cfg;
              cfg.TX = tx; cfg.TY = 8; cfg.TM = 1; cfg.CB = cb; cfg.CINC = cinc;
              cfg.STRIDE = s.stride; cfg.PAD = s.pad; cfg.ACT = s.act; cfg.SG = 16; cfg.WC = 1;
              std::string opts = cfg.options();
              if (s.W > 0 && s.H > 0 && s.W % tx == 0 && s.H % 8 == 0) opts += " -DFIT_WH=1";
              if (s.Cin > 0 && s.Cin % cinc == 0) opts += " -DFIT_CIN=1";
              if (s.Cout > 0 && s.Cout % cb == 0) opts += " -DFIT_CB=1";
              out.push_back(mk("conv3x3_f16", "conv", opts, cfg.label()));
            }
          }
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        // R39 fix: the native conv3x3 work-group is (TX/TM, TY) = (40, 8) = 320
        // work-items = 20 sub-groups, so the wave-quantization unit is the
        // sub-group, not the work-group. Counting WGs made the ceiling ~20x too
        // low on small grids (measured > ceiling, e.g. W80 Cin32 Cout16 ratio 1.59).
        const int subPerWg = (40 / 1) * 8 / 16;
        const long n = wgCount(s.W, s.H, s.Cout, 40, 8, 32) * subPerWg;
        double e = kConvStagingFreeCeiling * convAmort(s.Cin, 40, 8);
        e *= std::max(0.25, gridFactor(n, eu));
        return std::max(1.0, std::min(e, kConvStagingFreeCeiling));
      };
      v.push_back(std::move(f));
    }

    // =========================================================================
    // GEMM（同时服务 conv1x1 的 N>1 计算）
    // =========================================================================
    {
      KernelFamily f;
      f.name = "gemm_f16";
      f.op = "gemm";
      f.source = "gemm";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.layout.inIndex = 1;   // gemm 槽 0 = 权重，激活在槽 1（R48 D4）
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) {
        return s.op == "gemm" || (s.op == "conv1x1" && s.N > 1);
      };
      f.candidates = [](const OpSignature & s) { return gemmSpectrum(s, false); };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo &) { return gemmCeiling(s); };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "gemm_sk_f16";
      f.op = "conv1x1";
      f.source = "gemm_sk";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) { return s.op == "conv1x1" && s.N > 1; };
      f.candidates = [](const OpSignature & s) {
        struct SkOpt { int TM, TN, UK; };
        const SkOpt skopts[] = {{8, 4, 4}, {4, 4, 4}, {16, 4, 4}, {8, 8, 4}, {8, 4, 8}};
        std::vector<Candidate> out;
        for (const auto & o : skopts) {
          std::ostringstream os;
          os << "-DSK_TM=" << o.TM << " -DSK_TN=" << o.TN << " -DSK_SG=16 -DSK_UK=" << o.UK
             << " -DACT=" << s.act << " -DRES=" << ((s.groups & 2) ? 1 : 0)
             << " -cl-mad-enable -cl-fast-relaxed-math";
          std::ostringstream cc;
          cc << "sk TM" << o.TM << " TN" << o.TN << " u" << o.UK << " act" << s.act;
          out.push_back(mk("gemm_sk_f16", "gemm_sk", os.str(), cc.str()));
        }
        return out;
      };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo &) {
        const long grid = ((s.Cout + 7) / 8) * ((s.N + 3) / 4);
        double e = 13.7 * std::min(1.0, static_cast<double>(grid) / 64.0);
        return std::max(0.5, e);
      };
      v.push_back(std::move(f));
    }
    {
      // R48 §4bis P0: concat4→conv1x1 融合（R30c Route A）。此前完全在注册表之外：
      // `--candidates` 显示 `{}`，Autotuner/PlanModel 里手工复制 gemm 谱 + 追加 -DCAT4。
      // 现在作为独立族声明：布局契约（inIndex=1，激活在槽 1..4）、上限模型、候选谱
      // 全部走单一真相源，可被 mincut/报告/审计看见。
      //
      // 数据通路：A=权重[Cout][Cin]；C=输出[Cout][HW]；逻辑 B[Cin][HW] 由 4 个连续源
      // (b0..b3, 通道数 ca/cb/cc/cd) 拼接而成，由 CA/CB/CC/O0..O3 在 kernel 内重定向。
      // 当前只有 NCHW 变体（canOutFsv16=false）；blocked 变体是后续「扩充算子族」的项。
      KernelFamily f;
      f.name = "gemm_cat4_f16";
      f.op = "conv1x1_cat4";
      f.source = "gemm";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.layout.inIndex = 1;   // 槽 0 = 权重，激活源在槽 1..4（与 gemm_f16 同约定）
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) { return s.op == "conv1x1_cat4"; };
      f.candidates = [](const OpSignature & s) { return gemmSpectrum(s, true); };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo &) { return gemmCeiling(s); };
      v.push_back(std::move(f));
    }

    // =========================================================================
    // conv1x1
    // =========================================================================
    {
      KernelFamily f;
      f.name = "conv1x1_gemv_f16";
      f.op = "conv1x1";
      f.source = "conv1x1";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.layout.inIndex = 1;   // R48 D4: conv1x1 槽 0 = 权重，激活在槽 1
      f.bottleneck = Bottleneck::Latency;
      f.supports = [](const OpSignature & s) { return s.op == "conv1x1" && s.N == 1; };
      f.candidates = [](const OpSignature & s) {
        Conv1x1Cfg cfg;
        cfg.ACT = s.act;
        cfg.RES = (s.groups & 2) ? 1 : 0;
        cfg.SG = 16;
        return std::vector<Candidate>{mk("conv1x1_gemv_f16", "conv1x1", cfg.options(), cfg.label())};
      };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        return std::max(1.0, 8.0 * gridFactor(s.Cout, eu));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "conv1x1_blk";
      f.op = "conv1x1";
      f.source = "conv1x1_blk";
      f.layout = {Layout::FSV16, Layout::NCHW, true, true};
      f.layout.inIndex = 1;   // R48 D4: 槽 0 = os_is_yx_isv16_osv16 权重，激活在槽 1
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) {
        return s.op == "conv1x1" && s.N > 1 && s.Cin >= 16;
      };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        // R48 D1: Y_BLOCK = 每 WI 的输出行数（空间 tiling）。权重跨行复用、摊薄权重读；
        // 与 X_BLOCK 正交。实测：大 H（>=40）上 XB8+YB2 比 XB4+YB1 快 13–24%，小 H 上
        // 因网格饥饿变慢——正是「按场景分族」要的跨瓶颈候选。约束 xb*yb<=16（累加器
        // 寄存器预算）且 yb>1 时不再叠 split-K（两个 latency 旋钮同时上收益低、候选翻倍）。
        auto emit = [&](int xb, int yb, int slm) {
          std::ostringstream o;
          o << "-DX_BLOCK=" << xb << " -DY_BLOCK=" << yb << " -DSLM_DIV=" << slm
            << " -DACT=" << s.act << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
          std::ostringstream cc;
          cc << "XB" << xb << " YB" << yb << " slm" << slm << " act" << s.act;
          out.push_back(mk("conv1x1_blk", "conv1x1_blk", o.str(), cc.str()));
        };
        for (int xb : {2, 4, 8}) {
          if (s.W > 0 && xb > s.W) continue;
          for (int slm : {1, 2, 4}) {
            if (slm > 1 && s.N < 64) continue;
            emit(xb, 1, slm);
          }
        }
        for (int yb : {2, 4}) {
          if (s.H > 0 && yb > s.H) continue;
          for (int xb : {2, 4, 8}) {
            if (s.W > 0 && xb > s.W) continue;
            if (xb * yb > 16) continue;
            emit(xb, yb, 1);
          }
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = static_cast<long>((s.N + 3) / 4) * static_cast<long>((s.Cout + 15) / 16);
        // 离线 ISA（host ocloc，XB=8）：主循环 128 mad / 356 指令 = 36.0% → 配额 11.5。
        double e = peak * 0.360 * (static_cast<double>(s.Cin) / (s.Cin + 32.0));
        e *= std::max(0.25, gridFactor(n, eu));
        return std::max(0.5, std::min(e, 11.5));
      };
      v.push_back(std::move(f));
    }

    // =========================================================================
    // depthwise
    // =========================================================================
    {
      KernelFamily f;
      f.name = "depthwise_f16";
      f.op = "depthwise";
      f.source = "conv_general";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Instruction;
      f.supports = [](const OpSignature & s) { return s.op == "depthwise" && (s.K == 3 || s.K == 5); };
      f.candidates = [](const OpSignature & s) {
        std::ostringstream o;
        o << "-DDW_K=" << s.K << " -DDW_S=" << s.stride << " -DDW_P=" << s.pad
          << " -DDW_ACT=" << s.act << " -cl-mad-enable -cl-fast-relaxed-math";
        std::ostringstream cc;
        cc << "K=" << s.K << ",S=" << s.stride << ",P=" << s.pad << ",ACT=" << s.act;
        return std::vector<Candidate>{mk("depthwise_f16", "conv_general", o.str(), cc.str())};
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = static_cast<long>(s.Cout) * s.W * s.H;
        return std::max(0.3, peak * 0.086 * gridFactor(n, eu));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "depthwise_v";
      f.op = "depthwise";
      f.source = "conv_general";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Instruction;
      f.supports = [](const OpSignature & s) { return s.op == "depthwise" && (s.K == 3 || s.K == 5); };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        for (int tw : {2, 4, 8}) {
          if (s.W > 0 && tw > s.W) continue;
          std::ostringstream o;
          o << "-DDW_K=" << s.K << " -DDW_S=" << s.stride << " -DDW_P=" << s.pad
            << " -DDW_ACT=" << s.act << " -DDW_TW=" << tw
            << " -cl-mad-enable -cl-fast-relaxed-math";
          std::ostringstream cc;
          cc << "K=" << s.K << ",S=" << s.stride << ",P=" << s.pad << ",ACT=" << s.act << ",TW=" << tw;
          out.push_back(mk("depthwise_v", "conv_general", o.str(), cc.str()));
        }
        return out;
      };
      // R30 ISA：depthwise_v 1485 指令 / 128 mad = 8.6% → 配额 32×0.086 ≈ 2.8。
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = static_cast<long>(s.Cout) * s.W * s.H;
        return std::max(0.3, peak * 0.086 * gridFactor(n, eu));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "depthwise_vp";
      f.op = "depthwise";
      f.source = "conv_general";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Instruction;
      // R31 负结果：默认不进候选，只有 INFVINO_DW_PAD 打开时才参与。
      f.supports = [](const OpSignature & s) {
        return std::getenv("INFVINO_DW_PAD") && s.op == "depthwise" && (s.K == 3 || s.K == 5);
      };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        for (int tw : {2, 4, 8}) {
          if (s.W > 0 && tw > s.W) continue;
          std::ostringstream o;
          o << "-DDW_K=" << s.K << " -DDW_S=" << s.stride << " -DDW_P=" << s.pad
            << " -DDW_ACT=" << s.act << " -DDW_TW=" << tw
            << " -cl-mad-enable -cl-fast-relaxed-math";
          std::ostringstream cc;
          cc << "K=" << s.K << ",S=" << s.stride << ",P=" << s.pad << ",ACT=" << s.act
             << ",TW=" << tw << ",PAD";
          out.push_back(mk("depthwise_vp", "conv_general", o.str(), cc.str()));
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = static_cast<long>(s.Cout) * s.W * s.H;
        return std::max(0.3, peak * 0.12 * gridFactor(n, eu));
      };
      v.push_back(std::move(f));
    }
    {
      KernelFamily f;
      f.name = "depthwise_blk";
      f.op = "depthwise";
      f.source = "depthwise_blk";
      f.layout = {Layout::FSV16, Layout::NCHW, true, true};
      f.bottleneck = Bottleneck::Instruction;
      f.supports = [](const OpSignature & s) {
        return s.op == "depthwise" && (s.K == 3 || s.K == 5);
      };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        // R50: Y_BLOCK = 每 WI 的输出行数（输入行滑动窗口复用）。K 个输入行服务 1 个输出行 →
        // YB=T 时 (T-1)*S+K 个输入行服务 T 个输出行，输入 load/地址计算摊薄 ~K/T 倍。逐位一致
        // （每个输出的 kh 累加顺序不变）。约束 XB*YB<=16（累加器/line 寄存器预算）。
        for (int yb : {1, 2, 4}) {
          if (s.H > 0 && yb > s.H) continue;
          for (int xb : {4, 8}) {
            if (s.W > 0 && xb > s.W) continue;
            if (xb * yb > 16) continue;
            std::ostringstream o;
            o << "-DX_BLOCK=" << xb << " -DY_BLOCK=" << yb << " -DDWK=" << s.K
              << " -DSTRIDE=" << s.stride << " -DPAD=" << s.pad << " -DACT=" << s.act
              << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
            std::ostringstream cc;
            cc << "XB" << xb << " YB" << yb << " K" << s.K << " s" << s.stride << " act" << s.act;
            out.push_back(mk("depthwise_blk", "depthwise_blk", o.str(), cc.str()));
          }
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const long n = static_cast<long>((s.Cout + 15) / 16) *
                       static_cast<long>((s.W + 7) / 8) * static_cast<long>(s.H);
        // 离线 ISA（host ocloc）：K=5 主循环 82 mad / 652 指令 = 12.6%；K=3 全展开（整核 ~7%）。
        return std::max(0.3, peak * 0.11 * gridFactor(n, eu));
      };
      v.push_back(std::move(f));
    }
    // 小算子族（无布局/权重重排；launch/带宽受限）。候选逻辑见 smallCandidates。
    for (const char * op : {"ew_binary", "ew_unary", "ew_binary_bcast", "concat4", "copy_c",
                            "slice_axis", "maxpool", "resize_nn", "permute_0213", "bmm",
                            "softmax_axis", "gap"}) {
      KernelFamily f;
      f.name = std::string("small_") + op;
      f.op = op;
      f.source = "ops";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Memory;
      std::string o = op;
      f.supports = [o](const OpSignature & s) { return s.op == o; };
      f.candidates = smallCandidates;
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo & d) { return expectedOps(s, d); };
      v.push_back(std::move(f));
    }
    {
      // R51 D4+ / R52：`gap` 的 fsv16 输入布局契约（**无候选**，仅让布局规划器把它当作
      // 「可读 fsv16 的消费者」）。实际 kernel 由 run() 在输入 fsv16 时加
      // `-DGAP_IN_FSV16=1` 编译。这是「SE value 生产者直写 fsv16」联动的关键：
      // SE 的 value 同时被 gap 与 conv1x1 消费，gap 若只吃 NCHW 就会把 value 钉死 NCHW。
      // R52：`supports` 必须是**真实的 op 判据**（而非恒 false）——`opCanReadFsv16`
      // （分配补齐集合 mayBeFsv16）按 supports 过滤；恒 false 会让补齐集合漏掉 gap 的
      // 输入，而 planBlockedLayout 仍按 nodeFamily 把它标成 fsv16 → 缓冲越界（R51 §5.1）。
      KernelFamily f;
      f.name = "gap_fsv16";
      f.op = "gap";
      f.source = "ops";
      f.layout = {Layout::FSV16, Layout::NCHW, false, false};
      f.layout.inIndex = 0;
      f.bottleneck = Bottleneck::Memory;
      f.supports = [](const OpSignature & s) { return s.op == "gap"; };   // 布局契约族（无候选）
      v.push_back(std::move(f));
    }
    // 激活码契约（规范码，全族统一）：conv3x3 只实现 {0,1,3}；depthwise {0..4}；其余 {0..5}。
    for (auto & f : v) f.actMask = (f.op == "conv3x3") ? 0xB : (f.op == "depthwise" ? 0x1F : 0x3F);
    // R43: 硬上限（只放 ISA 指令发射配额 / roofline 下界，不含 amort/gridFactor/延迟等
    // 经验 derate）。集中在这里声明，避免散落；未命中者回退为软 `ceiling`。
    for (auto & f : v) {
      if (f.hardCeiling) continue;
      const std::string & n = f.name;
      if (n == "conv3x3_ov")         f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return kConvOvIssueCeiling; };
      else if (n == "conv3x3_blk")   f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 16.9; };
      else if (n == "conv3x3_f16")   f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return kConvStagingFreeCeiling; };
      else if (n == "conv3x3_cin3")  f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 12.0; };
      else if (n == "gemm_f16" || n == "gemm_sk_f16" || n == "gemm_cat4_f16")
        f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 13.7; };
      else if (n == "conv1x1_blk")   f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 11.5; };
      else if (n == "conv1x1_gemv_f16") f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 8.0; };
      else if (n == "depthwise_f16" || n == "depthwise_v") f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 32.0 * 0.086; };
      else if (n == "depthwise_vp")  f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 32.0 * 0.12; };
      else if (n == "depthwise_blk") f.hardCeiling = [](const OpSignature &, const ClDeviceInfo &) { return 32.0 * 0.11; };
      else f.hardCeiling = f.ceiling;   // 小算子 = roofline，本身就是物理下界
    }
    return v;
  }();
  return fams;
}

std::vector<Candidate> candidatesFromRegistry(const OpSignature & sig)
{
  std::vector<Candidate> out;
  for (const auto & f : kernelFamilies())
  {
    if (!f.candidates) continue;
    // 不按 f.op 过滤：一个族可服务多个 op（如 gemm_f16 也服务 conv1x1 N>1），
    // 由 supports(sig) 负责精确判定。
    if (!(f.actMask & (1 << sig.act))) continue;
    if (f.supports && !f.supports(sig)) continue;
    auto c = f.candidates(sig);
    // R49: 族的布局契约 → 候选的输出能力（生产者直写 fsv16）。候选可自行覆盖。
    for (auto & cand : c)
      if (!cand.canOutFsv16) cand.canOutFsv16 = f.layout.canOutFsv16;
    out.insert(out.end(), c.begin(), c.end());
  }
  if (std::getenv("INFVINO_CAND_STATS"))
    std::fprintf(stderr, "[cand-stats] %s: %zu candidates\n", sig.str().c_str(), out.size());
  return applyCandidateBudget(std::move(out), sig);
}

const KernelFamily * familyByName(const std::string & kernelName)
{
  for (const auto & f : kernelFamilies())
    if (f.name == kernelName) return &f;
  return nullptr;
}

double bestFamilyCeiling(const OpSignature & sig, const ClDeviceInfo & dev)
{
  double best = 0.0;
  for (const auto & f : kernelFamilies())
  {
    if (!f.ceiling) continue;
    if (f.supports && !f.supports(sig)) continue;
    best = std::max(best, f.ceiling(sig, dev));
  }
  return best;
}

}  // namespace infvino
