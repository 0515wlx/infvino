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
#include <cstdlib>
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
std::string ovOptions(int obw, int obh, int stride, int pad, int act, int res, const std::string & extra)
{
  std::ostringstream o;
  o << "-DOBW=" << obw << " -DOBH=" << obh << " -DSTRIDE=" << stride << " -DPAD=" << pad
    << " -DACT=" << act << " -DRES=" << res << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math"
    << extra;
  return o.str();
}
std::string ovConfig(int obw, int obh, int stride, int pad, int act, int res, bool fit)
{
  std::ostringstream o;
  o << "OBW=" << obw << ",OBH=" << obh << ",STRIDE=" << stride << ",PAD=" << pad
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
    if (C > 0 && C < n && n % C == 0) add("ew_binary_ch", "ops", "", "channel");
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
          (s.stride == 2) ? std::vector<std::pair<int, int>>{{5, 4}, {4, 4}, {5, 2}, {6, 2}, {8, 2}, {3, 4}}
                          : std::vector<std::pair<int, int>>{{8, 2}, {5, 2}, {6, 2}, {4, 4}, {8, 1}, {4, 2}, {10, 2}, {8, 4}};
        for (auto [obw, obh] : blocks) {
          if (s.W > 0 && obw > s.W) continue;
          if (s.H > 0 && obh > s.H) continue;
          std::string extra;
          bool fit = false;
          if (s.W > 0 && s.H > 0 && s.W % obw == 0 && s.H % obh == 0) { extra += " -DFIT_WH=1"; fit = true; }
          if (s.Cout > 0 && s.Cout % 32 == 0) extra += " -DFIT_COUT=1";
          const int res = (s.groups == 2) ? 1 : 0;
          out.push_back(mk("conv3x3_ov", "conv_ov", ovOptions(obw, obh, s.stride, s.pad, s.act, res, extra),
                           ovConfig(obw, obh, s.stride, s.pad, s.act, res, fit)));
        }
        return out;
      };
      f.ceiling = [peak](const OpSignature & s, const ClDeviceInfo & dev) {
        const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
        const int obw = (s.stride == 2) ? 5 : 8, obh = (s.stride == 2) ? 4 : 2;
        const long n = wgCount(s.W, s.H, s.Cout, obw, obh, 16);
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
        for (int obw : {2, 4, 8}) {
          if (s.W > 0 && obw > s.W) continue;
          std::ostringstream o;
          o << "-DOBW=" << obw << " -DSTRIDE=" << s.stride << " -DPAD=" << s.pad
            << " -DACT=" << s.act << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
          if (s.W > 0 && s.W % obw == 0) o << " -DFIT_WH=1";
          if (s.Cout > 0 && s.Cout % 16 == 0) o << " -DFIT_COUT=1";
          if (s.Cin > 0 && s.Cin % 16 == 0) o << " -DFIT_CIN=1";
          std::ostringstream cc;
          cc << "OBW=" << obw << ",STRIDE=" << s.stride << ",PAD=" << s.pad << ",ACT=" << s.act;
          out.push_back(mk("conv3x3_blk", "conv_blk", o.str(), cc.str()));
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
        for (int tx : {40, 20}) {
          if (s.W > 0 && tx > s.W) continue;
          for (int cb : {32, 16}) {
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
        const long n = wgCount(s.W, s.H, s.Cout, 40, 8, 32);
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
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) {
        return s.op == "gemm" || (s.op == "conv1x1" && s.N > 1);
      };
      f.candidates = [](const OpSignature & s) {
        struct Opt { int BM, BN, BK, TM, TN, DBUF; };
        const Opt opts[] = {
          {128, 64, 32, 8, 4, 0}, {128, 64, 16, 8, 4, 1}, {128, 64, 8, 8, 4, 0},
          {64, 64, 16, 8, 4, 1}, {64, 64, 8, 8, 4, 0}, {64, 32, 32, 8, 4, 0},
          {32, 64, 32, 8, 4, 0},
        };
        const int K = (s.op == "gemm") ? s.K : s.Cin;
        std::vector<Candidate> out;
        for (const auto & o : opts) {
          Tiles t;
          t.BM = o.BM; t.BN = o.BN; t.BK = o.BK; t.TM = o.TM; t.TN = o.TN; t.DBUF = o.DBUF;
          if (K < 32) t.SG = 0;
          std::string optsS = t.options();
          std::string cfgS = t.label();
          if (s.op == "conv1x1") {
            optsS += " -DEPI=1 -DACT=" + std::to_string(s.act);
            cfgS += " epi=1 act=" + std::to_string(s.act);
          }
          out.push_back(mk("gemm_f16", "gemm", optsS, cfgS));
        }
        return out;
      };
      f.ceiling = [](const OpSignature & s, const ClDeviceInfo &) {
        const int M = (s.op == "gemm") ? s.M : s.Cout;
        const int N = (s.op == "gemm") ? s.N : s.N;
        const int K = (s.op == "gemm") ? s.K : s.Cin;
        const long grid = ((M + 63) / 64) * ((N + 63) / 64);
        double e = 13.7;
        e *= std::min(1.0, static_cast<double>(grid) / 64.0);
        e *= static_cast<double>(K) / (static_cast<double>(K) + 64.0);
        return std::max(0.5, e);
      };
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
             << " -DACT=" << s.act << " -DRES=" << ((s.groups == 2) ? 1 : 0)
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

    // =========================================================================
    // conv1x1
    // =========================================================================
    {
      KernelFamily f;
      f.name = "conv1x1_gemv_f16";
      f.op = "conv1x1";
      f.source = "conv1x1";
      f.layout = {Layout::NCHW, Layout::NCHW, false, false};
      f.bottleneck = Bottleneck::Latency;
      f.supports = [](const OpSignature & s) { return s.op == "conv1x1" && s.N == 1; };
      f.candidates = [](const OpSignature & s) {
        Conv1x1Cfg cfg;
        cfg.ACT = s.act;
        cfg.RES = (s.groups == 2) ? 1 : 0;
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
      f.bottleneck = Bottleneck::Fma;
      f.supports = [](const OpSignature & s) {
        return s.op == "conv1x1" && s.N > 1 && s.Cin >= 16;
      };
      f.candidates = [](const OpSignature & s) {
        std::vector<Candidate> out;
        for (int xb : {2, 4, 8}) {
          if (s.W > 0 && xb > s.W) continue;
          for (int slm : {1, 2, 4}) {
            if (slm > 1 && s.N < 64) continue;
            std::ostringstream o;
            o << "-DX_BLOCK=" << xb << " -DSLM_DIV=" << slm << " -DACT=" << s.act
              << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
            std::ostringstream cc;
            cc << "XB" << xb << " slm" << slm << " act" << s.act;
            out.push_back(mk("conv1x1_blk", "conv1x1_blk", o.str(), cc.str()));
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
        for (int xb : {4, 8}) {
          if (s.W > 0 && xb > s.W) continue;
          std::ostringstream o;
          o << "-DX_BLOCK=" << xb << " -DDWK=" << s.K << " -DSTRIDE=" << s.stride
            << " -DPAD=" << s.pad << " -DACT=" << s.act
            << " -DSG=16 -cl-mad-enable -cl-fast-relaxed-math";
          std::ostringstream cc;
          cc << "XB" << xb << " K" << s.K << " s" << s.stride << " act" << s.act;
          out.push_back(mk("depthwise_blk", "depthwise_blk", o.str(), cc.str()));
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
    // 激活码契约（规范码，全族统一）：conv3x3 只实现 {0,1,3}；depthwise {0..4}；其余 {0..5}。
    for (auto & f : v) f.actMask = (f.op == "conv3x3") ? 0xB : (f.op == "depthwise" ? 0x1F : 0x3F);
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
    out.insert(out.end(), c.begin(), c.end());
  }
  return out;
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
