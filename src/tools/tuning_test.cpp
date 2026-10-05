// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// tuning_test —— 自动调优基础设施的离线自检（**不需要 GPU**）。
//
// 覆盖：签名稳定性、缓存 round-trip、设备键、expected_ops 的单调性/上界。
// 用法：./tuning_test            （成功返回 0）
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "infvino/Autotuner.hpp"
#include "infvino/KernelFamily.hpp"
#include "infvino/LayoutSolver.hpp"
#include "infvino/Tuning.hpp"

namespace
{
int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
                              else std::printf("ok:   %s\n", msg); } while (0)

infvino::ClDeviceInfo dev()
{
  infvino::ClDeviceInfo d;
  d.name = "TestGT"; d.eu = 80; d.clock_mhz = 1300; d.subgroup_size = 16;
  d.pci_vendor_id = 0x8086; d.pci_device_id = 0x9a49;
  return d;
}
}  // namespace

int main()
{
  using namespace infvino;

  // --- 签名稳定性/可区分性 ---
  const auto a = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
  const auto b = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
  const auto c = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 0);
  CHECK(a.str() == b.str(), "same params -> same signature");
  CHECK(a.str() != c.str(), "different act -> different signature");
  CHECK(a.str().find("W40H40") != std::string::npos, "signature encodes spatial");
  CHECK(OpSignature::gemm(128, 1600, 192, 0).str().find("M128N1600K192") != std::string::npos,
        "gemm signature encodes M/N/K");

  // --- 缓存 round-trip ---
  std::filesystem::create_directories("/tmp/opencode");
  const std::string path = "/tmp/opencode/tuning_test_cache.json";
  TuningCache out;
  out.setDeviceId(TuningCache::deviceKey(dev()));
  TuningEntry e;
  e.kernel = "conv3x3_ov"; e.config = "OBW=8,OBH=1"; e.options = "-DOBW=8 -DOBH=1";
  e.ms = 0.123; e.ops = 8.4; e.expected = 12.0; e.ratio = 0.70; e.iters = 12;
  e.device_id = out.deviceId(); e.source = "tuned";
  e.exact = false; e.tol = 0.05;   // R48: 数值契约标记 round-trip
  out.put(a, e);
  CHECK(out.save(path), "cache save");
  TuningCache in = TuningCache::load(path);
  CHECK(in.size() == 1, "cache load count");
  const TuningEntry * le = in.lookup(a);
  CHECK(le != nullptr, "cache lookup hit");
  if (le) {
    CHECK(le->kernel == "conv3x3_ov", "cache kernel round-trip");
    CHECK(le->config == "OBW=8,OBH=1", "cache config round-trip");
    CHECK(le->options == "-DOBW=8 -DOBH=1", "cache options round-trip");
    CHECK(le->iters == 12, "cache iters round-trip");
    CHECK(le->ratio > 0.69 && le->ratio < 0.71, "cache ratio round-trip");
    CHECK(le->exact == false, "R48: numeric contract (exact=false) round-trip");
    CHECK(le->tol > 0.049 && le->tol < 0.051, "R48: numeric tol round-trip");
  }
  in.setDeviceId("different_device_eu16_clk300");
  CHECK(in.lookup(a) == nullptr, "device key mismatch -> miss（回退启发式）");

  // --- Round 28: 通用小算子签名 + 候选枚举 ---
  ClDeviceInfo d = dev();
  const auto ew1 = OpSignature::custom("ew_binary", {409600, 0, 0});
  const auto ew2 = OpSignature::custom("ew_binary", {409600, 0, 0});
  const auto ew3 = OpSignature::custom("ew_binary", {204800, 0, 0});
  CHECK(ew1.str() == ew2.str(), "custom signature stable");
  CHECK(ew1.str() != ew3.str(), "custom signature encodes params");
  CHECK(ew1.str().find("ew_binary|409600,0,0") != std::string::npos,
        "custom signature string format");
  const auto cands = candidatesSmall(ew1);
  CHECK(cands.size() >= 4, "ew_binary has scalar + vector variants");
  bool hasVec = false;
  for (const auto & c : cands) if (c.kernel.find("_v") != std::string::npos) hasVec = true;
  CHECK(hasVec, "ew_binary candidate set includes a vectorized variant");
  const auto copyc = candidatesSmall(OpSignature::custom("copy_c", {25600, 16}));
  bool has3d = false;
  for (const auto & c : copyc) if (c.kernel == "copy_c2") has3d = true;
  CHECK(has3d, "copy_c candidate set includes the 2-D grid variant");
  // --- R33: native conv3x3 的 CINC 候选（Cin%16!=0 时才有 CINC=8）---
  {
    const auto sig8 = OpSignature::conv3x3(160, 160, 1, 1, 8, 16, 1);
    bool hasCinc8 = false, hasCinc16 = false;
    for (const auto & c : candidatesConv3x3(sig8))
      if (c.kernel == "conv3x3_f16") {
        if (c.config.find("CINC8") != std::string::npos) hasCinc8 = true;
        if (c.config.find("CINC16") != std::string::npos) hasCinc16 = true;
      }
    CHECK(hasCinc8, "R33: Cin%16!=0 -> native candidate set includes CINC8");
    CHECK(hasCinc16, "R33: native candidate set keeps CINC16");
    const auto sig16 = OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1);
    bool cinc8OnAligned = false;
    for (const auto & c : candidatesConv3x3(sig16))
      if (c.kernel == "conv3x3_f16" && c.config.find("CINC8") != std::string::npos)
        cinc8OnAligned = true;
    CHECK(!cinc8OnAligned, "R33: Cin%16==0 -> no redundant CINC8 candidate");
    // R34: 小 Cin（stem）枚举精确 CINC 与 CINC=4
    const auto stem = OpSignature::conv3x3(320, 320, 2, 1, 3, 16, 1);
    bool hasCinc3 = false, hasCinc4 = false;
    for (const auto & c : candidatesConv3x3(stem))
      if (c.kernel == "conv3x3_f16") {
        if (c.config.find("CINC3 ") != std::string::npos) hasCinc3 = true;
        if (c.config.find("CINC4 ") != std::string::npos) hasCinc4 = true;
      }
    CHECK(hasCinc3, "R34: stem Cin=3 -> native candidate set includes exact CINC3");
    CHECK(hasCinc4, "R34: stem Cin=3 -> native candidate set includes CINC4 fallback");
  }
  CHECK(expectedOps(ew1, d) > 0.0, "small-op expected_ops is positive");

  // --- Round 30: 剩余 kernel 的内存 roofline 中间标准 ---
  const auto cp1 = OpSignature::custom("copy_c", {25600, 16});
  const auto cp2 = OpSignature::custom("copy_c", {400, 128});
  CHECK(expectedOps(cp1, d) > expectedOps(cp2, d), "R30: bigger copy -> higher expected");
  const double eb = expectedOps(OpSignature::custom("ew_binary_bcast", {285600, 2, 0, 0}), d);
  CHECK(eb > 0.05 && eb < 1.0, "R30: bcast expected in memory-roofline range");
  const double bm = expectedOps(OpSignature::custom("bmm", {1, 2, 64, 400, 400}), d);
  CHECK(bm > 0.0 && bm < 0.1, "R30: bmm expected is compute/grid bounded");
  const double dw = expectedOps(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 1), d);
  CHECK(dw > 0.5 && dw < 4.0, "R30: depthwise expected = ISA instruction quota (~2.8)");
  CHECK(expectedOps(OpSignature::gap(120, 196), d) > 0.0, "R30: gap expected positive");
  CHECK(expectedOps(OpSignature::custom("softmax_axis", {1, 16, 33600}), d) > 0.0,
        "R30: softmax expected positive");

  // --- expected_ops：单调 + 上界 ---
  const double big = expectedOps(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1), d);
  const double mid = expectedOps(OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1), d);
  const double small = expectedOps(OpSignature::conv3x3(20, 20, 1, 1, 64, 64, 1), d);
  CHECK(big > small, "larger grid -> higher expected (grid factor)");
  // R24: the middle standard now targets the issue-mix ceiling (~20.3), not the old
  // ~16 "1 broadcast : 1 mad" reading (the broadcast is folded into the mad).
  CHECK(mid <= kConvOvIssueCeiling + 1e-9, "conv expected <= OV issue ceiling");
  CHECK(big <= kConvOvIssueCeiling + 1e-9, "conv expected <= OV issue ceiling");
  CHECK(big > kConvStagingFreeCeiling, "R24: middle standard no longer anchored ~16");
  const double g = expectedOps(OpSignature::gemm(1024, 1024, 1024, 0), d);
  CHECK(g > 5.0 && g <= 17.7 + 1e-9, "gemm expected within (5, compute-only 17.7]");
  CHECK(expectedOps(c, d) >= 1.0, "expected has a positive floor");

  TuningCache disabled;
  disabled.setEnabled(false);
  CHECK(disabled.lookup(a) == nullptr, "disabled cache always misses");

  // --- R48 M0: 候选预算（确定性截断 + 环境覆盖）---
  {
    const auto bigSig = OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1);
    const auto full = candidatesConv3x3(bigSig);
    CHECK(full.size() > 4, "R48: uncapped candidate set is non-trivial");
    setenv("INFVINO_SIG_CAP", "4", 1);
    const auto capd = candidatesConv3x3(bigSig);
    CHECK(capd.size() == 4, "R48: per-signature cap truncates candidate set");
    std::set<std::string> fams;
    for (const auto & c : capd) fams.insert(c.kernel);
    CHECK(fams.size() >= 2, "R48: budget preserves cross-family diversity (round-robin)");
    unsetenv("INFVINO_SIG_CAP");
    setenv("INFVINO_FAMILY_QUOTA", "2", 1);
    const auto q = candidatesConv3x3(bigSig);
    std::map<std::string, int> qc;
    for (const auto & c : q) qc[c.kernel]++;
    bool qok = true;
    for (const auto & kv : qc) if (kv.second > 2) qok = false;
    CHECK(qok, "R48: per-family quota enforced");
    CHECK(q.size() < full.size(), "R48: per-family quota shrinks the set");
    unsetenv("INFVINO_FAMILY_QUOTA");
  }

  // --- R52: 布局契约族的「可读 fsv16」必须能被分配补齐集合发现 ---
  // 根因（R51 §5.1）：`gap_fsv16` 曾把 supports 设为恒 false，导致分配补齐集合
  // （mayBeFsv16 → opCanReadFsv16，按 supports 过滤）漏掉 gap 的输入，而布局标记
  // 仍按 nodeFamily 把它标成 fsv16 → OUT_FSV16 越界写，整网数值错乱。
  // 契约族可以**无候选**（candidates==nullptr），但其 supports 必须是真实的 op 判据。
  {
    const OpSignature gs = OpSignature::gap(120, 196);
    const KernelFamily * gf = familyByName("gap_fsv16");
    CHECK(gf != nullptr, "R52: gap_fsv16 layout-contract family is registered");
    if (gf)
    {
      CHECK(gf->layout.in == Layout::FSV16 && gf->layout.inIndex == 0,
            "R52: gap_fsv16 declares fsv16 input at the activation slot");
      CHECK(!gf->layout.canOutFsv16, "R52: gap_fsv16 output stays NCHW");
      CHECK(gf->candidates == nullptr, "R52: layout-contract family produces no candidates");
      CHECK(gf->supports && gf->supports(gs),
            "R52: gap_fsv16 supports() matches the gap signature (padding gate can see it)");
    }
    // 复刻规划器的 opCanReadFsv16 判据：gap 签名必须能找到一个声明「读 FSV16」的族。
    bool discoverable = false;
    for (const auto & f : kernelFamilies())
    {
      if (f.layout.in != Layout::FSV16 || f.layout.inIndex != 0) continue;
      if (!(f.actMask & (1u << gs.act))) continue;
      if (f.supports && !f.supports(gs)) continue;
      discoverable = true;
      break;
    }
    CHECK(discoverable, "R52: a gap signature is discoverable as an fsv16 reader (alloc/mark agree)");
    // 契约族不得污染候选集：gap 的候选里不能出现 gap_fsv16。
    bool pollutes = false;
    for (const auto & c : candidatesFromRegistry(gs))
      if (c.kernel == "gap_fsv16") pollutes = true;
    CHECK(!pollutes, "R52: contract family does not leak into the candidate set");
  }

  // --- R49: 布局标注的精确最小割求解器（穷举对照）---
  {
    // 一般二元代价表：把 f(x,y) 拆成 unary+吸引项后，解必须是**精确**最优。
    struct OrigPair { int i, j; double f[2][2]; };
    auto evalOrig = [](int n, const std::vector<std::array<double, 2>> & u,
                       const std::vector<OrigPair> & pt, const std::vector<int> & fx,
                       const std::vector<int> & x) {
      double E = 0.0;
      for (int i = 0; i < n; ++i) E += u[(size_t)i][(size_t)x[(size_t)i]];
      for (const auto & p : pt) E += p.f[x[(size_t)p.i]][x[(size_t)p.j]];
      (void)fx;
      return E;
    };
    auto brute = [&](int n, const std::vector<std::array<double, 2>> & u,
                     const std::vector<OrigPair> & pt, const std::vector<int> & fx) {
      double best = std::numeric_limits<double>::infinity();
      std::vector<int> bestX((size_t)n, 0);
      for (int mask = 0; mask < (1 << n); ++mask) {
        std::vector<int> x((size_t)n, 0);
        bool ok = true;
        for (int i = 0; i < n; ++i) {
          x[(size_t)i] = (mask >> i) & 1;
          if (fx[(size_t)i] >= 0 && x[(size_t)i] != fx[(size_t)i]) ok = false;
        }
        if (!ok) continue;
        const double e = evalOrig(n, u, pt, fx, x);
        if (e < best) { best = e; bestX = x; }
      }
      return std::make_pair(best, bestX);
    };

    // 链 t0 -n0- t1 -n1- t2，节点代价表来自「布局→kernel」模型（含输出 FSV16 约束）。
    // f00=min(B+r,G), f01=B+r, f10=B, f11=B。
    auto chainTable = [](double B, double G, double r) {
      OrigPair p;
      p.f[0][0] = std::min(B + r, G);
      p.f[0][1] = B + r;
      p.f[1][0] = B;
      p.f[1][1] = B;
      return p;
    };

    int n = 3;
    std::vector<std::array<double, 2>> u((size_t)n, {0.0, 0.0});
    std::vector<OrigPair> pt;
    OrigPair p0 = chainTable(1.0, 4.0, 2.0); p0.i = 0; p0.j = 1; pt.push_back(p0);
    OrigPair p1 = chainTable(1.0, 4.0, 2.0); p1.i = 1; p1.j = 2; pt.push_back(p1);
    std::vector<int> fx((size_t)n, -1);
    fx[0] = 0;  // 网络输入钉死 NCHW

    const auto bf = brute(n, u, pt, fx);

    // 用分解构造求解器能量。
    infvino::BinaryEnergy en(n);
    bool submodular = true;
    for (const auto & p : pt)
      submodular &= en.addPairwiseTable(p.i, p.j, p.f[0][0], p.f[0][1], p.f[1][0], p.f[1][1]);
    en.fix(0, 0);
    CHECK(submodular, "R49: layout pairwise tables are submodular");
    const auto sol = infvino::solveBinaryMinCut(en);
    CHECK(sol.optimal, "R49: min-cut returns exact optimum");
    CHECK(std::fabs(sol.energy - bf.first) < 1e-6, "R49: min-cut energy == brute force optimum");
    CHECK(sol.labels[0] == 0, "R49: fixed (network input) stays NCHW");

    // 解算出的标签代入原表，能量应与解一致（验证分解无损）。
    const double solE = evalOrig(n, u, pt, fx, sol.labels);
    CHECK(std::fabs(solE - sol.energy) < 1e-6, "R49: decomposition is lossless (energy matches)");

    // 非 submodular 的表必须被拒绝（返回 false，不静默改语义）。
    // K = (f00+f11-f01-f10)/2 = (0+10-0-0)/2 = 5 > 0 → supermodular。
    {
      infvino::BinaryEnergy bad(2);
      CHECK(!bad.addPairwiseTable(0, 1, 0.0, 0.0, 0.0, 10.0),
            "R49: non-submodular table rejected");
    }

    // 全局最优 vs 逐节点独立 argmin：证明「逐节点贪心」确实会错过全局。
    // 变量 3 个，unary 想 1/0/1，链上强吸引 → 全局全 0 更优。
    {
      int m = 3;
      std::vector<std::array<double, 2>> uu = {{{0.0, 1.0}}, {{0.0, 3.0}}, {{0.0, 1.0}}};
      std::vector<OrigPair> pp;
      OrigPair q0; q0.i = 0; q0.j = 1; q0.f[0][0]=0; q0.f[0][1]=2.5; q0.f[1][0]=2.5; q0.f[1][1]=0; pp.push_back(q0);
      OrigPair q1 = q0; q1.i = 1; q1.j = 2; pp.push_back(q1);
      std::vector<int> nofix((size_t)m, -1);
      const auto g = brute(m, uu, pp, nofix);
      infvino::BinaryEnergy ge(m);
      for (const auto & p : pp) ge.addPairwiseTable(p.i, p.j, p.f[0][0], p.f[0][1], p.f[1][0], p.f[1][1]);
      for (int i = 0; i < m; ++i) ge.addUnary(i, uu[(size_t)i][0], uu[(size_t)i][1]);
      const auto gs = infvino::solveBinaryMinCut(ge);
      CHECK(std::fabs(gs.energy - g.first) < 1e-6, "R49: chain coupling optimum == brute force");
    }
  }

  std::printf("\n%s (%d failures)\n", g_fail ? "TUNING TEST FAILED" : "TUNING TEST PASSED", g_fail);
  return g_fail ? 1 : 0;
}
