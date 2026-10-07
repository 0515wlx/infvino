// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Autotuner 实现：候选枚举（**全部委托给算子族注册表**）+ 计时。
//
// 候选 config 的单一真相源是 src/KernelFamilies.cpp（见 docs/kernel-families.md）：
// 这里的 candidatesXxx 只是 `candidatesFromRegistry(sig)` 的薄包装，保留旧 API 供
// PlanModel / kernel_autotune 使用。加任何算子族都不再需要改本文件。
#include "infvino/Autotuner.hpp"

#include "infvino/KernelFamily.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <stdexcept>

namespace infvino
{

std::vector<Candidate> candidatesConv3x3(const OpSignature & sig) { return candidatesFromRegistry(sig); }
std::vector<Candidate> candidatesGemm(const OpSignature & sig) { return candidatesFromRegistry(sig); }
std::vector<Candidate> candidatesConv1x1(const OpSignature & sig) { return candidatesFromRegistry(sig); }
std::vector<Candidate> candidatesDepthwise(const OpSignature & sig) { return candidatesFromRegistry(sig); }
std::vector<Candidate> candidatesSmall(const OpSignature & sig) { return candidatesFromRegistry(sig); }

bool benchCandidate(
  ClRuntime & rt, const std::function<cl_event()> & enqueue, int iters, double * ms,
  double * spread)
{
  try {
    // R42: use the MINIMUM over the measured samples, not the median. External
    // interference (LLC->DRAM spill near the 3.75 MB boundary, thermal, other
    // containers) can only ADD time, so the min is a far more stable estimator of
    // the kernel's intrinsic cost (min is ~±1–2% across repeats; the median shows
    // ±10–20% tails). `spread` = (p90/min − 1) exposes residual instability.
    double mn = 0.0, p90 = 0.0;
    rt.timeMs(enqueue, 5, iters, &mn, &p90);
    *ms = mn;
    if (spread) *spread = (mn > 0.0) ? (p90 / mn - 1.0) : 0.0;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

TuningEntry autotuneOp(
  ClRuntime & rt, const OpSignature & sig, const std::vector<Candidate> & cands,
  const std::function<std::function<cl_event()>(const Candidate &)> & makeEnqueue,
  double flops, int iters, std::vector<TuningEntry> * measured)
{
  TuningEntry best;
  best.expected = expectedOps(sig, rt.info(), &best);
  best.device_id = TuningCache::deviceKey(rt.info());
  best.iters = iters;
  best.source = "tuned";

  // 给一个（已填 kernel/options/ms/ops 的）entry 补齐中间标准：优先用「胜出族」自己的
  // 上限模型（修 R33 的口径失真），硬上限只取 ISA 配额/roofline 下界（R43）。
  auto applyStandard = [&](TuningEntry & e) {
    const KernelFamily * f = familyByName(e.kernel);
    e.expected = expectedOps(sig, rt.info(), &e);
    if (f && f->ceiling)
    {
      const double famCeil = f->ceiling(sig, rt.info());
      // R47-L3: conv/gemm 软标尺同时受「族计算上限」与「内存 roofline（含 L3 断崖）」约束。
      const bool memBind = (sig.op == "conv3x3" || sig.op == "gemm" || sig.op == "conv1x1" ||
                            sig.op == "conv1x1_cat4");
      e.expected = memBind ? std::min(e.expected, famCeil) : famCeil;
    }
    e.hard_ceiling = (f && f->hardCeiling) ? f->hardCeiling(sig, rt.info()) : e.expected;
    if (e.expected <= 0.0) e.expected = e.hard_ceiling;
    e.ratio = e.expected > 0 ? e.ops / e.expected : 0.0;
    e.hard_ratio = e.hard_ceiling > 0 ? e.ops / e.hard_ceiling : 0.0;
  };

  const bool dbg = std::getenv("INFVINO_AUTOTUNE_DEBUG") != nullptr;
  // R45 P2#8: select on min, then **re-measure the top-K** at the end of the sweep.
  // Sequential scanning suffers from thermal drift (earlier candidates measured in a
  // cooler state); re-measuring the top-K in the then-current state and taking the min
  // removes most of that systematic bias (R42 §3.3). We also warn on high spread.
  const double kSpreadWarn = 0.15;   // 15% (p90/min − 1)
  int noisy = 0;
  int skipped = 0;   // R45 P1#11: build/bench 失败的候选数（静默缩小候选集）
  struct Measured { TuningEntry e; const Candidate * c; };
  std::vector<Measured> all;
  if (dbg)
    std::fprintf(stderr, "[autotune] %s: %zu candidates\n", sig.str().c_str(), cands.size());
  for (const auto & c : cands) {
    std::function<cl_event()> enq;
    try {
      enq = makeEnqueue(c);
    } catch (const std::exception & e) {
      if (dbg)
        std::fprintf(stderr, "  [skip] %-18s %-16s makeEnqueue: %s\n", c.kernel.c_str(),
                     c.config.c_str(), e.what());
      ++skipped;
      continue;  // build 失败（资源/编译）→ 跳过
    }
    double ms = 0, sp = 0;
    if (!benchCandidate(rt, enq, iters, &ms, &sp)) {
      if (dbg)
        std::fprintf(stderr, "  [skip] %-18s %-16s bench failed\n", c.kernel.c_str(),
                     c.config.c_str());
      ++skipped;
      continue;
    }
    if (sp > kSpreadWarn) ++noisy;
    TuningEntry me;
    me.kernel = c.kernel;
    me.config = c.config;
    me.options = c.options;
    me.ms = ms;
    me.ops = rt.opsPerEuCycle(flops, ms);
    me.iters = iters;
    me.device_id = best.device_id;
    me.source = "candidate";
    me.exact = c.exact;   // R48: 数值契约随候选记录
    me.tol = c.tol;
    if (dbg)
      std::fprintf(stderr, "  [cand] %-18s %-42s %8.4f ms  ops=%6.2f  spread=%+.0f%%\n",
                   c.kernel.c_str(), c.options.c_str(), ms, me.ops, sp * 100.0);
    all.push_back({std::move(me), &c});
  }

  // Re-measure top-K at the current thermal point (take min over both measurements).
  {
    std::stable_sort(all.begin(), all.end(),
                     [](const Measured & a, const Measured & b) { return a.e.ms < b.e.ms; });
    const int K = std::min<int>(3, static_cast<int>(all.size()));
    const int iters2 = std::max(iters, 8);
    double winnerSpread = 0.0;
    for (int i = 0; i < K; ++i)
    {
      try
      {
        auto enq2 = makeEnqueue(*all[i].c);
        double ms2 = 0, sp2 = 0;
        if (benchCandidate(rt, enq2, iters2, &ms2, &sp2))
        {
          if (ms2 < all[i].e.ms)
          {
            all[i].e.ms = ms2;
            all[i].e.ops = rt.opsPerEuCycle(flops, ms2);
          }
          if (i == 0) winnerSpread = sp2;
        }
      }
      catch (const std::exception &) {}
    }
    if (all.empty()) { /* nothing measured */ }
    else
    {
      std::stable_sort(all.begin(), all.end(),
                       [](const Measured & a, const Measured & b) { return a.e.ms < b.e.ms; });
      best.kernel = all[0].e.kernel;
      best.config = all[0].e.config;
      best.options = all[0].e.options;
      best.ms = all[0].e.ms;
      best.ops = all[0].e.ops;
      best.exact = all[0].e.exact;   // R48: 数值契约随胜出候选带上来
      best.tol = all[0].e.tol;
    }
    if (winnerSpread > kSpreadWarn && !best.kernel.empty())
      std::fprintf(stderr,
        "[autotune] WARN %s: winner %s unstable (spread %.0f%%); result may be "
        "LLC/DRAM-noise limited — rerun (scripts/gpu_clocks.sh lock) or isolate.\n",
        sig.str().c_str(), best.kernel.c_str(), winnerSpread * 100.0);
  }

  if (noisy > 0 &&
      noisy >= std::max<int>(2, static_cast<int>(cands.size() / 10)))
    std::fprintf(stderr,
      "[autotune] WARN %s: %d/%zu candidates unstable (spread>%.0f%%) — likely LLC "
      "spill/DRAM contention.\n",
      sig.str().c_str(), noisy, cands.size(), kSpreadWarn * 100.0);
  // R45 P1#11: 候选静默跳过会缩小搜索空间（R37 曾因此丢掉 SLM 候选）。>20% 即可疑。
  if (skipped > 0 && skipped * 5 >= static_cast<int>(cands.size()))
    std::fprintf(stderr,
      "[autotune] WARN %s: %d/%zu candidates skipped (build/enqueue failed) — candidate "
      "set may be silently reduced; check geometry/parse.\n",
      sig.str().c_str(), skipped, cands.size());
  if (!best.kernel.empty()) applyStandard(best);
  if (measured)
  {
    measured->clear();
    for (auto & m : all) { applyStandard(m.e); measured->push_back(m.e); }
    std::stable_sort(measured->begin(), measured->end(),
                     [](const TuningEntry & a, const TuningEntry & b) { return a.ms < b.ms; });
  }
  return best;
}

}  // namespace infvino
