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
    if (best.kernel.empty() || ms < best.ms) {
      best.kernel = c.kernel;
      best.config = c.config;
      best.options = c.options;
      best.ms = ms;
      best.ops = ops;
    }
  }
  if (!best.kernel.empty()) {
    // 中间标准：优先用「胜出族」自己的上限模型（修 R33 的口径失真）。
    if (const KernelFamily * f = familyByName(best.kernel))
      if (f->ceiling) best.expected = f->ceiling(sig, rt.info());
    best.ratio = best.expected > 0 ? best.ops / best.expected : 0.0;
  }
  return best;
}

}  // namespace infvino
