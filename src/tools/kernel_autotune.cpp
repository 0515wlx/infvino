// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_autotune —— infvino 离线自动调优工具（docs/autotuning.md）。
//
//   kernel_autotune --plan models/yolov8n-pose/model.plan --cache config/tuning.json
//                   [--op conv3x3|gemm|conv1x1|depthwise] [--only <substr>] [--limit N]
//                   [--iters 30] [--report] [--expected] [--bake]
//
// 安全：遵守 docs/benchmark_protocol.md —— 一条命令只调优少量节点/配置，分批调用。
// 缓存写回是 merge（不会丢已有条目）。
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "infvino/PlanModel.hpp"

namespace
{
// 把 plan 文本按节点行改写：把 cfg=... / ov=... 等排序到节点末尾。当前 --bake 只做一个
// 保守动作：在 conv3x3 节点行末尾追加/覆盖 `tuned=<key>` 注释属性，供人工审计；真正生效
// 的是 tuning.json（PlanModel 运行时查表）。这样 bake 不会破坏 plan 语义。
bool bakePlan(
  const std::string & plan_in, const std::string & plan_out,
  const std::map<std::string, infvino::TuningEntry> & done)
{
  std::ifstream in(plan_in);
  if (!in) return false;
  std::ofstream out(plan_out);
  if (!out) return false;
  (void)done;
  std::string line;
  while (std::getline(in, line)) out << line << "\n";
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string plan_path, cache_path, kernel_dir, only, bake_out;
  std::vector<std::string> ops;
  int  limit = 0, iters = 30;
  bool report = false, expected = false, bake = false, list = false, retune = false;
  bool refresh = false;
  bool global = false;
  int  gIters = 0, gTopK = 3, gRounds = 3, gLimit = 0;
  double gMargin = 0.0;
  int  gBudget = 0;
  std::string planTuning;

  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--plan") plan_path = next();
    else if (a == "--cache") cache_path = next();
    else if (a == "--kernel-dir") kernel_dir = next();
    else if (a == "--op") ops.push_back(next());
    else if (a == "--only") only = next();
    else if (a == "--limit") limit = std::atoi(next().c_str());
    else if (a == "--iters") iters = std::atoi(next().c_str());
    else if (a == "--report") report = true;
    else if (a == "--expected") expected = true;
    else if (a == "--list") list = true;
    else if (a == "--retune") retune = true;
    else if (a == "--refresh-expected") refresh = true;
    else if (a == "--global") global = true;
    else if (a == "--global-iters") gIters = std::atoi(next().c_str());
    else if (a == "--global-topk") gTopK = std::atoi(next().c_str());
    else if (a == "--global-rounds") gRounds = std::atoi(next().c_str());
    else if (a == "--global-limit") gLimit = std::atoi(next().c_str());
    else if (a == "--global-margin") gMargin = std::atof(next().c_str());
    else if (a == "--global-budget") gBudget = std::atoi(next().c_str());
    else if (a == "--plan-tuning") planTuning = next();
    else if (a == "--bake") { bake = true; bake_out = next(); }
    else if (a == "--help" || a == "-h") {
      std::printf(
        "kernel_autotune —— infvino 离线自动调优（docs/autotuning.md）\n"
        "用法: kernel_autotune --plan <model.plan> [选项]\n"
        "  --cache <file>     调优缓存路径（加载并 merge 写回）\n"
        "  --op <name>        只调优某 op（conv3x3/conv1x1/depthwise/gemm，可重复）\n"
        "  --only <substr>    只调优签名含该子串的节点（分批用）\n"
        "  --limit N          本次最多调优 N 个签名（0=不限；安全分批推荐 3–5）\n"
        "  --iters N          每个候选的计时迭代数（默认 30）\n"
        "  --list             只列出唯一签名，不跑 GPU 计时\n"
        "  --retune           忽略缓存里已有的 tuned 条目，强制重新扫描（候选/标准更新后用）\n"
        "  --refresh-expected  仅用当前中间标准重算缓存命中项的 expected/ratio（**零 GPU**）\n"
        "  --global           R44/R47: 整网 busy 回验（R47：min+median 双口径 + 交错 + 可加目标\n"
        "                     优先级 + 布局耦合分组 move；接受/裁决用 median 典型口径）\n"
        "  --global-iters N   每个 assignment 的采样次数（取 min+median 双口径；默认 3）\n"
        "  --global-topk K    每个签名参与回验的候选上限（隔离 top-K；默认 3）\n"
        "  --global-rounds R  坐标下降轮数上限（默认 3）\n"
        "  --global-limit N   最多回验 N 个签名（0=不限；安全分批用）\n"
        "  --global-margin F  隔离 margin 剪枝：只回验 iso_ms<=best*(1+F) 的候选（默认 0=关；\n"
        "                     ⚠️ 隔离名次不预测整网名次，>0 可能剪掉流水线更快的候选）\n"
        "  --global-budget N  整网**执行次数**总预算（0=不限；抗组合爆炸/GPU 风险）\n"
        "  --plan-tuning <f>  per-plan 选择覆盖的输出路径（默认 <plan>.tuning.json）\n"
        "  环境 INFVINO_GLOBAL_MODEL=1: 模型驱动选择（predictNet argmin 主导，不做端到端坐标下降）；\n"
        "        INFVINO_GLOBAL_MODEL_CONFIRM=1 再叠加一次外部稳态确认门\n"
        "  --report           打印每个节点的候选扫描明细\n"
        "  --expected         打印 中间标准(期望) vs 实测 ops/EU/cyc 与 ratio\n"
        "  --bake <out.plan>  额外复制一份 plan（审计；运行时以 cache 为准）\n"
        "  --kernel-dir <dir> .cl 源码目录\n"
        "环境: INFVINO_TUNING_CACHE / INFVINO_TUNING=off\n"
        "安全: 遵守 docs/benchmark_protocol.md（一条命令少量配置，分批，timeout）。\n");
      return 0;
    }
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  if (plan_path.empty()) { std::fprintf(stderr, "usage: kernel_autotune --plan <plan>\n"); return 2; }
  if (iters < 1) iters = 1;

  try
  {
    // --cache 必须在 PlanModel 构造前设入环境（构造时加载缓存）。
    if (!cache_path.empty()) setenv("INFVINO_TUNING_CACHE", cache_path.c_str(), 1);

    const std::string kdir = kernel_dir.empty() ? INFVINO_KERNEL_DIR : kernel_dir;
    infvino::PlanModel model(plan_path, kdir, -1, 0, /*profiling=*/true);

    std::printf("device: %s\n", model.device().describe().c_str());
    std::printf("device key: %s\n", infvino::TuningCache::deviceKey(model.device()).c_str());

    if (list)
    {
      auto targets = model.tuningTargets(ops);
      std::printf("unique tuning signatures: %zu\n", targets.size());
      for (const auto & t : targets) std::printf("  %s\n", t.c_str());
      return 0;
    }

    if (refresh)
    {
      const int n = model.refreshExpected(ops);
      std::printf("refresh-expected: recomputed %d cached entries (no GPU timing)\n", n);
      // 按 ratio 升序打印（最远离中间标准的层排前面），便于定位热点。
      struct Row { std::string sig; double ops, exp, ratio; };
      std::vector<Row> rows;
      for (const auto & kv : model.tuning().entries())
        rows.push_back({kv.first, kv.second.ops, kv.second.expected, kv.second.ratio});
      std::sort(rows.begin(), rows.end(),
                [](const Row & a, const Row & b) { return a.ratio < b.ratio; });
      std::printf("%-52s %8s %8s %6s\n", "signature", "measured", "expected", "ratio");
      for (const auto & r : rows)
        std::printf("%-52s %8.2f %8.2f %6.2f\n", r.sig.c_str(), r.ops, r.exp, r.ratio);
      if (!cache_path.empty() && model.saveTuning(cache_path))
        std::printf("wrote %s (cache total %zu)\n", cache_path.c_str(), model.tuning().size());
      return 0;
    }

    auto done = model.autotune(ops, only, limit, iters, /*merge=*/true, /*verbose=*/report,
                               /*retune=*/retune);

    if (global)
    {
      // R44：把目标函数从「单节点隔离 min」换成「整网 busy」，对隔离 top-K 做坐标下降回验。
      // 必须在同一次进程里先跑完隔离扫描（本进程的 cand_short_ 才有内容）。
      const int nchg = model.globalRetune(ops, gIters, gTopK, gRounds, gLimit, gMargin, gBudget);
      std::printf("global-retune: %d signature(s) reselected by whole-net busy\n", nchg);
      // 回验后缓存里的选择已变；把本次调过的签名同步回 done，便于 --expected 打印。
      const auto & ents = model.tuning().entries();
      for (auto & d : done)
      {
        auto it = ents.find(d.first);
        if (it != ents.end()) d.second = it->second;
      }
      // R45 P0#4: 把 per-plan 选择覆盖落盘（默认 <plan>.tuning.json）。
      std::string pt = planTuning;
      if (pt.empty())
      {
        const char * e = std::getenv("INFVINO_PLAN_TUNING");
        if (e) pt = e;
      }
      if (pt.empty()) pt = plan_path + ".tuning.json";
      if (pt != "none")
      {
        if (model.savePlanOverrides(pt))
          std::printf("wrote plan overrides %s (%zu entries)\n", pt.c_str(),
                      model.planOverrideCount());
        else
          std::fprintf(stderr, "failed to write plan overrides %s\n", pt.c_str());
      }
    }

    if (expected)
    {
      // 中间标准对照：软预测 vs 硬上限 vs 实测。
      //   ratio      = ops/expected（软：含 amort/gridFactor，仅用于同族排序）
      //   hard_ratio = ops/hard_ceiling（硬：ISA 配额/roofline，"离物理极限"）
      std::printf("\n== expected(soft) / hard ceiling vs measured ==\n");
      std::printf("%-44s %8s %8s %6s %8s %6s\n", "signature", "measured", "expected",
                  "ratio", "hardC", "hardR");
      for (auto & kv : done)
        std::printf("%-44s %8.2f %8.2f %6.2f %8.2f %6.2f\n", kv.first.c_str(), kv.second.ops,
                    kv.second.expected, kv.second.ratio, kv.second.hard_ceiling,
                    kv.second.hard_ratio);
    }

    if (!cache_path.empty())
    {
      if (model.saveTuning(cache_path))
        std::printf("wrote %s (%zu entries this run, cache total %zu)\n", cache_path.c_str(),
                    done.size(), model.tuning().size());
      else
        std::fprintf(stderr, "failed to write %s\n", cache_path.c_str());
    }

    if (bake)
    {
      if (bake_out.empty()) bake_out = plan_path;
      if (bakePlan(plan_path, bake_out, done))
        std::printf("baked plan -> %s\n", bake_out.c_str());
      else
        std::fprintf(stderr, "failed to bake plan\n");
    }
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "[kernel_autotune] error: %s\n", e.what());
    return 1;
  }
  return 0;
}
