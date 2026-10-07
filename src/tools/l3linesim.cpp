// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// l3linesim —— 行粒度 L3 替换模拟的实验台（纯 host，无需 GPU）。
//
//   --op probe   复现 `kernel_bench --op l3retain` 的访问形状（先一段流式 aggressor，
//                再对热点集合重用若干遍），看**热点命中率**随 aggressor 足迹的变化，
//                用公开文档所述的 1b-NRU 与严格 LRU 对照。
//   --op scale   随机访问流下测「每次访问」的耗时，量化复杂度，并外推到整网行粒度。
//
// 见 docs/round61-l3-line-granular-model-and-complexity.md。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include "infvino/L3LineModel.hpp"

using namespace infvino;

namespace
{
struct Rng
{
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed ? seed : 0x9e3779b97f4a7c15ull) {}
  uint64_t next()
  {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return s;
  }
};

double nowMs() { return std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now().time_since_epoch()).count(); }

int runProbe(int sets, int ways, uint64_t hot_lines, const std::string & reuse_list,
             const std::string & agg_list, int /*policy*/)
{
  auto parse = [](const std::string & s) {
    std::vector<uint64_t> v;
    std::string cur;
    for (char c : s + ",")
      if (c == ',') { if (!cur.empty()) v.push_back(std::strtoull(cur.c_str(), nullptr, 10)); cur.clear(); }
      else cur.push_back(c);
    return v;
  };
  const std::vector<uint64_t> reuses = parse(reuse_list);
  const std::vector<uint64_t> aggs = parse(agg_list);
  const uint64_t hot_base = 0x100000000ull;   // 固定热点基址（与 aggressor 不重叠）
  std::printf("[probe] sets=%d ways=%d capacity=%.2f MB (line=64B)\n", sets, ways,
              static_cast<double>(sets) * ways * 64.0 / 1e6);
  std::printf("  reuse  agg_MB   NRU1B_hit%%  LRU_hit%%\n");
  for (uint64_t reuse : reuses)
    for (uint64_t agg_mb : aggs)
    {
      const uint64_t agg_lines = agg_mb * 1000000ull / 64ull;
      double hr[2] = {0.0, 0.0};
      for (int p = 0; p < 2; ++p)
      {
        L3LineConfig cfg;
        cfg.sets = sets; cfg.ways = ways;
        cfg.policy = (p == 0) ? L3LinePolicy::NRU1B : L3LinePolicy::LRU;
        L3LineSim sim(cfg);
        for (uint64_t j = 0; j < hot_lines; ++j) sim.access(hot_base + j);  // 预热热点
        for (uint64_t i = 0; i < agg_lines; ++i) sim.access(i);             // aggressor 流式一次
        uint64_t hits = 0, acc = 0;
        for (uint64_t r = 0; r < reuse; ++r)
          for (uint64_t j = 0; j < hot_lines; ++j) { if (sim.access(hot_base + j)) ++hits; ++acc; }
        hr[p] = static_cast<double>(hits) / static_cast<double>(acc ? acc : 1);
      }
      std::printf("  %-5llu  %6llu   %8.1f    %7.1f\n", static_cast<unsigned long long>(reuse),
                  static_cast<unsigned long long>(agg_mb), 100.0 * hr[0], 100.0 * hr[1]);
    }
  return 0;
}

int runScale(uint64_t accesses, uint64_t working_mb, int sets, int ways, int reps, int policy)
{
  const uint64_t working_lines = working_mb * 1000000ull / 64ull;
  for (int p = 0; p < 2; ++p)
  {
    const int pol = p;   // 0 NRU1B, 1 LRU
    L3LineConfig cfg; cfg.sets = sets; cfg.ways = ways;
    cfg.policy = pol == 0 ? L3LinePolicy::NRU1B : L3LinePolicy::LRU;
    L3LineSim sim(cfg);
    // warm + time
    Rng rng(12345);
    const auto t0 = nowMs();
    uint64_t hits = 0;
    for (int r = 0; r < reps; ++r)
    {
      Rng rr(999 + r);
      for (uint64_t i = 0; i < accesses; ++i)
        if (sim.access(rr.next() % working_lines)) ++hits;
    }
    const auto t1 = nowMs();
    const double ms = t1 - t0;
    std::printf("[scale] policy=%-8s working=%.0f MB sets=%d ways=%d accesses=%llu reps=%d "
                "-> %.1f ms  (%.2f ns/access)  hit=%.1f%%\n",
                pol == 0 ? "NRU1B" : "LRU", static_cast<double>(working_mb), sets, ways,
                static_cast<unsigned long long>(accesses), reps, ms,
                ms * 1e6 / (static_cast<double>(accesses) * reps),
                100.0 * static_cast<double>(hits) / (static_cast<double>(accesses) * reps));
    (void)rng;
  }
  return 0;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string op = "probe";
  int sets = 512, ways = 120, reps = 3, policy = 0;
  uint64_t hot_lines = 16384, accesses = 1000000, working_mb = 64, agg = 0;
  std::string reuse_list = "1,4,16", agg_list = "4,6,8,12,16,25";
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--op") op = next();
    else if (a == "--sets") sets = std::atoi(next().c_str());
    else if (a == "--ways") ways = std::atoi(next().c_str());
    else if (a == "--hot-lines") hot_lines = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--reuse") reuse_list = next();
    else if (a == "--agg") agg_list = next();
    else if (a == "--accesses") accesses = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--working-mb") working_mb = std::strtoull(next().c_str(), nullptr, 10);
    else if (a == "--reps") reps = std::atoi(next().c_str());
    else if (a == "--policy") policy = std::atoi(next().c_str());
    else if (a == "--both") policy = -1;
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
    (void)agg;
  }
  if (op == "probe") return runProbe(sets, ways, hot_lines, reuse_list, agg_list, 0);
  if (op == "scale") return runScale(accesses, working_mb, sets, ways, reps, policy);
  std::fprintf(stderr, "unknown op: %s\n", op.c_str());
  return 2;
}
