// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_l3_model —— 全局 LRU 溢出模拟 + 有限差分逐节点定价的离线回归（R55 方案 A）。
//
// 覆盖不变量：
//   * 生产者写、消费者读（放得下）→ 0 miss（命中）；
//   * 冷输入（无人写）→ 记一次 miss；
//   * 占用把有效容量压到读不进 → 逐出 → 后续读 miss；
//   * 逐节点价 ρ_i ≥ 0；唯一「元凶」时 ρ_i == 该次溢出耗时；占用无害时 ρ_i == 0；
//   * compute_prices=false 时不产生价格。
//
// 纯 CPU、不需要 GPU。
#include <cmath>
#include <string>
#include <vector>

#include "infvino/L3Model.hpp"
#include "test_util.hpp"

using namespace infvino;

namespace
{
L3Access acc(std::string op, double occ)
{
  L3Access a;
  a.op = op;
  a.occ_bytes = occ;
  return a;
}
constexpr double MB = 1e6;
}  // namespace

static void run_tests()
{
  // 测试用配置：干净的数字（spill_per_byte_ms=1e-7 → 1MB miss = 0.1ms）。
  L3ModelConfig cfg;
  cfg.l3_bytes = 3.75 * MB;
  cfg.eff_cap_anchor = 1.0 * MB;
  cfg.occ_cap_slope = 0.5;
  cfg.spill_per_byte_ms = 1e-7;
  cfg.compute_prices = true;

  // --- 生产者→消费者命中：写过的张量被读到 → 0 miss ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("producer", 0.0));
    n.back().writes.push_back({"t", 1.0 * MB});
    n.push_back(acc("consumer", 0.0));
    n.back().reads.push_back({"t", 1.0 * MB});
    const L3Result r = evaluateL3(n, cfg);
    CHECK_NEAR(r.spill_bytes, 0.0, 1e-9, "producer->consumer resident: no miss");
    CHECK_NEAR(r.spill_ms, 0.0, 1e-12, "producer->consumer resident: no spill ms");
  }

  // --- 冷输入：无人写的读 → 一次 miss ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("reader", 0.0));
    n.back().reads.push_back({"in", 2.0 * MB});
    const L3Result r = evaluateL3(n, cfg);
    CHECK_NEAR(r.spill_bytes, 2.0 * MB, 1e-6, "cold input read counts one miss");
  }

  // --- 占用压缩容量 → 逐出 → 后续读 miss，且唯一元凶夺得全部价格 ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("p", 0.0));            // 写 t (0.5MB)
    n.back().writes.push_back({"t", 0.5 * MB});
    n.push_back(acc("aggressor", 8.0 * MB));  // 占用 8MB → cap=max(1MB, 3.75−4)=1MB
    n.back().writes.push_back({"u", 0.7 * MB});  // t(0.5)+u(0.7)=1.2>1 → 逐出 t
    n.push_back(acc("c", 0.0));
    n.back().reads.push_back({"t", 0.5 * MB});   // t 已被逐出 → miss
    const L3Result r = evaluateL3(n, cfg);
    CHECK_NEAR(r.spill_bytes, 0.5 * MB, 1e-6, "occupancy evicts resident tensor -> miss");
    CHECK(r.node_price_ms.size() == 3, "prices sized to node count");
    CHECK(r.node_price_ms[0] == 0.0 && r.node_price_ms[2] == 0.0,
          "non-aggressor nodes have zero price");
    CHECK_NEAR(r.node_price_ms[1], 0.5 * MB * 1e-7, 1e-12,
               "sole aggressor carries the whole spill price");
  }

  // --- 占用无害（容量仍够）→ 价格 0、无 miss ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("p", 0.0));
    n.back().writes.push_back({"t", 0.5 * MB});
    n.push_back(acc("mild", 0.1 * MB));   // cap=3.7MB，t 留得住
    n.back().writes.push_back({"u", 0.7 * MB});
    n.push_back(acc("c", 0.0));
    n.back().reads.push_back({"t", 0.5 * MB});
    const L3Result r = evaluateL3(n, cfg);
    CHECK_NEAR(r.spill_bytes, 0.0, 1e-9, "mild occupancy does not evict -> no miss");
    CHECK_NEAR(r.node_price_ms[1], 0.0, 1e-12, "mild occupancy has zero price");
  }

  // --- 价格随占用单调（更大的占用 ≥ 更小的占用）---
  {
    auto priceFor = [&](double occ) {
      std::vector<L3Access> n;
      n.push_back(acc("p", 0.0));
      n.back().writes.push_back({"t", 0.5 * MB});
      n.push_back(acc("a", occ));
      n.back().writes.push_back({"u", 1.0 * MB});
      n.push_back(acc("c", 0.0));
      n.back().reads.push_back({"t", 0.5 * MB});
      return evaluateL3(n, cfg).node_price_ms[1];
    };
    const double lo = priceFor(1.0 * MB), hi = priceFor(8.0 * MB);
    CHECK(lo >= 0.0 && hi >= 0.0, "prices non-negative");
    CHECK(hi >= lo, "price monotone in occupancy");
  }

  // --- 关闭定价副产物时不产出价格向量 ---
  {
    L3ModelConfig noPrice = cfg;
    noPrice.compute_prices = false;
    std::vector<L3Access> n;
    n.push_back(acc("r", 1.0 * MB));
    n.back().reads.push_back({"in", 1.0 * MB});
    const L3Result r = evaluateL3(n, noPrice);
    CHECK(r.node_price_ms.empty(), "compute_prices=false -> no prices");
    CHECK(r.spill_bytes > 0.0, "spill still computed without prices");
  }

  // --- 确定性：同输入两次结果一致 ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("p", 2.0 * MB));
    n.back().writes.push_back({"t", 1.0 * MB});
    n.push_back(acc("c", 0.0));
    n.back().reads.push_back({"t", 1.0 * MB});
    const L3Result a = evaluateL3(n, cfg);
    const L3Result b = evaluateL3(n, cfg);
    CHECK_NEAR(a.spill_bytes, b.spill_bytes, 1e-9, "evaluateL3 deterministic");
    CHECK_NEAR(a.node_price_ms[0], b.node_price_ms[0], 1e-12, "prices deterministic");
  }

  // ================= R60: 精确 spill（复用/栈距离模型）=================

  // --- 栈距离 CDF 与「常数容量有限 LRU 模拟」逐位一致 ---
  {
    auto buildSeq = []() {
      std::vector<L3Access> n;
      const double sz[8] = {0.5, 0.3, 0.9, 0.7, 1.1, 0.4, 0.8, 0.6};
      const char * nm[8] = {"a", "b", "c", "d", "e", "f", "g", "h"};
      unsigned seed = 12345u;
      auto rnd = [&]() { seed = seed * 1103515245u + 12345u; return (seed >> 16) & 0x7fffu; };
      for (int step = 0; step < 60; ++step)
      {
        L3Access a;
        a.op = "n" + std::to_string(step);
        const int t = static_cast<int>(rnd() % 8), u = static_cast<int>(rnd() % 8);
        if (rnd() % 2)
          a.writes.push_back({nm[t], sz[t] * MB});
        else
          a.reads.push_back({nm[t], sz[t] * MB});
        if (u != t) a.reads.push_back({nm[u], sz[u] * MB});
        n.push_back(std::move(a));
      }
      return n;
    };
    const std::vector<L3Access> seq = buildSeq();
    bool allEq = true;
    for (double C : {0.5, 1.0, 2.0, 3.75, 8.0})
    {
      L3ModelConfig cc;
      cc.l3_bytes = C * MB;
      cc.eff_cap_anchor = C * MB;
      cc.occ_cap_slope = 0.0;
      cc.compute_prices = false;
      const double fin = evaluateL3(seq, cc).spill_bytes;
      const double cdf = spillAtCapacity(seq, C * MB, L3Policy::LRU);
      if (std::abs(fin - cdf) > 1e-6) allEq = false;
    }
    CHECK(allEq, "R60: stack-distance CDF == constant-capacity finite LRU spill");

    // 复用/栈距离分解：非冷访问 d_t == I_other + bytes；冷访问 d_t == ∞。
    const L3StackProfile prof = profileL3Stack(seq, L3ModelConfig{});
    bool decomp = true;
    int colds = 0;
    for (const auto & t : prof.touches)
    {
      if (t.cold) { ++colds; if (!std::isinf(t.stack_dist)) decomp = false; }
      else if (std::abs(t.stack_dist - (t.reuse_bytes + t.bytes)) > 1e-9) decomp = false;
    }
    CHECK(decomp, "R60: stack_dist == interference + own bytes (Delta_local + I_other)");
    CHECK(colds > 0, "R60: cold accesses are flagged");
    CHECK_NEAR(prof.spill_bytes, evaluateL3(seq, L3ModelConfig{}).spill_bytes, 1e-9,
               "R60: profile spill == evaluateL3 spill");
  }

  // --- 精确逐出归因：唯一 aggressor 承担全部（干扰）价，Σ价 ≤ spill；冷读不记价 ---
  {
    std::vector<L3Access> n;
    n.push_back(acc("p", 0.0));
    n.back().writes.push_back({"t", 0.5 * MB});
    n.push_back(acc("aggressor", 8.0 * MB));
    n.back().writes.push_back({"u", 0.7 * MB});
    n.push_back(acc("c", 0.0));
    n.back().reads.push_back({"t", 0.5 * MB});
    const L3Result r = evaluateL3(n, cfg);
    CHECK(r.node_evict_ms.size() == 3, "R60: evict prices sized to node count");
    CHECK(r.node_evict_ms[0] == 0.0 && r.node_evict_ms[2] == 0.0,
          "R60: non-evictors have zero evict price");
    CHECK_NEAR(r.node_evict_ms[1], 0.5 * MB * 1e-7, 1e-12,
               "R60: sole evictor carries the whole interference price");
    double sum = 0.0;
    for (double v : r.node_evict_ms) sum += v;
    CHECK(sum <= r.spill_ms + 1e-12, "R60: evict blame <= total spill");

    std::vector<L3Access> c;
    c.push_back(acc("r", 0.0));
    c.back().reads.push_back({"in", 2.0 * MB});
    const L3Result rc = evaluateL3(c, cfg);
    double csum = 0.0;
    for (double v : rc.node_evict_ms) csum += v;
    CHECK_NEAR(csum, 0.0, 1e-12, "R60: cold read is not blamed on any node");
  }

  // --- NRU 等价工程模型：evict-first 插入保护反复重用的热点（抗流式污染）---
  {
    std::vector<L3Access> n;
    n.push_back(acc("p", 0.0));
    n.back().writes.push_back({"hot", 0.5 * MB});
    n.push_back(acc("rd", 0.0));
    n.back().reads.push_back({"hot", 0.5 * MB});   // 预热：命中 → 转为受保护
    for (int round = 0; round < 6; ++round)
    {
      for (int k = 0; k < 4; ++k)
      {
        L3Access s;
        s.op = "s";
        s.writes.push_back({"s" + std::to_string(round) + "_" + std::to_string(k), 0.5 * MB});
        n.push_back(std::move(s));
      }
      L3Access rd;
      rd.op = "rd";
      rd.reads.push_back({"hot", 0.5 * MB});
      n.push_back(std::move(rd));
    }
    L3ModelConfig base;
    base.l3_bytes = 1.0 * MB;
    base.eff_cap_anchor = 1.0 * MB;
    base.occ_cap_slope = 0.0;
    base.compute_prices = false;
    L3ModelConfig lru = base;
    lru.policy = L3Policy::LRU;
    L3ModelConfig nru = base;
    nru.policy = L3Policy::NRU;
    const double sl = evaluateL3(n, lru).spill_bytes;
    const double sn = evaluateL3(n, nru).spill_bytes;
    CHECK(sl > 0.0, "R60: strict LRU evicts the hot tensor under streaming pressure");
    CHECK(sn < sl, "R60: NRU (evict-first insert) protects the reused hot tensor");
  }
}

ITEST_MAIN("test_l3_model")
