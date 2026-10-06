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
}

ITEST_MAIN("test_l3_model")
