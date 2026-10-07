// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3Model 实现：全局 L3 溢出模拟 + 复用/栈距离画像 + 精确逐出归因 + 有限差分逐节点价。
// 见 L3Model.hpp。
#include "infvino/L3Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <unordered_map>

namespace infvino
{
namespace
{

constexpr double kInf = std::numeric_limits<double>::infinity();

struct TensorState
{
  bool   ever = false;    ///< 是否曾被访问过（区分冷启动）
  size_t lastEvictor = 0; ///< 上一次把它逐出栈的节点（无则 0）
};

struct SimStats
{
  double              miss = 0.0;
  std::vector<double> blame;   ///< 逐节点：其逐出导致、之后真 miss 的字节（冷启动不计）
  std::unordered_map<std::string, int> touchCount;  ///< 每个张量在整条序列里的访问次数
};

// 单一求值核心：按拓扑序跑一遍有限 L3 模拟；`prof` 非空时同时记录无限栈的复用距离画像。
SimStats simulateCore(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg,
                      L3StackProfile * prof)
{
  struct Item
  {
    std::string name;
    double      bytes = 0.0;
    bool        recent = true;   ///< NRU 标志（严格 LRU 下不使用）
  };
  std::vector<Item> res;                     // MRU 在尾
  std::vector<std::pair<std::string, double>> inf;  // 无限栈，MRU 在尾（仅画像时维护）
  std::unordered_map<std::string, TensorState> st;
  double resBytes = 0.0;

  SimStats out;
  out.blame.assign(nodes.size(), 0.0);
  for (const auto & n : nodes)
  {
    for (const auto & r : n.reads) ++out.touchCount[r.first];
    for (const auto & w : n.writes) ++out.touchCount[w.first];
  }
  if (prof)
  {
    prof->node_evict_bytes.assign(nodes.size(), 0.0);
    prof->spill_bytes = 0.0;
    prof->total_bytes = 0.0;
    prof->touches.clear();
  }

  // 逐出一个牺牲者（LRU=最旧；NRU=优先未置位者，全置位则清位后取最旧）。
  auto evictOne = [&](size_t nodeIdx) {
    size_t victim = 0;
    if (cfg.policy == L3Policy::NRU)
    {
      size_t k = res.size();
      for (size_t j = 0; j < res.size(); ++j)
        if (!res[j].recent) { k = j; break; }
      if (k == res.size())
      {
        for (auto & it : res) it.recent = false;   // aging
        k = 0;
      }
      victim = k;
    }
    st[res[victim].name].lastEvictor = nodeIdx;
    resBytes -= res[victim].bytes;
    res.erase(res.begin() + static_cast<long>(victim));
  };

  // 一次访问：更新有限栈 + 逐出/归因；返回是否命中。
  auto touchFinite = [&](const std::string & name, double bytes, bool write, size_t ni,
                         double cap) -> bool {
    for (size_t k = 0; k < res.size(); ++k)
      if (res[k].name == name)
      {
        Item pv = res[k];
        pv.recent = true;
        res.erase(res.begin() + static_cast<long>(k));
        res.push_back(pv);
        return true;
      }
    const bool wasKnown = st[name].ever;
    if (!write) out.miss += bytes;                       // 读 miss 计一次 DRAM 往返
    if (!write && wasKnown) out.blame[st[name].lastEvictor] += bytes;  // 非冷启动 → 归因
    // NRU（等价工程模型）：**单次使用（流式）**的数据以 evict-first 插入，**会被重用**
    // （整条序列访问次数 > 1）的数据受保护；再次命中即置位。这样反复重用的热点抗流式污染，
    // 与自标定曲线（knee ≈ 11 MB × reuse^0.30，见 config/l3_calibration.json）一致。
    const bool recentOnFill =
        (cfg.policy != L3Policy::NRU) ||
        (out.touchCount.count(name) && out.touchCount[name] > 1);
    res.push_back({name, bytes, recentOnFill});
    st[name].ever = true;
    resBytes += bytes;
    while (resBytes > cap && !res.empty()) evictOne(ni);
    return false;
  };

  for (size_t ni = 0; ni < nodes.size(); ++ni)
  {
    const L3Access & n = nodes[ni];
    // 占用压缩有效容量（R55：锚点 ~2MB）；先逐出至新容量。
    const double cap = std::max(cfg.eff_cap_anchor, cfg.l3_bytes - cfg.occ_cap_slope * n.occ_bytes);
    while (resBytes > cap && !res.empty()) evictOne(ni);

    auto access = [&](const std::string & name, double bytes, bool write) {
      // ---- 有限模拟（权威 spill / 命中 / 逐出归因）----
      const bool hit = touchFinite(name, bytes, write, ni, cap);
      // ---- 无限栈复用距离（容量无关）----
      double reuse = 0.0, dist = 0.0;
      bool cold = false;
      if (prof)
      {
        size_t at = inf.size();
        for (size_t k = 0; k < inf.size(); ++k)
          if (inf[k].first == name) { at = k; break; }
        if (at == inf.size())
        {
          cold = true;
          for (const auto & e : inf) reuse += e.second;   // 此前所有不同数据
          dist = kInf;
          inf.push_back({name, bytes});
        }
        else
        {
          for (size_t k = at + 1; k < inf.size(); ++k) reuse += inf[k].second;
          dist = reuse + bytes;
          const auto pv = inf[at];
          inf.erase(inf.begin() + static_cast<long>(at));
          inf.push_back(pv);
        }
        prof->touches.push_back({name, bytes, write, ni, cold, reuse, dist, hit});
        prof->total_bytes += bytes;
      }
    };

    for (const auto & r : n.reads) access(r.first, r.second, false);
    for (const auto & w : n.writes) access(w.first, w.second, true);
  }

  if (prof)
  {
    prof->spill_bytes = out.miss;
    for (size_t i = 0; i < out.blame.size(); ++i) prof->node_evict_bytes[i] = out.blame[i];
  }
  return out;
}

}  // namespace

L3Policy l3DefaultPolicy()
{
  const char * p = std::getenv("INFVINO_L3_POLICY");
  if (p)
  {
    const std::string s(p);
    if (s == "nru" || s == "plru" || s == "pLRU" || s == "PLRU") return L3Policy::NRU;
  }
  return L3Policy::LRU;
}

L3Result evaluateL3(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg)
{
  L3Result out;
  const SimStats s = simulateCore(nodes, cfg, nullptr);
  out.spill_bytes = s.miss;
  out.spill_ms = out.spill_bytes * cfg.spill_per_byte_ms;
  out.node_evict_ms.assign(nodes.size(), 0.0);
  for (size_t i = 0; i < s.blame.size(); ++i)
    out.node_evict_ms[i] = s.blame[i] * cfg.spill_per_byte_ms;
  if (!cfg.compute_prices) return out;
  out.node_price_ms.assign(nodes.size(), 0.0);
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    if (nodes[i].occ_bytes <= 0.0) continue;   // 占用为 0 → 无价格
    std::vector<L3Access> tmp = nodes;
    tmp[i].occ_bytes = 0.0;                    // 反事实：该节点不产生占用
    const double without = simulateCore(tmp, cfg, nullptr).miss;
    const double delta = out.spill_bytes - without;   // ≥0（占用只会增加 miss）
    out.node_price_ms[i] = (delta > 0.0 ? delta : 0.0) * cfg.spill_per_byte_ms;
  }
  return out;
}

L3StackProfile profileL3Stack(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg)
{
  L3StackProfile prof;
  (void)simulateCore(nodes, cfg, &prof);
  return prof;
}

double spillAtCapacity(const std::vector<L3Access> & nodes, double capacity_bytes, L3Policy policy)
{
  if (policy != L3Policy::LRU || !(capacity_bytes > 0.0))
  {
    L3ModelConfig cfg;
    cfg.l3_bytes = capacity_bytes;
    cfg.eff_cap_anchor = capacity_bytes;
    cfg.occ_cap_slope = 0.0;
    cfg.policy = policy;
    cfg.compute_prices = false;
    return evaluateL3(nodes, cfg).spill_bytes;
  }
  const L3StackProfile prof = profileL3Stack(nodes, L3ModelConfig{});
  double spill = 0.0;
  for (const auto & t : prof.touches)
    if (!t.write && (t.cold || t.stack_dist > capacity_bytes)) spill += t.bytes;
  return spill;
}

}  // namespace infvino
