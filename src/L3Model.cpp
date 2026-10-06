// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3Model 实现：全局 LRU 溢出模拟 + 有限差分逐节点价。见 L3Model.hpp。
#include "infvino/L3Model.hpp"

#include <algorithm>

namespace infvino
{
namespace
{

// 单次 LRU 模拟：返回总 miss 字节。resident 列表 MRU 在尾部。
double simulate(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg)
{
  std::vector<std::pair<std::string, double>> res;  // MRU 在尾
  double resBytes = 0.0, miss = 0.0, cap = cfg.l3_bytes;
  auto touch = [&](const std::string & name, double bytes, bool write) {
    for (size_t k = 0; k < res.size(); ++k)
      if (res[k].first == name)
      {
        const auto pv = res[k];
        res.erase(res.begin() + static_cast<long>(k));
        res.push_back(pv);
        return;
      }
    if (!write) miss += bytes;  // 读未命中 → 一次 DRAM 往返
    res.push_back({name, bytes});
    resBytes += bytes;
    while (resBytes > cap && !res.empty())
    {
      resBytes -= res.front().second;
      res.erase(res.begin());
    }
  };
  for (const auto & n : nodes)
  {
    // 占用压缩有效容量（R55：锚点 ~2MB）。
    cap = std::max(cfg.eff_cap_anchor, cfg.l3_bytes - cfg.occ_cap_slope * n.occ_bytes);
    while (resBytes > cap && !res.empty())
    {
      resBytes -= res.front().second;
      res.erase(res.begin());
    }
    for (const auto & r : n.reads) touch(r.first, r.second, false);
    for (const auto & w : n.writes) touch(w.first, w.second, true);
  }
  return miss;
}

}  // namespace

L3Result evaluateL3(const std::vector<L3Access> & nodes, const L3ModelConfig & cfg)
{
  L3Result out;
  out.spill_bytes = simulate(nodes, cfg);
  out.spill_ms = out.spill_bytes * cfg.spill_per_byte_ms;
  if (!cfg.compute_prices) return out;
  out.node_price_ms.assign(nodes.size(), 0.0);
  for (size_t i = 0; i < nodes.size(); ++i)
  {
    if (nodes[i].occ_bytes <= 0.0) continue;   // 占用为 0 → 无价格
    std::vector<L3Access> tmp = nodes;
    tmp[i].occ_bytes = 0.0;                    // 反事实：该节点不产生占用
    const double without = simulate(tmp, cfg);
    const double delta = out.spill_bytes - without;   // ≥0（占用只会增加 miss）
    out.node_price_ms[i] = (delta > 0.0 ? delta : 0.0) * cfg.spill_per_byte_ms;
  }
  return out;
}

}  // namespace infvino
