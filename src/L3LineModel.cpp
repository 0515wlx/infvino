// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3LineModel 实现：行粒度 1b-NRU（公开文档算法）与严格 LRU。见 L3LineModel.hpp。
#include "infvino/L3LineModel.hpp"

#include <algorithm>
#include <limits>

namespace infvino
{

L3LineSim::L3LineSim(const L3LineConfig & cfg) : cfg_(cfg)
{
  if (cfg_.sets < 1) cfg_.sets = 1;
  if (cfg_.ways < 1) cfg_.ways = 1;
  sets_.assign(static_cast<size_t>(cfg_.sets), Set{});
  for (auto & s : sets_)
  {
    s.tag.assign(static_cast<size_t>(cfg_.ways), kInvalid);
    s.bit.assign(static_cast<size_t>(cfg_.ways), 0);
    s.stamp.assign(static_cast<size_t>(cfg_.ways), 0);
  }
}

void L3LineSim::reset()
{
  sets_.assign(static_cast<size_t>(cfg_.sets), Set{});
  for (auto & s : sets_)
  {
    s.tag.assign(static_cast<size_t>(cfg_.ways), kInvalid);
    s.bit.assign(static_cast<size_t>(cfg_.ways), 0);
    s.stamp.assign(static_cast<size_t>(cfg_.ways), 0);
  }
  st_ = L3LineStats{};
}

bool L3LineSim::bitSet(const Set & s, int w) const { return s.bit[static_cast<size_t>(w)] != 0; }
void L3LineSim::setBit(Set & s, int w) const { s.bit[static_cast<size_t>(w)] = 1; }
void L3LineSim::clearBits(Set & s)
{
  std::fill(s.bit.begin(), s.bit.end(), static_cast<uint8_t>(0));
  ++st_.agings;
}

// 公开文档规则：选**第一个 0 位**的 way（空 way 也是 0）。返回 -1 表示全 1（需 aging）。
int L3LineSim::firstZeroWay(Set & s, int & outFreed) const
{
  for (int w = 0; w < cfg_.ways; ++w)
    if (!bitSet(s, w)) { outFreed = (s.tag[static_cast<size_t>(w)] == kInvalid) ? 0 : 1; return w; }
  return -1;
}

bool L3LineSim::contains(uint64_t line_addr) const
{
  const int setIdx = static_cast<int>(line_addr % static_cast<uint64_t>(cfg_.sets));
  const uint64_t tag = line_addr / static_cast<uint64_t>(cfg_.sets);
  return sets_[static_cast<size_t>(setIdx)].tag2way.count(tag) > 0;
}

bool L3LineSim::access(uint64_t line_addr)
{
  ++st_.accesses;
  const int      setIdx = static_cast<int>(line_addr % static_cast<uint64_t>(cfg_.sets));
  const uint64_t tag = line_addr / static_cast<uint64_t>(cfg_.sets);
  Set & s = sets_[static_cast<size_t>(setIdx)];

  auto it = s.tag2way.find(tag);
  if (it != s.tag2way.end())
  {
    ++st_.hits;
    const int w = it->second;
    if (cfg_.policy == L3LinePolicy::NRU1B)
    {
      setBit(s, w);
      // 文档：「If during a hit a set becomes fully 'recent' ... the vector is cleared」。
      bool all1 = true;
      for (int k = 0; k < cfg_.ways; ++k)
        if (!bitSet(s, k)) { all1 = false; break; }
      if (all1) clearBits(s);
    }
    else
    {
      s.stamp[static_cast<size_t>(w)] = ++clock_;
    }
    return true;
  }

  // miss → 选一个 way
  int way = -1, freed = 0;
  if (cfg_.policy == L3LinePolicy::NRU1B)
  {
    way = firstZeroWay(s, freed);
    if (way < 0) { clearBits(s); way = 0; freed = 1; }   // 全 1 → 清空，取 way 0
  }
  else
  {
    for (int w = 0; w < cfg_.ways; ++w)
      if (s.tag[static_cast<size_t>(w)] == kInvalid) { way = w; freed = 0; break; }
    if (way < 0)
    {
      uint64_t best = std::numeric_limits<uint64_t>::max();
      for (int w = 0; w < cfg_.ways; ++w)
        if (s.stamp[static_cast<size_t>(w)] < best) { best = s.stamp[static_cast<size_t>(w)]; way = w; }
      freed = 1;
    }
  }

  if (freed && s.tag[static_cast<size_t>(way)] != kInvalid)
  {
    s.tag2way.erase(s.tag[static_cast<size_t>(way)]);
    ++st_.evictions;
  }
  s.tag[static_cast<size_t>(way)] = tag;
  s.tag2way[tag] = way;
  if (cfg_.policy == L3LinePolicy::NRU1B) setBit(s, way);
  else s.stamp[static_cast<size_t>(way)] = ++clock_;
  ++st_.fills;
  return false;
}

}  // namespace infvino
