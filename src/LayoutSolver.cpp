// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// LayoutSolver 实现：Dinic 最大流求 s-t 最小割 + 二元代价表分解。
#include "infvino/LayoutSolver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>

namespace infvino
{
namespace
{
constexpr double kEps = 1e-9;
// 用于表示「不可行」的软无穷：足够大以主导任何真实代价，又不会在算术里溢出。
constexpr double kBig = 1e18;

// 标准 Dinic 最大流（double 容量）。
struct Dinic
{
  struct Edge
  {
    int    to;
    double cap;
    int    rev;
  };
  std::vector<std::vector<Edge>> g;
  std::vector<int>               level, iter;

  explicit Dinic(int n) : g(n), level(n), iter(n) {}

  void add(int a, int b, double c)
  {
    if (c < 0.0) c = 0.0;
    g[a].push_back({b, c, static_cast<int>(g[b].size())});
    g[b].push_back({a, 0.0, static_cast<int>(g[a].size()) - 1});
  }

  bool bfs(int s, int t)
  {
    std::fill(level.begin(), level.end(), -1);
    std::queue<int> q;
    level[s] = 0;
    q.push(s);
    while (!q.empty())
    {
      const int v = q.front();
      q.pop();
      for (const auto & e : g[v])
        if (e.cap > kEps && level[e.to] < 0)
        {
          level[e.to] = level[v] + 1;
          q.push(e.to);
        }
    }
    return level[t] >= 0;
  }

  double dfs(int v, int t, double f)
  {
    if (v == t) return f;
    for (int & i = iter[v]; i < static_cast<int>(g[v].size()); ++i)
    {
      Edge & e = g[v][i];
      if (e.cap > kEps && level[v] < level[e.to])
      {
        const double d = dfs(e.to, t, std::min(f, e.cap));
        if (d > kEps)
        {
          e.cap -= d;
          g[e.to][e.rev].cap += d;
          return d;
        }
      }
    }
    return 0.0;
  }

  double maxflow(int s, int t)
  {
    double flow = 0.0;
    while (bfs(s, t))
    {
      std::fill(iter.begin(), iter.end(), 0);
      double f;
      while ((f = dfs(s, t, 1e300)) > kEps) flow += f;
    }
    return flow;
  }
};
}  // namespace

BinaryEnergy::BinaryEnergy(int n_)
  : n(n_), unary(static_cast<size_t>(n_ < 0 ? 0 : n_), {0.0, 0.0}),
    fixed(static_cast<size_t>(n_ < 0 ? 0 : n_), {-1, -1})
{
}

void BinaryEnergy::addUnary(int i, double c0, double c1)
{
  if (i < 0 || i >= n) return;
  unary[static_cast<size_t>(i)][0] += c0;
  unary[static_cast<size_t>(i)][1] += c1;
}

void BinaryEnergy::addPairwise(int i, int j, double w)
{
  if (i < 0 || j < 0 || i == j || i >= n || j >= n) return;
  if (w <= kEps) return;
  pairs.push_back({i, j, w});
}

bool BinaryEnergy::addPairwiseTable(int i, int j, double f00, double f01, double f10, double f11)
{
  if (i < 0 || j < 0 || i == j || i >= n || j >= n) return false;
  const double K = (f00 + f11 - f01 - f10) / 2.0;
  if (K > kEps) return false;  // 非 submodular：调用方回退，不静默改语义
  constant += f00;
  addUnary(i, 0.0, f10 - f00 + K);
  addUnary(j, 0.0, f01 - f00 + K);
  if (-K > kEps) addPairwise(i, j, -K);
  return true;
}

void BinaryEnergy::fix(int i, int label)
{
  if (i < 0 || i >= n) return;
  fixed[static_cast<size_t>(i)] = {label == 0 ? 0 : 1, -1};
}

bool BinaryEnergy::feasible() const
{
  return n >= 0;
}

BinarySolution solveBinaryMinCut(const BinaryEnergy & e)
{
  BinarySolution sol;
  sol.labels.assign(static_cast<size_t>(e.n < 0 ? 0 : e.n), 0);
  if (e.n <= 0)
  {
    sol.energy = e.constant;
    return sol;
  }

  const int S = e.n;
  const int T = e.n + 1;
  Dinic din(2 * e.n + 2);

  // unary → source/sink 容量。先处理固定与不可行。
  double constant = e.constant;
  for (int i = 0; i < e.n; ++i)
  {
    double d0 = e.unary[static_cast<size_t>(i)][0];
    double d1 = e.unary[static_cast<size_t>(i)][1];
    if (!std::isfinite(d0)) d0 = kBig;
    if (!std::isfinite(d1)) d1 = kBig;
    if (e.fixed[static_cast<size_t>(i)][0] >= 0)
    {
      const int l = e.fixed[static_cast<size_t>(i)][0];
      if (l == 0) { d0 = 0.0; d1 = kBig; }
      else        { d0 = kBig; d1 = 0.0; }
    }
    if (d0 >= kBig && d1 >= kBig)
    {
      sol.energy = BinaryEnergy::kInf;  // 不可行
      return sol;
    }
    // 平移最小 unary 到常数，保证容量非负（不改 argmin）。
    const double m = std::min(d0, d1);
    constant += m;
    d0 -= m;
    d1 -= m;
    // 约定：变量在 source 侧=标签 0，sink 侧=标签 1。
    // 取标签 1 → 割 s->i（容量 d1）；取标签 0 → 割 i->t（容量 d0）。
    din.add(S, i, d1);
    din.add(i, T, d0);
  }

  // 吸引项：无向，两向各 w（取不同标签时恰好一个方向跨割）。
  for (const auto & p : e.pairs)
  {
    din.add(p.i, p.j, p.w);
    din.add(p.j, p.i, p.w);
  }

  const double cut = din.maxflow(S, T);

  // 残量图上从 S 可达 = source 侧 = 标签 0。
  std::vector<char> reach(static_cast<size_t>(e.n) + 2, 0);
  std::queue<int>   q;
  reach[static_cast<size_t>(S)] = 1;
  q.push(S);
  while (!q.empty())
  {
    const int v = q.front();
    q.pop();
    for (const auto & ed : din.g[static_cast<size_t>(v)])
      if (ed.cap > kEps && !reach[static_cast<size_t>(ed.to)])
      {
        reach[static_cast<size_t>(ed.to)] = 1;
        q.push(ed.to);
      }
  }
  for (int i = 0; i < e.n; ++i)
    sol.labels[static_cast<size_t>(i)] = reach[static_cast<size_t>(i)] ? 0 : 1;

  sol.energy  = constant + cut;
  sol.optimal = true;
  return sol;
}

}  // namespace infvino
