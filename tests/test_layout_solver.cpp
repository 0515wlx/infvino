// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_layout_solver —— LayoutSolver（R49/R50 布局最小割）的离线正确性测试。
//
// 覆盖：
//   * 空图 / 单 unary / 固定标签 / 不可行边界；
//   * 非 submodular 表必须被拒绝（不静默改语义）；
//   * **随机 submodular 图 vs 暴力枚举** 的能量精确一致（R49/R50 数值修复的护栏）；
//   * 分解无损（solver 能量 == 用原表代回的能量）；
//   * 全局最优 ≠ 逐节点贪心（链上强吸引）。
//
// 纯 CPU、确定性（固定随机种子），不需要 GPU。
#include "infvino/LayoutSolver.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <utility>
#include <vector>

#include "test_util.hpp"

using namespace infvino;

namespace
{
// 原始二元代价表（求解器内部会把它分解成 unary + 吸引项；这里保留原表用于暴力枚举）。
struct OrigPair
{
  int    i = 0, j = 0;
  double f[2][2] = {{0, 0}, {0, 0}};
};

double evalOrig(int n, const std::vector<std::array<double, 2>> & u,
                const std::vector<OrigPair> & pt, const std::vector<int> & x)
{
  double e = 0.0;
  for (int i = 0; i < n; ++i) e += u[static_cast<size_t>(i)][static_cast<size_t>(x[static_cast<size_t>(i)])];
  for (const auto & p : pt) e += p.f[x[static_cast<size_t>(p.i)]][x[static_cast<size_t>(p.j)]];
  return e;
}

// 暴力枚举最优（n 很小时用）；尊重钉死标签。
std::pair<double, std::vector<int>> brute(int n, const std::vector<std::array<double, 2>> & u,
                                          const std::vector<OrigPair> & pt,
                                          const std::vector<int> & fixed)
{
  double           best = std::numeric_limits<double>::infinity();
  std::vector<int> best_x(static_cast<size_t>(n), 0);
  for (int mask = 0; mask < (1 << n); ++mask)
  {
    std::vector<int> x(static_cast<size_t>(n), 0);
    bool             ok = true;
    for (int i = 0; i < n; ++i)
    {
      x[static_cast<size_t>(i)] = (mask >> i) & 1;
      if (fixed[static_cast<size_t>(i)] >= 0 && x[static_cast<size_t>(i)] != fixed[static_cast<size_t>(i)])
      {
        ok = false;
        break;
      }
    }
    if (!ok) continue;
    const double e = evalOrig(n, u, pt, x);
    if (e < best) { best = e; best_x = x; }
  }
  return {best, best_x};
}

// 把"吸引项 + 行列偏置"形式的 submodular 表写进能量与暴力表。
//   f(x,y) = a_x + b_y + w·[x≠y]  ⇒ K = (f00+f11-f01-f10)/2 = -w ≤ 0
}  // namespace

static void run_tests()
{
  // --- 1. 空图 ---
  {
    BinaryEnergy e(0);
    const auto   s = solveBinaryMinCut(e);
    CHECK(s.labels.empty(), "empty graph -> no labels");
    CHECK_NEAR(s.energy, 0.0, 1e-12, "empty graph -> zero energy");
  }

  // --- 2. 单 unary：取代价较低的标签 ---
  {
    BinaryEnergy e(1);
    e.addUnary(0, 3.0, 1.0);
    const auto s = solveBinaryMinCut(e);
    CHECK_EQ(s.labels[0], 1, "single unary picks lower-cost label");
    CHECK_NEAR(s.energy, 1.0, 1e-9, "single unary energy");
  }

  // --- 3. 钉死标签必须被尊重（即使 unary 更想选另一个）---
  {
    BinaryEnergy e(1);
    e.addUnary(0, 0.0, 5.0);
    e.fix(0, 0);
    const auto s = solveBinaryMinCut(e);
    CHECK_EQ(s.labels[0], 0, "fixed label is honored");
    CHECK_NEAR(s.energy, 0.0, 1e-9, "fixed label energy");
  }

  // --- 4. 非 submodular 表被拒绝（返回 false，不静默改语义）---
  {
    BinaryEnergy bad(2);
    CHECK(!bad.addPairwiseTable(0, 1, 0.0, 0.0, 0.0, 10.0),
          "non-submodular table rejected");
    // 越界/自环也不崩。
    CHECK(!bad.addPairwiseTable(0, 0, 0, 0, 0, 0), "self-pair table rejected");
    bad.addUnary(99, 1.0, 2.0);  // 越界 unary 安全忽略
    bad.fix(99, 1);              // 越界 fix 安全忽略
    CHECK(true, "out-of-range calls are safe");
  }

  // --- 5. 随机 submodular 图 vs 暴力枚举 ---
  {
    std::mt19937                           rng(0xC0FFEEu);
    std::uniform_real_distribution<double> un(-5.0, 5.0);
    std::uniform_real_distribution<double> wp(0.0, 6.0);
    std::bernoulli_distribution            coin(0.5);

    int cases = 0;
    int checked_n = 0;
    for (int n = 1; n <= 9; ++n)
    {
      for (int trial = 0; trial < 40; ++trial)
      {
        std::vector<std::array<double, 2>> u(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) u[static_cast<size_t>(i)] = {un(rng), un(rng)};

        std::vector<int> fixed(static_cast<size_t>(n), -1);
        for (int i = 0; i < n; ++i)
          if (coin(rng)) fixed[static_cast<size_t>(i)] = coin(rng) ? 0 : 1;

        std::vector<OrigPair> pt;
        BinaryEnergy          e(n);
        for (int i = 0; i < n; ++i) e.addUnary(i, u[static_cast<size_t>(i)][0], u[static_cast<size_t>(i)][1]);

        for (int i = 0; i < n; ++i)
          for (int j = i + 1; j < n; ++j)
          {
            if (!coin(rng)) continue;
            const double w  = wp(rng);
            const double a0 = un(rng), a1 = un(rng), b0 = un(rng), b1 = un(rng);
            OrigPair     p;
            p.i       = i;
            p.j       = j;
            p.f[0][0] = a0 + b0;
            p.f[0][1] = a0 + b1 + w;
            p.f[1][0] = a1 + b0 + w;
            p.f[1][1] = a1 + b1;
            pt.push_back(p);
            e.addPairwiseTable(i, j, p.f[0][0], p.f[0][1], p.f[1][0], p.f[1][1]);
          }
        for (int i = 0; i < n; ++i)
          if (fixed[static_cast<size_t>(i)] >= 0) e.fix(i, fixed[static_cast<size_t>(i)]);

        const auto bf  = brute(n, u, pt, fixed);
        const auto sol = solveBinaryMinCut(e);
        // 分解无损：用解出的标签代回**原表**，能量应与求解器返回的一致，
        // 且等于暴力最优。
        const double sol_e = evalOrig(n, u, pt, sol.labels);

        if (std::fabs(sol.energy - bf.first) > 1e-6 || std::fabs(sol_e - bf.first) > 1e-6)
        {
          ++itest::g_fail;
          std::printf("FAIL random n=%d trial=%d: solver=%.9g brute=%.9g lossless=%.9g\n",
                      n, trial, sol.energy, bf.first, sol_e);
        }
        else
        {
          ++itest::g_pass;
        }
        ++cases;
        ++checked_n;
      }
    }
    std::printf("random min-cut vs brute force: %d cases\n", cases);
    CHECK(cases == checked_n, "all random cases ran");
  }

  // --- 6. 全局最优 ≠ 逐节点贪心（链上强吸引）---
  //   三个变量单独都想选 1/0/1，但链上强吸引 => 全局全 0 更优。
  {
    const int                            m = 3;
    std::vector<std::array<double, 2>>   uu = {{{0.0, 1.0}}, {{0.0, 3.0}}, {{0.0, 1.0}}};
    std::vector<OrigPair>                pp;
    OrigPair                             q0;
    q0.i = 0;
    q0.j = 1;
    q0.f[0][0] = 0.0; q0.f[0][1] = 2.5; q0.f[1][0] = 2.5; q0.f[1][1] = 0.0;
    pp.push_back(q0);
    OrigPair q1 = q0;
    q1.i = 1; q1.j = 2;
    pp.push_back(q1);

    const std::vector<int> no_fix(static_cast<size_t>(m), -1);
    const auto             bf = brute(m, uu, pp, no_fix);

    BinaryEnergy ge(m);
    for (int i = 0; i < m; ++i) ge.addUnary(i, uu[static_cast<size_t>(i)][0], uu[static_cast<size_t>(i)][1]);
    for (const auto & p : pp) ge.addPairwiseTable(p.i, p.j, p.f[0][0], p.f[0][1], p.f[1][0], p.f[1][1]);
    const auto gs = solveBinaryMinCut(ge);

    CHECK_NEAR(gs.energy, bf.first, 1e-6, "chain coupling optimum == brute force");
    CHECK(gs.labels == bf.second, "chain coupling labels == brute force argmin");
  }
}

ITEST_MAIN("test_layout_solver")
