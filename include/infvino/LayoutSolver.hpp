// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// LayoutSolver —— 布局标注的**精确最小割**求解器（R49 统一布局试点）。
//
// 把「每个激活张量用 NCHW 还是 FSV16」建模成一个二元标注问题：
//
//     E(x) = Σ_i unary[i][x_i] + Σ_{(i,j)} w_ij·[x_i ≠ x_j]         (x_i ∈ {0,1})
//
// 成对项是**吸引式**（w ≥ 0，即「同布局免费、异布局付 reorder」）→ submodular →
// 可用 s-t 最小割**精确**求解（本文件用 Dinic 最大流）。
//
// 一个算子节点自身的代价同时依赖「输入张量布局」和「输出张量布局」；这种二元代价表
// 由 `addPairwiseTable` 自动分解成 unary + 吸引项（若违反 submodular 则拒绝）。
//
// 定位（与项目分层一致）：
//   * kernel 层：ops/EU/cyc 是标尺，这里不涉及；
//   * 算子族/系统层：layout 是一等公民，本求解器给出**确定性、零 GPU、可离线自检**的
//     全局布局标注，替代「逐节点贪心 + 事后规划」。
//
// 设计见 docs/round49-layout-mincut-pilot.md。
#ifndef INFVINO__LAYOUT_SOLVER_HPP_
#define INFVINO__LAYOUT_SOLVER_HPP_

#include <array>
#include <limits>
#include <vector>

namespace infvino
{

/**
 * @brief 二元布局能量的容器（变量 0=NCHW，1=FSV16）。
 *
 * 约定：unary[i][l] 是变量 i 取标签 l 的代价；pairs 是吸引式成对项（取不同标签付 w）。
 * 能量中还有一个只用于报告的常数项（分解产生）。
 */
struct BinaryEnergy
{
  static constexpr double kInf = std::numeric_limits<double>::infinity();

  struct Pair
  {
    int    i = 0;
    int    j = 0;
    double w = 0.0;  // ≥ 0
  };

  int                                n = 0;    ///< 变量数
  std::vector<std::array<double, 2>> unary;   ///< unary[i][label]
  std::vector<std::array<int, 2>>    fixed;   ///< {-1,-1}=自由；{l,-1}=固定为 l
  std::vector<Pair>                  pairs;   ///< 吸引项（去重由调用方负责）
  double                             constant = 0.0;

  explicit BinaryEnergy(int n_ = 0);

  /** @brief 累加变量 i 的 unary（可多次调用）。*/
  void addUnary(int i, double c0, double c1);
  /** @brief 吸引项：i 与 j 取不同标签的代价 w（w<0 视为 0 并忽略）。*/
  void addPairwise(int i, int j, double w);
  /**
   * @brief 把一般二元代价表 f(x,y)=f[xx][yy] 分解成 unary + 吸引项。
   *
   * 分解（x,y∈{0,1}）：
   *   K   = (f00 + f11 - f01 - f10) / 2
   *   E   = f00 + (f10 - f00 + K)·[x=1] + (f01 - f00 + K)·[y=1] + (-K)·[x≠y]
   * submodular（K ≤ 0）时成立；否则返回 false（调用方应回退，不要静默改语义）。
   */
  bool addPairwiseTable(int i, int j, double f00, double f01, double f10, double f11);
  /**
   * @brief `addPairwiseTable` 的**永不失败**版本（R56 回退局部化）。
   *
   * 非 submodular（K>0）时把耦合项 clamp 到 0（丢掉落 repulsive 项），得到一个合法的
   * **松弛**能量：解仍是确定的最优解，但不再让「一个非 submodular 表」作废整份布局提案。
   * 仅用于「求解器不应整体回退」的场景；需要严格语义时仍用 `addPairwiseTable`。
   */
  void addPairwiseTableRelaxed(int i, int j, double f00, double f01, double f10, double f11);
  /** @brief 钉死变量 i 的标签（0/1）。*/
  void fix(int i, int label);
  /** @brief 是否存在「两个标签都被钉死/都不可行」的矛盾变量。*/
  bool feasible() const;
};

struct BinarySolution
{
  std::vector<int> labels;        ///< 每个变量的标签 0/1
  double           energy = 0.0;  ///< 最小能量（含 constant）
  bool             optimal = true;
};

/** @brief 精确求解 s-t 最小割；返回能量最小的一组标签。*/
BinarySolution solveBinaryMinCut(const BinaryEnergy & e);

}  // namespace infvino

#endif  // INFVINO__LAYOUT_SOLVER_HPP_
