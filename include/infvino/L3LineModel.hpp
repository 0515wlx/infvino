// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// L3LineModel —— **行粒度**的 L3 替换模拟（严格 LRU vs 公开文档所述的 1b LRU）。
//
// 这是本仓库**自行编写**的实现，只依据 Intel 公开的 *Iris Xe / UHD Graphics Open Source
// PRM*（Tiger Lake, Vol.7 Memory Cache，Doc Ref IHD-OS-TGL-Vol 7-12.21）中**已公开**的
// 算法描述：每 set 一个 N-bit 向量；Fill 选第一个 0 位 way、翻转为 1；命中把对应 way 置 1；
// 全 1 时清空；分配策略 "Allocate on fill"。文档许可**允许基于该文档创建软件实现**。
//
// 许可合规见 third_party/intel-prm/NOTICE.md：本文件**不**包含该文档的任何文本/图/表，
// 只把公开的算法事实编码为独立实现。
//
// 用途：
//   * 精确复现替换行为（用于成本模型 / 标定），替代此前粗糙的经验幂律；
//   * 作为**复杂度**与**行粒度可行性**的实验台（见 src/tools/l3linesim.cpp）。
#ifndef INFVINO__L3_LINE_MODEL_HPP_
#define INFVINO__L3_LINE_MODEL_HPP_

#include <cstdint>
#include <unordered_map>
#include <vector>

namespace infvino
{

/** @brief 行粒度的替换策略。 */
enum class L3LinePolicy
{
  NRU1B = 0,  ///< 公开文档所述的 1b LRU（N-bit 向量 / first-zero-way / 全 1 清空）
  LRU = 1,    ///< 严格 LRU（对照基线）
};

/** @brief 行缓存的几何与策略（默认 = TGL GT2 实测工作点，R62：bank 480KiB=120 way×64 set×64B，
 *  8 bank → 512 set = 3.75 MiB）。`ways`、`sets` 由 SKU 的 Configurations 文档决定，可配置。*/
struct L3LineConfig
{
  int sets = 512;               ///< 组数（= bank 数 × 64；R62 实测 8 bank）
  int ways = 120;               ///< 每组相连度（way 数）
  int line_bytes = 64;          ///< 行大小
  L3LinePolicy policy = L3LinePolicy::NRU1B;
};

struct L3LineStats
{
  uint64_t accesses = 0, hits = 0, fills = 0, evictions = 0, agings = 0;
  double hitRate() const { return accesses ? static_cast<double>(hits) / accesses : 0.0; }
};

/**
 * @brief 一个行缓存的模拟器。`access(line_addr)` 返回是否命中（line_addr 为 64B 行号）。
 *
 * 复杂度：每次访问 O(1) 期望（每 set 一个 tag→way 的 hash 表 + bit 向量；first-zero 用
 * 位掩码 ctz）。内存 O(sets × ways)。`reset()` 清空。
 */
class L3LineSim
{
public:
  explicit L3LineSim(const L3LineConfig & cfg);

  bool access(uint64_t line_addr);
  /** @brief 非破坏性查询：该行是否驻留（仅用于测试/诊断）。 */
  bool contains(uint64_t line_addr) const;
  const L3LineStats & stats() const { return st_; }
  void reset();

private:
  struct Set
  {
    std::unordered_map<uint64_t, int> tag2way;  ///< tag -> way
    std::vector<uint64_t>             tag;      ///< way -> tag（无效 = kInvalid）
    std::vector<uint8_t>              bit;      ///< 1b-NRU：way -> 「最近」位
    std::vector<uint64_t>             stamp;    ///< 严格 LRU：way -> 时间戳
  };
  int     firstZeroWay(Set & s, int & outFreed) const;
  void    clearBits(Set & s);
  bool    bitSet(const Set & s, int w) const;
  void    setBit(Set & s, int w) const;

  L3LineConfig        cfg_;
  std::vector<Set>    sets_;
  L3LineStats         st_;
  uint64_t            clock_ = 0;   ///< LRU 用的单调时间戳
  static constexpr uint64_t kInvalid = ~0ull;
};

}  // namespace infvino

#endif  // INFVINO__L3_LINE_MODEL_HPP_
