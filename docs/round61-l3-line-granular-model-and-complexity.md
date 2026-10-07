# R61：L3 行粒度精确模型（公开 PRM）+ 成本函数复杂度分析

> 用户：`reuse^0.30` 太粗，最好**根据公开资料精确建模**（注意许可）；并想知道**严格 LRU
> 算整个成本函数的大 O**，以判断 cache-line 精确模拟扛不扛得住、栈距离技巧帮了多少。
>
> 结论先给：
> * 公开 PRM（Tiger Lake Vol.7 Memory Cache）**逐字给出**替换算法：**每 set 一个 N-bit 向量的
>   1b LRU**（Fill 选第一个 0 位 way 并翻转为 1；命中置 1；全 1 清空；"Allocate on fill"）。
>   几何：**bank = 480 KB = 120 way × 64 set × 64 B**。→ 精确模型是**组相联、行粒度 1b-NRU**，
>   不是幂律。已实现 `L3LineModel` + 实验台 `l3linesim`，复现出**阈值 ≈ 物理容量（7.86 MB）**；
>   `reuse^0.30` 的经验上升**主要来自测量摊销**（多遍掩盖首遍 miss），已降级为交叉验证。
> * 复杂度：**严格 LRU 单次＝ O(A)（哈希表+双向链表）**；当前实现是 **O(A·D)**（vector 线性
>   扫描+`erase`）；有限差分定价再乘节点数。**栈距离把「容量扫描」从 O(K·A) 降到
>   O(A log D) 一次 + O(1)/容量**。行粒度 A_L≈3×10⁶ 时：naive O(A·D) 不可行（~10¹³）；
>   O(A) 实现 ~10–100 ms/次，定价 ×节点 → 秒级，**可行**（实测 `l3linesim` 线性 ~252 ns/访问）。

---

## 1. 公开资料给出的**精确**替换模型

### 1.1 事实（引自公开 PRM，见 `third_party/intel-prm/NOTICE.md`）

* **算法（逐字要点）**：*"An N-way 1b LRU has an N-bit vector for each set. … As each Fill request
  arrives, the first bit in the vector with a 0 is selected, that way is used, and the bit is flipped.
  … when all N bits are 1 and there are no candidates, all bits are cleared. Any hits … will update
  the LRU vector to set the bit corresponding to the way. … If during a hit operation, a set becomes
  fully 'recent' … the vector is cleared."*
* **分配**：*"Allocate on fill"*（数据回来后分配，不是命中 miss 时）。
* **几何**：每个 bank `480 KB = 120 logical ways`（≤104 way 作 L3$，其余 URB），`64B` 行，
  `2 ways/sector`，8 KB 分段粒度 ⇒ **64 set/bank**。SKU bank 数见 Configurations 卷；本机 8 MB
  工作点取 **1024 set × 120 way × 64 B = 7.86 MB**。

### 1.2 实现与验证

`include/infvino/L3LineModel.hpp` / `src/L3LineModel.cpp`：`L3LineSim` 实现上述算法（`NRU1B`）
与严格 LRU 对照；每 set 用 `tag→way` 表（O(1) 命中）与逐 way 位向量。`src/tools/l3linesim.cpp`：

```
$ ./build/l3linesim --op probe --sets 1024 --ways 120 --hot-lines 16384 --reuse 1,4,16 --agg 4,6,8,12,16,25
 reuse  agg_MB   NRU1B_hit%  LRU_hit%
  1      4/6      100 / 100
  1      8/…/25    0 /   0        <- 阈值 ≈ 7.86 MB ≈ 物理容量
  4      8+        75 /  75       <- 后几遍补回（摊销）
 16      8+       93.8/ 93.8
```

→ 1b-NRU 的**首次逐出阈值 = 容量**（与 R59 的 `l3DefaultCapBytes()=8MB` 一致）；
`--op probe` 的 reuse 上升线与硬件 `--op l3retain` 的保留率形状一致，且**主要是每多一遍就在
时间上摊销首遍的 miss**（模型里 NRU1B 与 LRU 在纯流式+热点下几乎同形）。

`config/l3_calibration.json` 现以 **documented_model**（sets/ways/line/policy=nru1b）为权威，
把 `reuse^0.30` 明确标为 `empirical_fit_cross_check`（**不作为模型**）。

---

## 2. 严格 LRU 成本函数的大 O

记：`A` = 访问事件数（读+写，张量粒度≈节点数；行粒度=总字节/64）；`D` = 出现过的不同对象数；
`R` = 当前驻留对象数（`≤D`）；`P` = 参与占用的节点数（有限差分次数）；`K` = 要评估的容量档数。

### 2.1 单次 LRU 模拟

| 实现 | 每次访问 | 单次模拟 | 空间 |
|---|---|---|---|
| 当前 `simulateCore`（`std::vector` 线性查 + `erase` 搬移） | O(R) | **O(A·D)** | O(D) |
| 哈希表 + 侵入式双向链表（教科书最优） | O(1) 摊还 | **O(A)** | O(D) |
| 组相联 + 位掩码 first-zero（1b-NRU，way≤64） | O(1)（`ctz`） | **O(A)** | O(sets·ways) |

### 2.2 整个成本函数

```
Cost = Σ_i kernel_ms     (O(N)，查表，可加)
     + Σ_t reorder_ms    (O(#reorder)，查表/解析式)
     + spill             (唯一非可加项)
```

* `spill`（`evaluateL3`，`compute_prices=false`）：**一次**单次模拟 = **O(A·D)**（现状）/ O(A)（优化后）。
* `spill`（`compute_prices=true`，R55 有限差分）：**1 + P 次**模拟 ⇒ **O((P+1)·A·D)**（现状）/
  O((P+1)·A)（优化后）。`P ≤ N`。
* 每评估一个布局候选就重跑一次 ⇒ 布局搜索的总代价 = 候选数 × 上式。

**结论（现状）**：当前 `spill` 是 **O(A·D)**，加定价后 **O(P·A·D)**；在张量粒度（A、D ~ 10²）
这完全可忽略；一旦行粒度（A、D ~ 10⁶）就崩。

### 2.3 栈距离技巧的收益

`profileL3Stack` 一次遍历得到所有访问的复用距离直方图 `h_d`；之后 **任意容量 C** 的
`spill(C)=Σ_{d>C} h_d` 是前缀和 → **O(1)**。

| 任务 | 直接 LRU | 栈距离 |
|---|---|---|
| 单次固定容量 spill | O(A·D) / O(A) | O(A·D) / **O(A log D)** |
| K 档容量扫描 F(C) | O(K·A·D) / O(K·A) | **O(A log D + K)** |
| 容量分配/背包（需要 F(C)） | 每点一跑 | 直方图 O(A + bins) 一次 |
| 逐节点定价（有限差分） | O(P·A·D) | 干扰分解 `d_t=Δ_local+I_other`（每访问 O(1) 记账） |

即：**栈距离消掉了「容量维度 K」**，并让定价从「重跑 P 次模拟」变成「一次遍历里记账」——
这正是把 LRU 从「动态替换策略」转成「静态阈值 + 直方图」的价值。**但它不改变单次模拟里
「找/移驻留对象」的 O(A·D)**——那一项必须靠数据结构（哈希+链表 / 位掩码）来除。

---

## 3. cache-line 精确模拟能不能扛住？

### 3.1 规模估算（三张图之一，y8 量级）

* 激活总流量 `B ≈ 2×10⁸ B`（输入+各层特征图读写，量级）⇒ **A_L = B/64 ≈ 3×10⁶**。
* 唯一行 `D_L ≤ B/64`；常驻行 `C_L = 8 MB/64 = 1.31×10⁵`。

### 3.2 复杂度 × 常数

| 方案 | 时间 | 判定 |
|---|---|---|
| naive `vector` LRU O(A·D)：3×10⁶ × 3×10⁶ | ~10¹³ | **不可行** |
| 哈希+双向链表 O(A)：3×10⁶ 次访问 | ~10–30 ms/次 | **可行** |
| 1b-NRU + 位掩码 O(A) | ~10–30 ms/次 | **可行** |
| 现状 1b-NRU（O(ways) 扫描，实测 252 ns/访问）| 3×10⁶ × 252 ns ≈ **0.76 s/次** | 可行但慢 |
| 定价 ×P 节点 | ×~10² | 秒–分钟级（离线可接受） |
| 栈距离 O(A log D)：3×10⁶ × 22 | ~0.1–0.5 s **一次**，之后任意容量 O(1) | **可行** |

内存：`sets×ways`（1024×120）+ 每 set 的 `tag→way` 表；行粒度常驻仅 `1.3×10⁵` 项 → 几 MB。

### 3.3 实测（`l3linesim --op scale`，纯 host）

```
accesses=100000  -> 40.7 ms  (203 ns/access)
accesses=300000  -> 145.0 ms (242 ns/access)
accesses=1000000 -> 504.7 ms (252 ns/access)
accesses=3000000 -> 1513.2 ms(252 ns/access)   # 严格线性：O(A)
```

### 3.4 判定与优化清单

**能扛住**，但前提是把「单次模拟」从 O(A·D) 降到 O(A)，并把乘法因子钉住：

1. **驻留结构**：张量粒度也建议换成哈希 + 双向链表（现在是 `vector` 线性扫描 + `erase`
   搬移，D 小所以无所谓；行粒度必须换）。
2. **受害者选择**：1b-NRU 用 **每 set 一个 `ways≤64` 位掩码 + `ctz`**（现在是 O(ways)
   扫描）；`ways=120` 用两字或 sector 分组。
3. **定价**：用 `node_evict_ms` 的**一次遍历记账**替代 P 次重跑（R60 已做），行粒度下这是数量级差别。
4. **容量维度**：用栈距离 CDF（R60 已做），避免 K 次重跑。
5. **粒度选择**：行粒度仅在「需要 set/way 冲突/组相联语义」时才值得（它是非可加性的根源）；
   否则张量粒度 + `occupancyPressure` 已够。

---

## 4. 与 R60 的关系 / 诚实边界

* R60 把 spill 做成「复用/栈距离 + 逐出归因」；R61 补上**行粒度的精确替换机制**（公开文档），
  并把二者用**复杂度**串起来。R60 的 `L3Policy::NRU` 是**张量粒度**的等价近似；行粒度的权威
  模型是 `L3LineModel`（`NRU1B`）。
* 默认路径不变：`L3LineModel` / `l3linesim` 是**新增**、独立的 host 实验台；不进入默认 cost。
* 行粒度接入整网成本前仍需：把`L3Access` 展开为行访问（buffer→地址→set）、对齐真实
  `MOCS`/`L3ALLOCREG` 分区、并用 `kernel_bench` 复标几何（bank 数/SKU）。
* 许可：PRM 只作**事实**来源；`third_party/intel-prm/NOTICE.md` 逐字收录其 notice，
  不复制/不派生文档本身；实现为本仓库原创。

---

## 5. 复现

```bash
# 公开文档所述的 1b-NRU 行粒度模型（host，无需 GPU）
cmake --build build -j --target l3linesim
./build/l3linesim --op probe --sets 1024 --ways 120 --hot-lines 16384 \
    --reuse 1,4,16 --agg 4,6,8,12,16,25
./build/l3linesim --op scale --accesses 3000000 --working-mb 64 --reps 2

# 标定（documented_model 为权威；reuse^alpha 仅交叉验证）
python3 scripts/l3_calibrate.py --raw /tmp/l3retain.csv --out config/l3_calibration.json \
    --sets 1024 --ways 120

# 回归
./build/test_l3_model   # 新增 L3LineModel 用例
```
