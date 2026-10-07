# R60：L3 spill 的**精确（复用/栈距离）建模** + pLRU 等价工程模型 + reorder 可加性

> 用户决策：本轮一次性推进三件事——
> 1. **精确 spill**：把 spill 从「有限容量模拟」升级为**复用距离/栈距离**模型；
> 2. **pLRU 等价工程模型 + 自标定工具**（谨慎处理逆向与专利，写第三方声明）；
> 3. **reorder 精确建模**：实测「可加性差了点」，把它拆开、给出可加形式。
>
> 结论先给：
> * 新增 `profileL3Stack` / `spillAtCapacity`——**栈距离 CDF 与常数容量有限 LRU 逐位一致**
>   （单测保证），spill 由此可写成**静态阈值**；并给出**精确逐出归因价**（谁把谁挤出去）。
> * 新增替换策略等价模型 `L3Policy::NRU`（重用感知的 1-bit NRU + aging），默认仍严格 LRU，
>   **默认数值逐位不变**（y8 spill=1.1357，与 R59 相同）。自标定工具实测：保留 knee
>   ≈ **11.1 MB × reuse^0.30**（reuse=1 已到 1.3×物理 L3，reuse=16 到 2.3×）。
> * reorder 可加性用新微基准 `--op reorder_seq`（2×2：共享/独立 输入×输出）量化：
>   **全热 1.20×、全冷 1.77×**；拆为「单趟 kernel + **逐次 dispatch 间隔 ~12.4 µs** +
>   工作集/L3 项」。
>
> 本机 `websearch` 不可用（Google 不可达、Bing 被污染），故一切以**自研标定**为准——
> 这恰好也是本文件的来源原则。

---

## 0. 交付物总览

| 类别 | 落点 |
|---|---|
| 复用/栈距离画像 | `L3Model.hpp/cpp`：`L3StackTouch` / `L3StackProfile` / `profileL3Stack()` / `spillAtCapacity()` |
| 精确逐出归因价 | `L3Result::node_evict_ms`（模拟中记录「谁逐出了谁」） |
| 替换策略等价模型 | `L3Policy{LRU,NRU}` + `l3DefaultPolicy()`（`INFVINO_L3_POLICY=nru`） |
| reorder 可加成本 | `Tuning.hpp/cpp`：`reorderCostMs()` / `kReorderLaunchMs` / `kReorderStreamBwGbps` / `kReorderDispatchGapMs` |
| 微基准 | `kernel_bench --op reorder_seq`（2×2 可加性）、`--op l3retain`（保留曲线） |
| 自标定基础设施 | `scripts/l3_calibrate.py` → `config/l3_calibration.json` |
| 策略接入 | `PlanModel::resolveLayoutMinCut`（状态感知 reorder）、`layoutModelBreakdown`（可选 dispatch-gap 项） |
| 逆向/合规声明 | [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) §"L3 replacement model (equivalent, self-calibrated)" |
| 回归 | `test_l3_model` 15→**26**、`test_rulers` 87→**96**、`check` 6/6 |

---

## 1. 精确 spill：复用/栈距离模型

### 1.1 关系式与实现

对访问序列，在**无限 LRU 栈**里记每次访问的

```
d_t = Δ_local,t + I_other,t        (栈距离 = 自身字节 + 两次访问之间的干扰字节)
命中(C)  ⇔  d_t ≤ C               (stack-distance 定理；严格 LRU，常数容量)
```

`profileL3Stack()` 在同一次遍历里同时：
* 跑**有限调度**（占用压缩容量 → 逐出 → 读 miss/写），得到权威 `spill_bytes` 与 `hit`；
* 跑**无限栈**，得到与容量无关的 `reuse_bytes`(I) 与 `stack_dist`(d)。

`spillAtCapacity(nodes, C)` 由 CDF 直接给出任意常数容量 C 下的 spill。

**等价性（单测）**：对 60 步确定性随机序列、C∈{0.5,1,2,3.75,8} MB，
`spillAtCapacity(seq,C)` 与「容量固定 C 的 `evaluateL3`」**逐位相同**——这把动态替换
问题严格化为**静态阈值约束**，可嵌入容量分配/背包模型（见 R60 后续）。

### 1.2 精确逐出归因（replace R55 的过乐观有限差分）

模拟中，每当某张量被挤出栈，记下「执行到哪个节点把它挤出去的」；之后它再被读且 miss，
就把这次 miss 字节记到该节点头上 → `L3Result::node_evict_ms`。性质（单测）：

* 唯一 aggressor 承担全部干扰价；无害占用价 = 0；
* `Σ_i node_evict_ms ≤ spill_ms`（冷启动不计价）；
* 冷读（首次出现）不归因于任何节点。

这比「把节点占用归零」的有限差分更贴近**实际可削减量**——R55 §7.6 的 pernode 回归正来自
「把单节点占用归零的削减全额记在它头上」。归因价是确定性因果账，量纲 ms、≥0、有上界。

---

## 2. 硬件伪 LRU：**等价工程模型**（不是微架构复刻）

### 2.1 事实锚点（只用**公开**资料，非任何厂商内部文档）

* 本项目 `docs/xe-lp-isa.md` §8 记录：*Tiger Lake Open Source PRM* Vol.7 Memory Cache —
  「GFX L3，480 KB/bank、64 B line、**1b LRU**、按 way 分区 L3ALLOCREG」。这是**公开**的
  开源 PRM；「1b LRU」（每 way 一个替换位的伪 LRU）与 R55 的实测一致。
* R55 §3.5 双租户实测：1 MB 被反复重用的热点，**aggressor 到 ~8–12 MB 才被逐出**（2–3× 标称 L3）。
  严格 LRU 无法解释这个「MRU 抗污染」——它是**近期重用加权**的替换（含 evict-first/流式提示）。

> ⚠️ **免责边界**：我们**没有**、也不声称读到了任何 Intel 内部/未公开文档或寄存器规格。
> 下面的 `NRU` 是**行为等价**的工程模型，用于编译器成本函数；它**不是**对 tag RAM、way 选择
> 逻辑或替换位具体实现的复刻。参见 `THIRD_PARTY_NOTICES.md`。

### 2.2 模型：重用感知的 NRU（evict-first）

`L3Policy::NRU`：整条序列里**会被重复访问（reuse>1）**的张量以「受保护」态填入；
**只使用一次（流式）**的以 **evict-first** 填入；需逐出时牺牲未置位者，全部置位则 aging
清位后退化为 LRU。

在三个模型上（`INFVINO_L3_POLICY=nru` vs 默认 LRU），NRU 的 spill **不劣于** LRU：

| 模型 | LRU spill | NRU spill |
|---|---:|---:|
| yolov8n-pose | 1.1357 | **1.1137** |
| yolo11n-pose | 1.2508 | **1.2464** |
| mobilenetv3-small | 0.2317 | 0.2317 |

（当前三张图以 mandatory miss 为主，故差距小；策略的价值将由 §2.3 的标定与后续端到端 A/B 检验。
**默认仍是严格 LRU，数值逐位不变**。）

### 2.3 自标定工具（基础设施）：`l3retain` + `scripts/l3_calibrate.py`

微基准 `kernel_bench --op l3retain`：固定 1 MB 热点集合，把它**重用 `iters` 次**，前置一段
私有流式 aggressor（足迹 `R_A`），测热点再读的吞吐保留率。扫 `iters ∈ {1,4,16}`。
（锁频 `scripts/gpu_clocks.sh lock`。）

**实测保留率（1 MB 热点，retained = ref_ms/hot_ms）**：

| R_A | reuse=1 | reuse=4 | reuse=16 |
|---:|---:|---:|---:|
| 4.19 MB | 0.96–0.98 | 0.99–1.00 | 0.99–1.01 |
| 6.29 MB | 0.62–0.71 | 0.77–0.86 | 0.94–0.95 |
| 8.39 MB | 0.66–0.68 | 0.87–0.88 | 0.95–0.97 |
| 12.58 MB | 0.36–0.42 | 0.61–0.63 | 0.83–0.86 |
| 16.78 MB | 0.32 | 0.51 | 0.76 |
| 25.17 MB | 0.235 | 0.402 | 0.689 |

→ **同一足迹下，重用越多、保留越好**（R_A=12.58 MB：0.36→0.63→0.86）。
这正是「近期重用加权 / 伪-LRU」的签名，也解释了 R55 的 ~8–12 MB 阈值。

`scripts/l3_calibrate.py` 拟合保留 knee  `R_50(reuse)`：

```
R_50(reuse=1)  = 10.96 MB   (1.00x，本身已是 1.3x 物理 L3)
R_50(reuse=4)  = 17.34 MB   (1.58x)
R_50(reuse=16) = 25.17 MB   (2.30x)
fit: knee ≈ 11.12 MB · reuse^0.30
```

写出的 `config/l3_calibration.json` 带 **provenance** 字段：数据来源 = 本仓库自研微基准；
`vendor_documents_used=false`；模型为「行为等价的工程近似」。这就是「我们自行标定出来的」
基础设施证据链。

> ⚠️ **R61/R62 更正**：上述 `reuse^0.30` 只是**经验交叉验证**，**不是模型**。公开 PRM 给出的真实
> 机制是「每 set 一个 N-bit 向量的 1b LRU」（行粒度、组相联），已实现为 `L3LineModel`；
> reuse 上升**主要来自测量摊销**。且 R59 的 `l3DefaultCapBytes()=8MB` 是**CPU** L3——R62 实测
> GPU L3 = **3.75 MiB**（512 set × 120 way），已纠正默认容量。见
> [`round61-l3-line-granular-model-and-complexity.md`](round61-l3-line-granular-model-and-complexity.md)、
> [`round62-l3-fidelity-geometry-correction.md`](round62-l3-fidelity-geometry-correction.md)。

---

## 3. reorder 精确建模（可加性）

### 3.1 新微基准：`--op reorder_seq`（2×2）

N 趟相同 reorder 背靠背，GPU-spanning 事件计时（首个 START→末个 END，排除 host enqueue）：
* bit0：输入**共享**（热）或 N 份独立（冷）；
* bit1：输出**共享**或 N 份独立。

**锁频实测（Cin16 162²，N=16，bytes/pass=3.36 MB）**：

| 输入/输出 | per-op (ms) | N×single |
|---|---:|---:|
| 独立 / 独立（全冷） | 0.1020 | **1.767×** |
| 共享 / 独立 | 0.0987 | 1.355× |
| 独立 / 共享 | 0.0798 | 1.355× |
| 共享 / 共享（全热） | 0.0741 | **1.200×** |
| 单趟（基准） | 0.0617 | 1.00× |

### 3.2 分解：为什么「可加差了点」

```
per-op（序列） = 单趟 kernel（事件计时，全热 0.0617）
              + 逐次 dispatch 间隔（≈ 0.0124 ms = 12.4 µs，全热→1.20×）
              + 工作集/L3 项（输出独立 → +0.028 ms，→1.77×）
```

* **dispatch 间隔**：事件计时的单趟 `#reorder.ms` 不含「上一 kernel 结束→下一开始」的间隔；
  把 N 趟相加会少算 N×间隔。→ 新常数 `kReorderDispatchGapMs = 0.0124`。
* **工作集项**：输出越接近「各写各的、总量 >> L3」，单趟越接近 DRAM 写代价（~30 GB/s vs
  隔离热测 ~58 GB/s）。→ `reorderCostMs(read,write,input_resident)` 用 `input_resident`
  在 L3 档（145 GB/s）与流式档（60 GB/s）间切换。

### 3.3 接入与 A/B

* `resolveLayoutMinCut` 的结构性兜底改用 `reorderCostMs`，并用生产者在图中的位置估计
  输入是否仍在 L3（中间足迹 < 私有容量膝点）。因当前缓存相关签名都有实测 `#reorder`，
  **默认数值不变**；`INFVINO_NO_REORDER_STATE=1` 关闭。
* `layoutModelBreakdown`（离线评分/门）可显式补回 dispatch 间隔：
  `INFVINO_LAYOUT_REORDER_GAP=1` 时 `reorder += calls × 0.0124 ms`（默认关）。
  mb 例：`reorder 0.0561 → 0.1677`（12 次 dispatch）。

---

## 4. 数值/回归

```text
check 6/6 PASS
test_l3_model 15 → 26   (CDF≡有限LRU、复用分解、逐出归因、NRU、冷读不归因)
test_rulers   87 → 96   (reorderCostMs、policy 解析)
默认数值逐位不变：y8 model score total=11.3791（= R59）、spill=1.1357
```

---

## 5. 复现

```bash
# 离线单测（零 GPU）
cmake --build build -j --target check && ctest --test-dir build --output-on-failure

# 栈距离 CDF / 逐出归因 / NRU
./build/test_l3_model

# L3 策略 A/B（默认 LRU vs NRU）
INFVINO_LAYOUT_REPORT=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 2
INFVINO_L3_POLICY=nru INFVINO_LAYOUT_REPORT=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 2

# 自标定（锁频）
scripts/gpu_clocks.sh lock
./build/kernel_bench --op l3retain --hot-lines 16384 --hot-gws 8192 \
  --hot-sweep 1,4,16 --wi 1024,2048,4096,8192 --fp 4,8,16,24,32,48 > /tmp/l3retain.csv
python3 scripts/l3_calibrate.py --raw /tmp/l3retain.csv --out config/l3_calibration.json
scripts/gpu_clocks.sh unlock

# reorder 可加性（2×2）
scripts/gpu_clocks.sh lock
for m in 0 1 2 3; do ./build/kernel_bench --op reorder_seq --nseq 16 --seq-mode $m \
  --conv-shape 16,0,162,162 --iters 30; done
scripts/gpu_clocks.sh unlock

# reorder dispatch-gap 项 A/B
INFVINO_LAYOUT_REORDER_GAP=1 INFVINO_LAYOUT_REPORT=1 \
  ./build/kernel_run --plan models/mobilenetv3-small/model.plan --report --iters 2
```

---

## 6. 诚实边界 / 下一步

1. **精确 spill 的「精确」边界**：栈距离 CDF 与有限 LRU 的等价**只对严格 LRU + 常数容量**成立；
   本模型的容量随节点占用变化（`max(anchor, L3−slope·occ)`），故权威 spill 仍取有限模拟，
   栈距离用于**容量扫描/定价/干扰分解**这一层。逐出按**张量粒度**（非 cache-line 粒度），
   R47 §5 的老边界仍在。
2. **pLRU 等价模型**：`NRU` 是行为等价近似，参数（reuse 保护、aging）来自自标定；
   `reuse^0.30` 的**定量**接入有效容量（而非仅策略开关）留作下一步——需要端到端 A/B
   证明收益，且不得退化成「Σ 足迹 > 常数」硬阈值（R55 §3.5）。
3. **reorder**：dispatch 间隔（12.4 µs）与工作集项都需在**更多 shape/更多并发**下复标；
   当前仅在 Cin16 162² 上做 2×2，`kReorderDispatchGapMs` 是该 shape 的实测值。
4. **不可观测**：本机无 OA 计数器（R43），逐节点 spill/命中仍无法直接测——所有标定都是一阶差分。
   模型提高的是**选择依据**，最终裁决仍是整网实测门。
5. 默认仍 **LRU + 关闭 gap 项 + 关闭 L3 定价**：本轮的建模能力可开关、可 A/B，不改默认选择。
