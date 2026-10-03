# P0 设计：激活内存复用 + view/skip alias

> 目标：消除「每个 tensor 一块常驻 `cl_mem`」造成的 **90% 冗余分配**，并减少
> `copy_c`/`slice_axis`/`concat4` 的物化 launch。
> 依据：`docs/openvino-gap-analysis.md` §4.1/§4.2、§5 P0。
> 参考实现：OpenVINO `memory_pool.{hpp,cpp}` + `basic_memory_dependencies.cpp` +
> `primitive_inst::execute` 的 SKIP 路径。

---

## 0. 先量化（离线脚本，不需要 GPU）

`scripts/analyze_memreuse.py`（本设计配套）对三个 plan 做生存期分析：

| 模型 | 激活/中间 tensor | 朴素分配总量 | 峰值活跃 | 可复用比例 |
|---|---:|---:|---:|---:|
| yolov8n-pose | 150 | 70.2 MB | 6.7 MB | **90%** |
| yolo11n-pose | 200 | 79.2 MB | 7.3 MB | **91%** |
| mobilenetv3-small | 92 | 4.6 MB | 0.8 MB | **84%** |

> 注意：`cl_mem` 常驻 70 MB **不会立刻 OOM**（显存共享系统内存），但会把 Iris Xe 的
> **3.75 MB LLC 冲光**——每个节点读到的输入都已不在 L3，正是 R30 观察到的
> 「单节点 autotune 时热缓存 56 GB/s、部署时掉到 24 GB/s」的根因。

---

## 1. 分层设计

```
┌─ A. 生存期分析（纯 host，可单测）───────────────────────────┐
│  拓扑序 + first/last use → 每个 tensor 的 [birth, death)     │
│  参考: basic_memory_dependencies（只做直接依赖版，先不管 OOO） │
└─────────────────────────────────────────────────────────────┘
             ▼
┌─ B. 尺寸桶 memory pool（device）────────────────────────────┐
│  multimap<bytes, cl_mem>；复用条件：                         │
│    - 尺寸 ≥ 需求（lower_bound）                              │
│    - 「不得共享」冲突集不重叠（来自 A）                       │
│    - 对齐/元素类型一致（全 fp16，天然满足）                   │
│  参考: memory_pool::get_from_non_padded_pool                │
└─────────────────────────────────────────────────────────────┘
             ▼
┌─ C. view/skip alias（device，零 launch）───────────────────┐
│  C1. reshape/flatten（已有）                                 │
│  C2. copy_c 单消费者 → 消费者按 channel base offset 直读父张量 │
│  C3. slice_axis 单消费者 → 同理（axis 内 offset）             │
│  C4. concat4 单消费者 → 消费者直读多源（Route A 已做 conv1x1，  │
│      推广到访存型消费者）                                     │
└─────────────────────────────────────────────────────────────┘
```

**分步落地、每步独立可回退**（吸取 R30 §7.7 多消费者 alias FAIL 的教训）：
A+B 先做（不改数值），C 逐算子做且**每个消费者加独立数值用例**。

---

## 2. A：生存期分析

输入：`nodes_`（拓扑序，已是执行序）+ `T_`（张量集合）。
输出：`std::vector<Interval>`，每个中间张量一条 `[birth, death]`。

规则：
- **input**：`[0, N-1]`（整段存活）。
- **init（权重）**：**排除**，不参与复用（常驻）。
- **output**：`[first_use, N-1]`（必须活到最后被读回）。
- **中间**：`[producer_index, last_consumer_index]`。
- **view 别名张量**（reshape 输出）：与其父张量视为同一块（生存期取并集）。

冲突集：若 `I_a.death >= I_b.birth`（区间重叠），二者不得共享。
（先只用「直接依赖 + 输出不可复用」的保守版；OOO 版留到 P2。）

---

## 3. B：memory pool

```cpp
class ActPool {
public:
  // 申请一块「够大」的 buffer：先找可复用的（尺寸 >=、冲突集不相交），否则新建。
  cl_mem acquire(size_t bytes, const std::vector<int>& conflict);
  // 在一个节点的输入全部消费完后，释放该节点输出对 pool 的占用。
  void   release(cl_mem);
  size_t allocatedBytes() const;   // 统计：实际 clCreateBuffer 总量
  size_t reusedBytes() const;      // 统计：命中复用的字节数
};
```

**关键决策**：
- 复用是**按字节桶**而非精确匹配（学 OV 的 `lower_bound(bytes)`），避免大量小 tensor
  各占一块。
- 冲突集用**排序 vector + 双指针**判断重叠（OV 用 `memory_restricter<uint32_t>`）。
- 释放时机：在拓扑序的 `death` 点之后，buffer 归还 pool 可被后续 acquire 复用。
- **保留一个 `owned_` 的 fallback**：pool 无法满足时仍可 `rt_.alloc`，保证正确性。

---

## 4. C：view/skip alias（逐算子）

### C2. `copy_c`（channel slice）

`copy_c(x, HW, c0, cnt, dst_off) → y`，单消费者时让消费者直接读 `x`。
当前内核 `copy_c` 做的就是 `y[c*HW+r] = x[(c0+c)*HW+r]`。若消费者是
`conv3x3`/`conv1x1`/`ew_binary` 等，需要给 kernel 传一个 **channel base offset**：
- OV 的做法是 `tensor` 带 `base_offset`，kernel 的地址计算加 offset。
- infvino：给消费者 kernel 加可选 `-DXBASE=<off>`（编制期常量）或运行期参数。
  **只对确定支持 offset 的消费者做**，逐消费者加 `kernel_check` 用例。

### C3. `slice_axis`

同 C2，但 offset 沿 `axdim` 分段（`x[(o*axdim+start)*inner+rem]`）。
消费者可能是 `bmm`/`softmax`/`ew`，offset 语义不同 → **优先级低于 C2**。

### C4. `concat4` 单消费者

Route A 已把 `concat→conv1x1` 折进 gemm 的 B-staging。推广：让访存型消费者
（`ew_binary`/`conv`）按 4 个源的 `ca/cb/cc/cd` 重定向取址。**工程量大、风险高**，
排在 C2/C3 之后，且必须有独立数值用例。

---

## 5. 验证与度量

1. **数值**：`model_check` 三模型 PASS，且误差与基线**逐位一致**（复用不改语义）。
2. **内存**：`PlanModel` 暴露 `poolAllocatedBytes()` / `poolReusedBytes()`，
   期望 yolov8 从 70 MB 降到 ~7 MB（复用率 ≥ 85%）。
3. **性能**：`kernel_run --report --iters 5` 同会话 A/B，期望 busy 下降（L3 命中改善）
   主要落在 R30 标记为「贴 DRAM 墙」的算子（concat4/ew_binary/bmm）。
4. **回归**：`kernel_check`（算子级）、`numerical_check`（库后端）、`engine_check`。

> 安全：所有 GPU 任务按 `docs/benchmark_protocol.md` 分批、限时。

---

## 5.1 实测结果（R-P0，已落地）

同会话 A/B（base = R30c HEAD，new = 本改动）：

| 模型 | busy base→new | wall base→new（net only） | 分配 base→new | 复用率 |
|---|---|---|---|---|
| yolov8n-pose | 13.53 → **13.07**（−3.4%）| 17.58 → **16.66**（**−5.2%**）| 67.7 → 26.1 MB | 62% |
| yolo11n-pose | 14.59 → **13.87**（−4.9%）| 19.04 → **17.95**（**−5.8%**）| 76.8 → 28.1 MB | 63% |
| mobilenetv3-small | 2.84 → **2.72**（−4.3%）| — | 4.3 → 0.7 MB | 83% |

数值：三模型 `model_check` PASS，误差与 R30c 基线**逐位一致**
（y8 `5.278e-4/8.874e-3`、y11 `8.945e-4/1.773e-2`、mb `1.306e-2/1.086e-2`）。

### 新发现 1（重要）：内存池对**墙钟**的收益大于 **GPU busy**

wall 降幅（5.2–5.8%）明显大于 busy 降幅（3.4–4.9%）。说明墙钟里那段
「busy 之外」的开销**并非纯 per-dispatch launch 固定成本**，其中一部分随
**常驻 `cl_mem` 对象数量**增长（驱动侧的驻留/绑定/页表开销）。
→ 对 P2「减少 launch 开销」是个修正：**先把 buffer 数降下来**本身就是降 wall 的杠杆，
不一定非要融合 kernel。

### 新发现 2：首版数值 FAIL，根因是 reshape/flatten 视图的生存期

第一版只按节点读写算 `[birth, death]`，`mean_rel=0.75` FAIL。**deciding 根因**：
`run()` 里 `reshape/flatten` 把 `out.mem = in.mem`（零拷贝视图），视图与其源
共享存储，但生存期分析没合并 → 源 buffer 在视图仍被读取时被复用。
修法：并查集合并视图链，取区间并集。修正后逐位一致。
→ **任何 alias/复用机制都必须把「视图」当一等公民处理**（这正是 R30 §7.7 
多消费者 alias 失败的同源教训）。

### 仍未吃满（62% vs 理论 87%）

差距来自：(a) 视图并集延长了部分区间；(b) 静态「整块复用」无法把大 buffer 的
空闲区切给小张量（需要 **byte-offset sub-allocation**，即 OV 的 padded pool）。
后者是 P0 的下一步候选（收益递减，见下方结论）。

---

## 5.2 P0 第二步：连续 `copy_c` → sub-buffer alias（已落地）

Post-fusion 的 8 条 `copy_c` 都是 `Split_output_1`：父张量的**一段连续通道范围**
`[c0*HW, (c0+cnt)*HW)`（`c0=Cin/2`、`dst_off=0`）。这类切片**不需要 kernel 改动**：
用 `clCreateSubBuffer` 在父 buffer 的字节偏移 `c0*HW*2` 处建子 buffer，消费者照常读
一个 `cl_mem`。节点在 `run()` 里被跳过（不再 launch）。

条件（全部满足才别名）：`dst_off==0`；源张量连续；输出元素数 == `cnt*HW`；
源元素数 ≥ `(c0+cnt)*HW`；偏移 1024 对齐（Intel `CL_DEVICE_MEM_BASE_ADDR_ALIGN`）。

实测（同二进制，`INFVINO_NO_POOL=1` 作为基线对照）：

| 模型 | baseline busy | pool+alias busy | 变化 | 分配量 |
|---|---|---|---|---|
| yolov8n-pose | 13.57 | **12.98** | **−4.3%** | 60.5 → 24.0 MB |
| yolo11n-pose | 14.64 | **13.80** | **−5.7%** | 69.0 → 26.3 MB |
| mobilenetv3-small | 2.86 | **2.66** | **−7.1%** | 4.3 → 0.7 MB |

`copy_c` 从报告消失（−0.092 ms）；随机输入下三模型输出与基线**逐位一致**。

### 仍未做（低 ROI，明确记录）

- **`slice_axis`（0.040 ms）**：沿 `axdim` 带内 stride，不是连续段，别名需要消费者
  支持跨步 offset（要改多个 kernel）——**不划算**。
- **尾部 `concat4`（0.222 ms，6 条）**：`outer=17/64/51`、`inner=1` 的检测头拼接，
  可仿 Route A 让 `ew`/`reshape` 直接读多源，但收益小、风险高，暂缓。

> **P0 第二步小结**：post-fusion 后这些小算子的余额只剩 ~0.35 ms（~2.7% busy），
> 其中 `copy_c` 已用 alias 吃掉。**继续攻 view/alias 的边际收益很低**，
> 结构性收益应转向 P1（布局/精度分层）。

---

## 5.3 顺带确认：mobilenet 的 1.3e-2 与 P0 无关

对 mobilenet 做了 `pool` vs `INFVINO_NO_POOL` 的逐位对比：**输出完全相同**，
且两者 vs onnxruntime 都是 mean_abs=1.5944e-2。差别在 `nopool` 下 dump 中间张量
才能读到正确值——`--dump-tensor` 是 **run() 之后**读，池化缓冲里只剩最后一个写者，
**是 dump 工具的语义限制，不是数值 bug**。
mobilenet 的误差根因与判据改进见 `docs/benchmark.md` §1.2。


---

## 6. 与后续阶段的关系

- **P1 布局**：memory pool 是 blocked 布局传播的前提（reorder 的中间 buffer 也要复用）。
- **P2 调度**：冲突集的 OOO 版本（`oooq_memory_dependencies`）在 P2 引入。
- **C 类 alias**：本质是把 OV 的「SKIP 节点」补上，直接减少 launch 数。

---

## 7. 复现

```bash
# 生存期/复用率分析（不需要 GPU）
python3 scripts/analyze_memreuse.py models/*/model.plan

# 整网数值（需要 iGPU）
python3 scripts/model_check.py --model yolov8n-pose --repo $PWD --image infvino-dev:latest

# 整网 busy A/B
./build-ct/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 5
```

### 诊断开关（仅调试用）

| 环境变量 | 作用 |
|---|---|
| `INFVINO_NO_POOL=1` | 关闭池化，退回「每 tensor 一块 buffer」（对照用）|
| `INFVINO_POOL_LIMIT=k` | 只对**前 k 个**（声明序）张量池化，用于二分定位复用 bug |

`PlanModel::allocateActivations()` 在分配后会跑一次 `verify()`：若发现有共享同一
buffer 的两张量生存期重叠，会打印 `[pool][BUG]` —— 这是复用正确性的廉价守卫。
