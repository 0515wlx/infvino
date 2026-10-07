# R65：L3 级隔离尝试与几何结论（地址→bank/set 能逆到哪一步）

> 用户选 (a)：设计「L3 级隔离」的 bank/set 探针（让工作集同时压满 L3 和 LLC，只留 L3 行为）。
>
> 结论先给：
> * **硬件计数器不可用**（R43 结论复现：`gpu_metrics --sample … ` → **`Activate failed`**，
>   内核缺 `CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS`）→ 无法直接读 L3 命中/逐出。
> * **计时探针无法隔离私有 L3**：LLC 在其后，任何填满 LLC 的流必然先填满 L3（层次性），
>   所以 L3 私有行为**总是经 LLC 观测**；流式膝点、冲突起点都是「L3+LLC」合成量。
> * **能逆到的**：地址索引**周期 = 512 行 = 2¹⁵ B**（锐、可复现）；几何 = 公开 PRM 的
>   **512 set × 120 way × 64 B = 3.75 MiB**（周期 512 与之一致）。**完整 XOR 哈希不可逆**。
> * 对成本模型：**不需要**完整哈希——周期 + 容量已足够；有效容量用两级和（≈12 MB）。
>
> **接受近似（正式）**：地址→(bank,set) 的完整 XOR 哈希**在本机不可解**（计数器不可用 + 层次性
> 阻断隔离）。我们**接受**「周期 512 行 + 512 set × 120 way」这一近似，理由见 §0；整网 A/B
> 见 §5.5（当前三模型**无差异**，因近似只改离线评分、不改选择）。

---

## 0. 为什么接受这个近似

1. **可逆到的那一层已被 PRM 完全确定**：周期 512 行（实测锐利）与 PRM 的
   512 set × 120 way = 3.75 MiB 一致；容量由「周期 × way × 64」唯一给出。剩下的只是
   **组内 way 选择顺序/XOR 位混合**，对**容量/周期**这两个成本模型真正用到的量没有影响。
2. **工具封顶**：`gpu_metrics`（OA）在本机 `Activate failed`（内核缺 `LOW_LEVEL_TRACEPOINTS`）；
   而 L3 在 LLC 之后，任何填满 LLC 的流必然先填满 L3 → 计时探针**原理上**无法隔离私有 L3。
3. **边际收益低**：即便逆出 XOR，`L3LineModel` 也只是把 `line mod 512` 换成 `hash(line)`，
   改变的是**逐出曲线形状**（已知近似），不改变容量/周期，也不改变复杂度结论。
4. **可验证性**：近似只影响**离线评分**；在拿到外部稳态 A/B 证据前，不应让一个不可验证的
   细节去改默认选择（项目纪律：模型改动默认关或只改评分）。

---

## 1. 尝试的隔离探针（与为何不成立）

设想：先用 filler 填满 LLC，再测 probe 的冲突——使 L3 miss 落到 DRAM 而非 LLC。

**为什么做不到**：L3 是 LLC 的**上游**。要填满 LLC（8 MB），必然先把 L3（3.75 MB）填满；
probe 一起手就 miss L3。反过来，若 filler < L3，则 LLC 仍有空间容纳 probe 的冲突行 → miss
被 LLC 吸收。**不存在**一个 filler 尺寸能「只压 LLC 不压 L3」。因此 L3-private 的
ways/哈希无法用驱逐/冲突计时分离（除非有计数器，或能标记 no-LLC 的 MOCS——不可达）。

## 2. 实测（本轮补测，锁频）

### 2.1 流式膝点是**渐变的**（不是锐利 3.75 MB）

`l3conflict --line-stride 1 --wi 1024 --passes 1024`：

| 工作集 | 2.10 | 3.15 | 3.67 | 3.93 | 4.19 | 4.72 | 5.24 | 6.29 MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 155.5 | 142.0 | 137.6 | 128.8 | 137.2 | 134.7 | 126.6 | 119.1 |

→ 从 ~2 MB 起**平滑下降**（到 12–16 MB 才贴底）；**没有** 3.75 MB 的锐利断崖。
这正是「L3（3.75）→ LLC（8）」两级 + 连续流自身逐出的合成形状。

### 2.2 冲突周期仍是**锐利 512 行**（与 R63 相同）

stride 511/513 干净、512 骤增（§R63 §2.1）。这是**最鲁棒**的映射特征。

### 2.3 stride-512 的起点 ≈ 512–640 行

`l3conflict --line-stride 512` 扫 nlines：≤512 平、≥640 起。与「冲突 set 在 L3+LLC 里的
**合计相连度**」一致（并不是纯 L3 的 ways）。

## 3. 几何结论（能逆到的与不能的）

| 量 | 值 | 来源 | 确定性 |
|---|---|---|---|
| 地址索引**周期** | **512 行 = 2¹⁵ B** | §2.2 步进冲突（锐） | **高** |
| GPU 私有 L3 几何 | **512 set × 120 way × 64 B = 3.75 MiB** | 公开 PRM（与周期一致） | **高** |
| L3 **ways**（纯） | 120 | PRM | 高（计时无法单独复核） |
| 完整 bank/set **XOR 哈希** | 未知 | — | **不可逆**（计数器/物理地址缺失） |
| 跨算子**有效容量** | ≈ L3+LLC ≈ 12 MB | R63 别名 ~12 MB | 中–高 |
| LLC（共享） | 8 MiB | sysfs index3 | 高 |

**结论**：地址→(bank,set) 只能逆到「**周期 512**」这一层；周期 + 容量已由公开 PRM 落到具体
的 **512×120**。更细的 XOR 位混合**用本机工具不可解**——需要 OA 计数器（不可用）或物理地址
可控 + MOCS no-LLC（不可达）。

## 4. 对成本模型/复杂度的影响

* **不需要**完整哈希：`L3LineModel` 用 `sets=512, ways=120`（`bucket = line mod 512`）
  即可复现**周期**与**容量**；逐出曲线**形状**的偏差来自未建模的 XOR/LLC，属已知近似。
* 有效容量用**两级和**（`l3WarmCapBytes = l3Physical + l3Llc`）——R64 已接入，评分改善
  且选择不变。
* 复杂度不变：单次 **O(A)**；行粒度内存 512×120 ≈1–3 MB；A_L≈3×10⁶ → ~10–100 ms/次。**可行**。

## 5. 建议

停止继续逆哈希（边际收益低、工具受限）；改为：
1. 把两级容量/几何**接进 `L3Model` 主问题**并做整网锁频交错 A/B；
2. 若要更进一步，只能等 OA 计数器（需换内核/权限）或 GDB/硬件调试通道。

### 5.5 整网 A/B（锁频、交错、交替臂序、6 rep、`--iters 15`）

两条臂：`warm12.3`（默认，L3+LLC ≈12.3 MB）vs `geom3.75`（`INFVINO_L3_GEOM=1`，仅 GPU 私有）。
度量 = `total kernel time`（busy，ms），每臂 6 rep 取 median：

| 模型 | warm12.3 median | geom3.75 median | Δ median |
|---|---:|---:|---:|
| yolov8n-pose | 10.534 | 10.558 | **−0.23%** |
| yolo11n-pose | 11.406 | 11.366 | **+0.35%** |
| mobilenetv3-small | 1.388 | 1.389 | **−0.07%** |

→ **三模型均在噪声内**（符号不一致，|Δ|<0.4%）。与 `model_check` 一致：容量常数只改**离线评分**，
三模型的**选择（kernel/layout/fsv16 0/9/23）逐位未变** → 整网**无差异**。

**要看到整网效果**，必须让近似**改变选择**：① 打开 L3 定价（`INFVINO_LAYOUT_L3=1`）并让分量门
因容量翻转；② 换容量跨过门阈值的模型/候选集；③ 把两级容量接进**内层 roofline/定价**（R55 §6.1
的 `effectiveBwGbps` 尚未接内核级）。在此之前，容量常数是**「评分正确性」修正，不是「性能收益」**。

## 6. 复现

```bash
scripts/gpu_clocks.sh lock
# 1) 计数器不可用（预期 Activate failed）
./build/gpu_metrics --sample L3_1 2000
# 2) 流式膝点（渐变）
for nl in 32768 49152 57344 61440 65536 73728 81920 98304; do
  ./build/kernel_bench --op l3conflict --nlines $nl --wi 1024 --passes 1024 --line-stride 1; done
# 3) 锐利周期 512
./build/kernel_bench --op l3conflict --nlines 1024 --wi 1024 --passes 64 \
  --line-stride 510,511,512,513,514
scripts/gpu_clocks.sh unlock

# 4) 整网 A/B（锁频、交错、6 rep）
scripts/gpu_clocks.sh lock
for rep in 1 2 3 4 5 6; do
  for m in yolov8n-pose yolo11n-pose mobilenetv3-small; do
    ./build/kernel_run --plan models/$m/model.plan --report --iters 15 | grep "total kernel time"
    INFVINO_L3_GEOM=1 ./build/kernel_run --plan models/$m/model.plan --report --iters 15 | grep "total kernel time"
  done
done
scripts/gpu_clocks.sh unlock
```
