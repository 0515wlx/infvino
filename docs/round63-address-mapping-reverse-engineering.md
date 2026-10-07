# R63：地址→bank/set 逆向（进度）+「物理容量 ≠ 跨算子热容量」的实测澄清

> 用户：逆出地址→bank/set 是必须的；继续。
> 结论先给：**索引周期（512 行 = 2¹⁵ B）已钉死**；但**完整 bank/set XOR 哈希未逆出**。
> 更重要的是澄清了**内存层次**：**GPU 私有 L3 Data Cache = 3.75 MiB**，其后是**与 CPU 共享的
> LLC = 8 MiB**（本机 4 核；sysfs index3）→ **跨算子热重用有效容量 ≈ 3.75+8 ≈ 12 MB**（别名实测
> 阈值 ~12MB）。R59 的 8 MB 物理上其实是**共享 LLC**；`L3Model` 默认容量改为 L3+LLC ≈12.3 MB
> （评分改善，选择不变），`INFVINO_L3_GEOM=1` 用仅 GPU 私有值做 A/B。

---

## 1. 工具（本轮新增）

| 工具 | 作用 |
|---|---|
| `kernel_bench --op l3conflict --line-stride <s>` | 行距冲突探针（`l3_stride`：全线程活跃、只触碰 nlines 条 line） |
| `kernel_bench --op l3alias --agg-lines <A> --line-stride <Δ>` | **单缓冲**别名探针：victim 在 line 0，aggressor 从 line Δ 起（同分配、已知偏移）→ 排除跨缓冲物理地址未知 |
| `l3retain --hot-offset` / `l3_probe in_off` | 可在缓冲内偏移的热点/流，用于别名与映射探测 |

---

## 2. 逆向结果

### 2.1 索引周期 = 512 行 = 2¹⁵ B（**极锐**）

`l3conflict --nlines 1024 --passes 64`（读 `ms`）：

| stride | 508 | 510 | **511** | **512** | **513** | 516 | 640 | 768 | 1024 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | .0233 | .0232 | .0233 | **.0998** | .0231 | .0232 | .0244 | .0266 | **.2259** |

511/513 完全干净，512 骤增 → 周期精确为 **512 行（32 KB）**。与「bank=64 set=120 way」×「8 bank」
= 512 个 (bank,set) 桶一致。

### 2.2 流式容量膝点 ≈ 4 MB（= 物理 3.75 MiB）

`l3conflict --line-stride 1 --passes 1024`：0.26MB→200 GB/s，1MB→170，2MB→144，4.19MB→126，
6.29→115，8.39→93.5，12.58→50，16.78→35。膝点 ~4 MB。

### 2.3 单缓冲别名：阈值为 **~12 MB ≈ GPU L3 + 共享 LLC**

> 初测用 256 KB victim + `hiters=8`，**不敏感**（一次 1 MB 内 miss 的成本被 kernel 底噪/摊销
> 掩盖）→ 误得「24 MB 也不逐出」。改用 **1 MB victim + `hiters=1`** 重测：

`l3alias --nlines 16384 --passes 1`（victim 1 MB，先预热；aggressor 一次流式，Δ 行后）：

| aggressor | Δ=0 | Δ=512 | Δ=2048 |
|---|---:|---:|---:|
| 2–8 MB | 0.0098 | 0.0098 | 0.0099 |
| 12 MB | 0.0104 | 0.0108 | 0.0106 |
| 16 MB | 0.0131 | 0.0122 | 0.0122 |
| 24 MB | 0.0186 | 0.0176 | 0.0236 |
| 32 MB | 0.0243 | 0.0202 | 0.0209 |

→ 逐出阈值 **≈12 MB**（1 MB victim + 12 MB aggressor ≈ 13 MB），到 24–32 MB 明显变慢；
且仍**与 Δ 无关**（地址映射/共享 LLC 使偏移不敏感）。

**≈12 MB ≈ GPU 私有 L3（3.75）+ 共享 LLC（8）≈ 11.75 MB** —— 与**两级层次**吻合。

### 2.4 未逆出的部分（诚实）

* stride=512 的**冲突起点 ≈512 行**（`l3conflict --line-stride 512` 扫 nlines：≤512 平、≥640 起）。
  若桶 = `line mod 512`、ways=120，起点应为 120——不符。说明存在**bank 层**或高位参与，
  单一步进实验无法分离 bank 与 set。
* 别名对 Δ 不敏感 → 主因是**共享 LLC**：victim 被 GPU L3 逐出后仍驻留 LLC（8 MiB），
  偏移不影响「谁在 LLC 里」；也与高位索引/共享映射一致。no-allocate 仍需后续实验判定。
* 要彻底逆出 XOR 哈希，需要**物理地址可控**的实验（大页对齐缓冲 + 基址偏移扫描，或 GPU 侧
  地址翻译），本轮工具尚不足以定论。

---

## 3. 关键澄清：**两级层次**（GPU 私有 L3 → 共享 LLC → DRAM）

用户澄清 + 文档确认：**3.75 MB 是 Iris Xe 核显私有 L3 Data Cache**；**共享 LLC（= CPU L3）**
是核显与 CPU 共用的下一级（本机 4 核/8 线程，sysfs `index3` = **8 MiB**）。PRM Vol.7 也明确写
GPU L3「把 dirty line push 到 **LLC**」（见 §1 表/`third_party/intel-prm/NOTICE.md`）。

| 量 | 值 | 证据 |
|---|---:|---|
| GPU **私有** L3 Data Cache | **3.75 MiB** = 8×480 KiB = 512×120×64 | §2.1 周期 512 + §2.2 膝点 ~4MB + PRM |
| **共享 LLC**（CPU L3） | **8 MiB** | `lscpu` / sysfs index3（shared_cpu_list=0-7） |
| **跨算子热重用**有效容量 | **≈12 MB** ≈ 3.75+8 | §2.3 别名阈值 ~12MB + R55 双租户 8–12MB |

因此：
* R59 的 `8.0e6` 物理上是**共享 LLC**（sysfs），**不是** GPU 私有 L3，也不是 CPU 独有；
* GPU L3 逐出后数据仍在共享 LLC → 面向**重用张量**的有效容量 ≈ **L3 + LLC ≈ 12 MB**；
* 正确模型：`l3PhysicalBytes()`=3.75 MiB（GPU 私有）、`l3LlcBytes()`=8 MiB（共享）、
  `l3WarmCapBytes()`=`l3PhysicalBytes()+l3LlcBytes()`≈12.3 MB（`L3Model` 默认），
  `INFVINO_L3_GEOM=1` 用仅 GPU 私有值 A/B。

评分影响（**不改选择**，三模型 `model_check` 逐位不变）：y8 `spill 1.1357→0.9239`、
`total 11.379→11.167`（对实测 busy 误差 +7.6%→**+5.6%**）；y11 `1.2508→1.0367`（+7.6%→+5.7%）。

---

## 4. 对复杂度的影响

不变：单次模拟 **O(A)**；几何 512 set × 120 way（ways>64 → 两字位掩码）；
行粒度内存 512×120 ≈ 1–3 MB；A_L≈3×10⁶ 时 ~10–100 ms/次。**可行**。
新增认识：真实替换有**两个容量尺度**，单容量 LRU 是近似；行粒度 1b-NRU 才有望显式表达
「热重用保护」，但需先解决 §2.4 的哈希与 no-allocate 问题。

---

## 5. 复现

```bash
scripts/gpu_clocks.sh lock
# 周期（锐）
./build/kernel_bench --op l3conflict --nlines 1024 --wi 1024 --passes 64 \
  --line-stride 508,510,511,512,513,516,640,768,1024
# 流式容量膝点
for nl in 1024 4096 16384 32768 65536 131072 262144; do
  ./build/kernel_bench --op l3conflict --nlines $nl --wi 1024 --passes 1024 --line-stride 1; done
# 单缓冲别名（热抗污染）
for a in 65536 131072 196608 262144 393216; do
  ./build/kernel_bench --op l3alias --nlines 4096 --agg-lines $a --passes 8 --line-stride 0,512,1024,2048; done
scripts/gpu_clocks.sh unlock
# 数值（默认=跨算子热容量 L3+LLC≈12.3MB，与 R59 逐位相同）
python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD"
INFVINO_L3_GEOM=1 python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD"   # A/B（仅 GPU 私有 3.75MiB）
```

---

## 6. 下一步（继续逆向）

1. **物理地址可控实验**：把 victim/aggressor 放在同一 2 MB 大页内并扫页内偏移，或申请
   大对齐缓冲，消除页表扰动，直接测 `index(Δ)`。
2. **bank/set 分离**：用两条步进（一条扰动 set、一条扰动 bank）或「同一 set 不同 bank」的
   指针追逐，测出 bank 位与 set 位。
3. **no-allocate 判定**：比较流式读 vs 回写写对热集合的污染强度，判定 aggressor 是否分配。
4. 逆出后把 `L3LineModel` 的 `sets/ways/hash` 换成实测映射，重新评估逐出曲线**形状**保真度。
