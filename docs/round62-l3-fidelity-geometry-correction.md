# R62：硬件 L3 模型的**保真度**——完整测量纠正 R59 的容量硬伤

> 用户：先做完整测量；硬件模型的保真度很重要，否则对离线仿真复杂度的判断也不一定对。
>
> 结论先给（一个**保真度硬伤**）：
> * 独立几何测量表明 **GPU L3 = 3.75 MiB = 8 bank × 480 KiB = 512 set × 120 way × 64 B
>   = 3,932,160 B**。R59 用的 `8 MiB` 是 **CPU 的** L3（`lscpu` / sysfs index3），**不是 GPU 的**。
> * 证据三条独立一致：① 步进冲突周期 **512 行（32 KB）**；② 容量扫描膝点 **≈4 MB**；
>   ③ 公开 PRM：bank=480 KiB=120 way×64 set×64 B，8 bank。
> * 已把默认容量改为 **3.75 MiB**（`INFVINO_L3_LEGACY=1` 回退 R59 的 8 MiB 做 A/B）。
>   三模型 `model_check` **逐位不变**（只改评分，不改选择）。
> * 保真度评估：`L3LineModel`（1b-NRU, 512×120）的容量膝点与实测吻合；过渡比实测**更陡**
>   （未建模真实地址哈希/分段）。
> * 复杂度结论随之**收紧**：ways=120（>64）需要两字位掩码或 sector 分组才能 O(1) 选 victim。
>
> ⚠️ **R63 更正**：物理 3.75 MiB 是**流式/物理**容量；实测（单缓冲别名 + R55 双租户）表明
> **跨算子热重用**的有效容量 ≈ **8–12 MB**。`L3Model` 模拟的是重用张量 ⇒ 默认容量仍取 8 MB
> （与 R59 数值一致），物理几何由 `l3PhysicalBytes()` 单独给出；`INFVINO_L3_GEOM=1` A/B。
> 见 [`round63-address-mapping-reverse-engineering.md`](round63-address-mapping-reverse-engineering.md)。

---

## 1. 完整测量

全部锁频（`scripts/gpu_clocks.sh lock`）。工具为 R61 新增的 `kernel_bench --op l3conflict`
（`l3_stride` 内核：所有线程活跃、只触碰 `nlines` 条 line、行距 `stride_lines`）与
`--op l3retain`（保留率）。

### 1.1 步进冲突 → 组数 / bank 周期

`l3conflict --nlines 1024 --wi 1024 --passes 64 --line-stride <s>`（读 `ms`，越大=冲突/逐出）:

| stride(行) | 1 | 2 | 4 | 16 | 32 | 64 | 128 | 256 | **512** | 768 | **1024** | 1536 | **2048** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | .023 | .023 | .022 | .022 | .023 | .023 | .022 | .031 | **.195** | .028 | **.373** | .171 | **.398** |

→ 冲突出现在 **512 的倍数**（512/1024/1536/2048），256 半程。即**组周期 = 512 行 = 32 KB**。
与 PRM「bank = 64 set」×「8 bank」= 512 set 一致。

### 1.2 容量扫描（stride 1，连续/无冲突）

`l3conflict --line-stride 1 --wi 1024 --passes 1024`：

| 工作集 | 0.26 MB | 1.05 MB | 2.10 MB | 3.15 MB | 4.19 MB | 6.29 MB | 8.39 MB | 12.58 MB | 16.78 MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GB/s | 200 | 170 | 144 | 141 | 126 | 115 | 93.5 | 50.2 | 34.8 |

→ 膝点 **≈4 MB**，8 MB 后**骤降**。若 L3 是 8 MB，连续流到 8 MB 才会掉——实测远早于此。
**3.75 MiB** 的几何（512×120×64）与膝点一致。

### 1.3 保留率（`l3retain`，reuse=4，热点尺寸扫描）

| 热点 | R_A=4.19 | 6.29 | 8.39 | 12.58 | 16.78 |
|---|---:|---:|---:|---:|---:|
| 256 KB | 1.02 | 0.99 | 0.99 | 0.70–0.78 | 0.54–0.60 |
| 512 KB | 1.01 | 0.95 | 0.78–0.88 | 0.54–0.56 | 0.48–0.52 |
| 2 MB | 1.00 | 0.97 | 0.85–0.92 | 0.66–0.68 | 0.58–0.61 |

→ 阈值随热点变大而**前移**（256KB 到 ~12MB 才掉，2MB 从 ~6MB 起掉）——与「热点+aggressor
超过容量即逐出」一致（容量 ~4MB），且 reuse 的抬升仍是**测量摊销**（R61 §1.2）。

---

## 2. 保真度评估（模型 vs 实测）

`l3linesim --op scale`（sets=512 ways=120，随机工作集，命中率）对照 §1.2：

| 工作集 | 1 MB | 2 MB | 4 MB | 8 MB | 16 MB |
|---|---:|---:|---:|---:|---:|
| 模型命中率 | 98.4% | 96.9% | 92.6% | 47.3% | 23.7% |
| 实测 GB/s（归一到 ~200） | 0.85 | 0.72 | 0.63 | 0.47 | 0.17 |

* **容量位置吻合**（都≈4 MB 附近开始掉）。
* **过渡形状**：模型在 4→8 MB **更陡**（92%→47%），实测更**平滑**（0.63→0.47→0.17）。
  根因：模型用 `set = line % sets`（均匀映射），真实硬件是**地址哈希 + 2-way sector**，
  冲突在组间更均匀 → 平滑。**精确到形状**需要真实地址→set/bank 映射（未公开/需进一步逆向）。
* 结论：**容量**可靠（用于成本模型的容量/逐出阈值）；**逐出曲线形状**是已知近似。

---

## 3. 纠正的常量（R59 → R62）

| 量 | R59 | **R62（实测几何）** |
|---|---|---|
| `l3PhysicalBytes()` | 8.0e6（= CPU L3） | **3,932,160**（GPU L3 = 8 bank × 480 KiB） |
| `l3CpuL3Bytes()` | — | 8,388,608（引用/对照） |
| `l3DefaultCapBytes()` | 8.0e6 | **3,932,160**；`INFVINO_L3_LEGACY=1` → 8 MiB（A/B） |
| `l3PrivateCapBytes()` | 2.0e6 | 2.0e6（不变） |

影响（评分，不改选择）：y8 `spill 1.1357 → 2.1291`，`total 11.379 → 12.372`。
三模型 `model_check` **逐位不变**（`4.114e-04 / 7.908e-04 / 1.215e-02`；fsv16 0/9/23 不变）——
即 R59 的「8 MiB 更贴近 busy」是**容量错误对其他误差的补偿**：现在 ~18% 的 busy 缺口是
**真实的模型误差**（kernel/roofline/逐出语义），不能再靠调容量掩盖。

---

## 4. 对复杂度判断的影响（收紧）

* 几何 = **512 set × 120 way**，总 61,440 行（3.75 MiB）。
* 选 victim：**ways=120 > 64**，单个 `uint64` 位掩码不够 → 需**两字**（或按 sector=2 way 分组）
  才能 `ctz` O(1)；否则是 O(ways)=O(120) 常数。
* 每访问：`set = hash(addr)`（O(1)）+ 组内 `tag→way` 表（O(1) 命中）+ victim 选择（位掩码 O(1)）。
  ⇒ 单次模拟 **O(A)**，与 R61 结论一致；常数项因 ways=120/两字掩码略增。
* 内存：512×120 = 61,440 项（tag 8B + bit 1B + map 开销）≈ 1–3 MB → 可忽略。
* 行粒度 A_L≈3×10⁶ 时仍 **~10–100 ms/次**（实测 252 ns/访问，O(ways) 扫描；位掩码后更低）。
  结论不变：**可行**。

---

## 5. 诚实边界 / 下一步

1. 组索引的**精确哈希**未逆出（只确认周期 512 行）。要精确到逐出曲线形状，需地址→bank/set
   映射（可继续用 stride 探针 + 缓冲区基址偏移扫描逆向，或接受「容量精确、形状近似」）。
2. `l3conflict` 的绝对 GB/s 口径（touch=64B、实际用 4B）与 copy 不可直接换算；本节只用**膝点/相对**。
3. R62 改默认容量属**保真修正**（有直接几何证据）；是否连带重标 `kL3SpillPerByteMs` 与逐节点
   定价，留待与整网 A/B 一起评估。
4. 许可：几何参数来自**公开** PRM（见 `third_party/intel-prm/NOTICE.md`）；实现为原创。

---

## 6. 复现

```bash
scripts/gpu_clocks.sh lock
# 组数/冲突周期
./build/kernel_bench --op l3conflict --nlines 1024 --wi 1024 --passes 64 --line-stride 1,64,128,256,512,768,1024,2048
# 容量膝点
for nl in 1024 4096 16384 32768 65536 131072 262144; do
  ./build/kernel_bench --op l3conflict --nlines $nl --wi 1024 --passes 1024 --line-stride 1; done
# 保留率
./build/kernel_bench --op l3retain --hot-lines 16384 --hot-gws 8192 --hot-sweep 1,4,16 \
  --wi 1024,2048,4096,8192 --fp 4,8,16,24,32,48 > /tmp/l3retain.csv
python3 scripts/l3_calibrate.py --raw /tmp/l3retain.csv --out config/l3_calibration.json --sets 512 --ways 120
scripts/gpu_clocks.sh unlock
# 模型 vs 实测
./build/l3linesim --op scale --sets 512 --ways 120 --working-mb 4 --accesses 500000 --reps 2
# 数值（默认=几何容量；应与 R59 逐位相同）
python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD"
INFVINO_L3_LEGACY=1 python3 scripts/model_check.py --model yolov8n-pose --repo "$PWD"   # A/B
```
