# R36：持久 blocked 布局 + 布局自动化（P1-layout 重开落地）

> 目标：把 `conv_blk`（OV 阻塞式 conv 移植）的输入重排从「每层每帧一次」变成
> **blocked 链内零重排**，并且让「哪里用阻塞布局」这件事**由 autotune 自动决定**，
> 不再靠人工按模型/按层标注。
>
> 结论：数值**逐位不变**、三模型 `model_check` PASS、`reuse_check` PASS；
> yolov8n reorder dispatch **24→10/帧（−58%）**、kernel busy **−2.2%**、墙钟 **−1.6%**；
> yolo11n reorder **23→13**、busy **−1.1%**、墙钟 **−2.2%**。收益量级与
> [`openvino-gap-analysis.md`](openvino-gap-analysis.md) §5 当初对 P1-layout 的
> 「≈2% 墙钟」预测一致——但现在是**自动、无风险、可回归**的机制。

---

## 1. 问题：reorder 是 blocked conv 的固定税

`conv3x3_blk` 读 `b_fs_yx_fsv16` 输入（`[Cin/16][H][W][16]`），旧实现里
`PlanModel::blkInput()` 对**每个** blocked conv 的输入都跑一次
`reorder_bfyx_to_fsv16`：

- 即使生产者本身就是一个 blocked conv（它完全可以直接产出 fsv16）；
- 即使同一张量被多个 blocked conv 消费（同一帧内重复重排 N 次）。

y8/y11 各有 24/23 个 blocked conv ⇒ 24/23 次 reorder dispatch、≈0.66–0.68 ms/帧（≈5% busy）。
`docs/openvino-gap-analysis.md` 把这块判为「需要持久 blocked 重构、ROI 与风险不匹配」而暂缓
（P1-layout 关闭）。R36 把它重开，但**只做「blocked 链」这一子集，并把决策自动化**。

---

## 2. 机制

### 2.1 布局作为张量属性

`PlanModel::Tensor` 增加 `bool fsv16`（默认 false = NCHW/bfyx）。元素数不变，因此
**不改变激活内存池**的分配（`allocateActivations()` 不感知布局）。

### 2.2 计划期布局规划：`planBlockedLayout()`

在构造期 `tuning_` 载入**之后**、首次 `run()` **之前**跑一次（P2 dispatch 缓存会在首帧录制
参数，布局必须在那之前定稿）。规则：

```
对每个 conv3x3 节点 P（判断它是否走 blocked，见 2.4）：
  张量 t = P 的输出
  若 P 走 blocked 且 t 不是 network output 且 Cout % 16 == 0
     且 t 的所有消费者都是「走 blocked 的 conv3x3」且都从输入槽 0 读 t：
         t.fsv16 = true
```

`INFVINO_NO_BLOCK_LAYOUT=1` 关闭。

### 2.3 运行期三处配合

1. **生产者**：`conv_blk.cl` 新增 `-DOUT_FSV16=1`，直接把结果写成
   `b_fs_yx_fsv16`（`output[((f_block*Hout+y)*Wout+ox)*16 + lid]`）。
   lane 0..15 落到连续 16 个 half（32B，**合并写**），比原来的 bfyx 跨通道散写更好。
   仅当 `Cout%16==0` 才启用（否则按 16 补齐的索引会越界）。
2. **消费者**：`blkInput()` 发现 `x.fsv16` 直接返回 `x.mem`，零 reorder、零 scratch。
3. **同帧去重**：`blkInput()` 对「本帧已重排过的张量」直接复用结果，跳过重复 launch
   （多个 blocked 消费者共享一次 `bfyx->fsv16`）。每帧 `run()` 开头清空 `reordered_frame_`——
   这是跨推理陈旧 bug 的护栏（见下）。`INFVINO_NO_REORDER_DEDUP=1` 可关。

### 2.4 为什么「自动化」是安全的

`convWillUseBlk()` 与 `run()` 里 conv3x3 分支的 kernel 选择**用同一套判据**（同一
`OpSignature`、同一 RES 回绕规则），并且规划用的就是构造期载入的 `tuning_`。因此
「规划说 fsv16」与「运行期走 blocked」永远一致；缓存缺失/关闭时两者都退回启发式（不阻塞），
不会出现布局错配。**布局完全由 autotune 驱动**：换个设备、重扫 tuning.json，布局自动跟着变。

> 规则最优性（在「同帧去重」前提下）：某张量存储 NCHW 时，其所有 blocked 消费者共享
> 1 次 reorder；存储 fsv16 时，若有非 blocked 消费者则需 1 次反向物化。故
> 「所有消费者都是 blocked ⇒ fsv16（省 1），否则 NCHW」是逐张量最优，无需全局搜索。

---

## 3. 实测

环境：Iris Xe 80EU；`iters=5`（+1 帧 B 输入），报告里的 op 计数为 `calls/iters`，
换算到每帧需 ×5/6。三模型 `model_check`（vs onnxruntime）**PASS**，A/B 输出**逐位一致**。

| 模型 | 配置 | reorder dispatch/帧 | kernel busy/帧 | 墙钟(profiling) |
|---|---|---:|---:|---:|
| yolov8n | base | 24 | 13.84 ms | 18.26 ms |
| yolov8n | +同帧去重 | 20 | 13.77 ms | 18.33 ms |
| yolov8n | +持久 fsv16 | 20 | 13.73 ms | 18.21 ms |
| yolov8n | **both** | **10** | **13.53 ms（−2.2%）** | **17.97 ms（−1.6%）** |
| yolo11n | base | 23 | 14.67 ms | 20.19 ms |
| yolo11n | **both** | **13** | **14.52 ms（−1.1%）** | **19.74 ms（−2.2%）** |
| mobilenet | any | 0（无 blocked conv） | 持平 | 持平 |

- 去重与持久化**各自独立有效**：y8 reorder 24→20（去重）→10（再持久化）。
- 逐位一致：base / 去重 / 持久化 / both 的 `kernel_run` 输出两两 `cmp` 相同；
  `INFVINO_NO_LAUNCH_CACHE=1` 路径同样逐位一致。
- 诊断读回：`readTensor()` 对 fsv16 张量在主机侧转回 NCHW，
  `--dump-tensor` 与数值检查看到的语义不变（`INFVINO_NO_POOL=1` 下与 NCHW 输出逐位一致）。

复现：

```bash
# 离线布局规划/分析（读 plan + config/tuning.json，不碰 GPU）
python3 scripts/analyze_layout.py                 # 扫全部模型
python3 scripts/analyze_layout.py models/yolov8n-pose/model.plan

# 端到端数值（含 A/B 逐位一致）
python3 scripts/model_check.py    --model yolov8n-pose --repo $PWD --image infvino-dev:latest
python3 scripts/reuse_check.py    --model yolov8n-pose --repo $PWD --image infvino-dev:latest

# 运行时 A/B：INFVINO_NO_BLOCK_LAYOUT=1 / INFVINO_NO_REORDER_DEDUP=1
./build/kernel_run --plan models/yolov8n-pose/model.plan \
  --input A.bin --input2 B.bin --iters 5 --report
INFVINO_LAYOUT_REPORT=1 ./build/kernel_run ...   # 打印标记为 fsv16 的张量数
```

---

## 4. 边界、未做与负结果

- **只做 fsv16 输出，不做反向物化**：若某 blocked 输出的消费者里既有 blocked conv 又有其他
  算子，规划器选择保持 NCHW（宁可让 blocked 消费者重排一次），不引入 `fsv16->bfyx` 物化。
  原因是「同帧去重」下两种选择的 reorder 次数相同（见 §2.4），物化没有净收益。
- **要求 `Cout % 16 == 0`**：fsv16 按 16 通道补齐，未补齐会越界；检测头里 Cout=51 之类的层
  因此不参与（它们本来也多半不是 blocked 链的中间层）。
- **网络输出永不 fsv16**：`readOutput` 仍是 NCHW。
- **不重排 killing**：`reorder_bfyx_to_fsv16` 仍是主路径；没有为「向量化 store 更快」
  再试（R-P1 已证负结果）。
- **未做：跨 kernel 的联合选择**（为了拉长 blocked 链而故意不选 autotune 的最快 kernel）。
  单 kernel 时间远大于一次 reorder（~28 µs），判为不划算。
- **未做：布局的全局搜索/物化**（OV 式 layout optimizer + reorder 插入/消除）。当前规则在
  「只优化 blocked 链」的约束下已是最优；不引入 OV 的 format/reorder 全家桶。

---

## 5. 关键文件

| 位置 | 作用 |
|---|---|
| `scripts/analyze_layout.py` | 离线规划/分析器：per-conv kernel 决策、fsv16 候选、reorder 计数（自动化大脑与规格） |
| `src/PlanModel.cpp` `planBlockedLayout()` / `convWillUseBlk()` | 计划期布局规划（由 tuning 驱动） |
| `src/PlanModel.cpp` `blkInput()` | fsv16 直接读 + 同帧去重 |
| `kernels/conv_blk.cl` `OUT_FSV16` | 生产者直接写 `b_fs_yx_fsv16` |
| `src/PlanModel.cpp` `readTensor()` | fsv16 诊断读回时转回 NCHW |
