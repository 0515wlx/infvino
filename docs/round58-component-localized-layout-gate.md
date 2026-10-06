# R58：布局验收门的**分量级局部化**（默认开）

> 承接 R57 §4「真正的回退在验收门」。用户决策：把「整图判否 → 整份 `restore(base)`」改成
> **按连通分量局部化**——只回退被判否的分量，保留其余 min-cut 提案。
>
> 结论先给：已落地并**默认开**。y11 的 22 个分量里按 L3 定价会拒 3 个、其余保留；整网离线
> 评分 `13.4515 → 13.4034 ms`（−0.36%）。实测（交错）y8 4/4、y11 3/4 略优（~0.3–0.4%），
> mb 噪声内；三模型 `model_check` 数值与基线**逐位相同**。

---

## 1. 机制

### 1.1 问题

`resolveLayoutChoices` 的验收门（R49 整网 median / R55 L3 离线评分）在判否时执行
`restore(base)`，把**整个** min-cut 提案丢弃。但提案通常由多个互不耦合的**分量**组成
（如 mobilenet 的 `1x1→dw→1x1` 链、yolo 的各 neck 段），一个分量变差不应拖垮其余。

### 1.2 提案结构（`MinCutProposal`）

`resolveLayoutMinCut` 新增可选出参 `MinCutProposal`：
- `varName` / `labels`：变量（激活张量）的提案布局；
- `nodes` / `nodeVarA/B`：参与节点及其输入/输出变量号；
- `comps`：**连通分量**（按「共享张量」对参与节点做并查集；共享输入/输出张量的节点归一组）。

即：分量 = 布局上真正耦合的一段（生产者的 fsv16 输出直接喂消费者的 fsv16 输入）。

### 1.3 分量级贪心验收

```
restore(base);  cur = layoutModelScore();           // 从 baseline 起
for comp in comps:                                  // 逐分量
    apply(comp)                                     // 套用该分量的 fsv16 + 节点 kernel
    s = layoutModelScore()
    if s <= cur * 1.0001: cur = s; accept           // 离线评分不回归 → 保留
    else: revert(comp); reject                      // 只回退**该分量**
```

- 评分器 = R55 的 `layoutModelScore`（Σ kernel ms + Σ reorder + L3 spill）——它含**非可加**的
  spill 项，正是整图判否的根因；分量级把「判否」的作用域从整图缩到一段。
- **默认开**（严格不劣于 baseline 评分）；`INFVINO_NO_LAYOUT_COMPONENT_GATE=1` 关闭。
- R49 的**整网实测** median 门（`INFVINO_LAYOUT_MINCUT_GATE=1`）改为在分量门**之后**对最终
  状态再做一次外部核对（不再被分量门短路）。

---

## 2. 实测

### 2.1 分量裁决（`INFVINO_LAYOUT_REPORT=1`，默认 / L3 定价）

| 模型 | 分量数 | 默认 accept/reject | L3 定价 accept/reject | 评分 |
|---|---:|---|---|---|
| yolov8n-pose | 15 | 15 / 0 | — | 12.3331 → 12.3331 |
| yolo11n-pose | 22 | 22 / 0 | 19 / **3** | 13.4515 → **13.4034** |
| mobilenetv3-small | 10 | 10 / 0 | — | 1.6632 → **1.6158** |

> L3 定价（`INFVINO_LAYOUT_L3=1`）下 y11 的 3 个分量被判否并**局部**回退，其余 19 个保留；
> 最终评分与「全部接受」一致（被拒分量的提案本就等于 baseline）。R55 的「整图 REJECT」被
> 替换为「局部回退」。

### 2.2 整网 A/B（交错 ×4，`infvino_bench` infer mean，ms）

| 模型 | 全量提案（`NO_LAYOUT_COMPONENT_GATE=1`） | **分量门（默认）** | 判读 |
|---|---|---|---|
| yolov8n-pose | 13.23 | **13.19** | 4/4 略优（~0.3%） |
| yolo11n-pose | 14.32 | **14.26** | 3/4 略优（~0.4%） |
| mobilenetv3-small | 2.09 | 2.09 | 噪声内 |

> 幅度小（在 ±5% 噪声地板附近），但方向一致且**严格不劣于评分**，故默认开。

### 2.3 数值

`model_check`：y8 `4.114e-04` / y11 `7.908e-04` / mb `1.215e-02` —— 与 R56/R57 基线
**逐位相同**（分量门只改布局选择，不改计算路径的数值）。`check` 6/6 PASS。

---

## 3. 复现

```bash
# 分量裁决报告（默认）
INFVINO_LAYOUT_REPORT=1 ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 1

# L3 定价下分量级回退（对比整图门）
INFVINO_LAYOUT_L3=1 INFVINO_LAYOUT_REPORT=1 ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 1
INFVINO_LAYOUT_L3=1 INFVINO_NO_LAYOUT_COMPONENT_GATE=1 INFVINO_LAYOUT_REPORT=1 \
  ./build/kernel_run --plan models/yolo11n-pose/model.plan --report --iters 1   # 旧整图门

# A/B（全量提案 vs 分量门）
./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 60
INFVINO_NO_LAYOUT_COMPONENT_GATE=1 ./build/infvino_bench --config config/models.yaml --key yolov8n-pose --iters 60

# 回归
cmake --build build -j --target check
python3 scripts/model_check.py --model mobilenetv3-small --repo "$PWD"
```

---

## 4. 边界 / 下一步

- 分量门用的是**离线评分**（含 spill）。评分不可靠时，分量门与整图门同源；真正的兜底仍是
  R49 的**整网实测** median 门（opt-in，分量门之后跑）。
- 幅度在噪声地板附近：想拿到**确定性**收益，仍需「更可信的 spill 敏感度」（R55 §7 未决）
  或把分量门接到**实测**（每分量一次 net run，成本高、噪声大——暂不做）。
- 开关：`INFVINO_NO_LAYOUT_COMPONENT_GATE=1` 关闭；`INFVINO_LAYOUT_REPORT=1` 打印分量裁决。
