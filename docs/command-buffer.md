# command buffer（`cl_khr_command_buffer`）可行性实测

> 目标：评估用 OpenCL command buffer 做「一次录制、每帧重放」（CUDA graph 类比）
> 是否可行，替代/缓解 P2 的 host launch 开销。R32 已有原型（`INFVINO_CMDBUF=1`），
> 见 [`PlanModel.cpp`](../src/PlanModel.cpp) 的 `buildCommandBuffer()`。
>
> **结论（2026-10 实测）：不可用。** 本机（Tiger Lake / Iris Xe 80EU）即使换到
> Intel 最新版 compute runtime，`cl_khr_command_buffer` 也不暴露；而且上游当前
> **有意只实现了一半**——连录制 kernel 的入口都还没有。短期应回到 P2 host 开销路线。

## 1. 现状：dev 镜像与最新 runtime

| | dev 镜像（`infvino-dev:latest`） | 最新（本文实测） |
|---|---|---|
| 基座 | Ubuntu 22.04（glibc 2.35） | Ubuntu 24.04（glibc 2.39） |
| OpenCL ICD | `intel-opencl-icd` 23.17.26241.33 | `intel-opencl-icd` **26.35.39758.10** |
| IGC | `libigc1` 1.0.13822.8 | `intel-igc-core-2/opencl-2` **2.41.5** |
| gmmlib | `libigdgmm12` 22.3.5 | 22.10.0 |
| ocl-icd | 2.2.14（Ubuntu 22.04 系统包） | 24.04 系统包 |

获取路径的坑（实测于本机网络）：

- Intel 给 Ubuntu 的 apt 源 `.../dists/jammy/unified/` **被冻结在 23.17**；noble 源
  在本网络返回 403。
- 最新版只在 Intel 的 **GitHub release**（`intel/compute-runtime` + `intel/intel-graphics-compiler`），
  且 deb 依赖 `libc6 (>= 2.38)`、`libstdc++6 (>= 13.1)` → **Ubuntu 22.04 装不上**，
  必须 24.04。
- 本机 docker.io 不可达（需镜像源），GitHub release 资产也不可达（需代理）。

> 复现用的临时镜像：`infvino-dev-noble:test`（Ubuntu 24.04 + 上述 26.35 deb）。

## 2. 探针与结果

探针做两件事：① `CL_DEVICE_EXTENSIONS` 里是否有 `cl_khr_command_buffer`；
② 用 `clGetExtensionFunctionAddressForPlatform` 枚举全部入口点。

```c
// 关键部分（完整探针见文末复现命令）
size_t n = 0; clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, 0, NULL, &n);
char *ext = malloc(n + 1); clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, n, ext, NULL);
ext[n] = 0;
int has = strstr(ext, "cl_khr_command_buffer") != NULL;   // 扩展串
void *f = clGetExtensionFunctionAddressForPlatform(plat, "clCommandNDRangeKernelKHR");
```

| 环境 | 扩展串 | 入口点 |
|---|---|---|
| dev 镜像（23.17） | 无 | **0 / 15** |
| 26.35 默认 | 无 | **0 / 15** |
| 26.35 + `NEOReadDebugKeys=1 EnableClKhrCommandBuffer=1` | 无 | **0 / 15** |

`EnableClKhrCommandBuffer` 的调试键前缀也逐一试过
（`NEO_` / `NEO_L0_` / `NEO_OCL_`），结果相同。

### 为什么最新版也没用

上游 commit `acdefe27c2`（2026-07-31）`feature(leo): add cl_khr_command_buffer infrastructure`
和 release tag 里的 `documentation/LEO_COMMAND_BUFFER.md` 写得很明确：

> **Partially implemented — do not enable outside development.**
> 已实现：command buffer 对象、创建/终结/引用计数、info 查询、**空** buffer 的重放。
> 未实现：**所有录制入口**（`clCommandNDRangeKernelKHR`、`clCommandCopyBufferKHR`、
> `clCommandFillBufferKHR` …）、sync point、`cl_khr_command_buffer_mutable_dispatch`。
> **因此 command buffer 目前装不进任何工作。**

源码印证：`level_zero/api/opencl/source/api/api_command_buffer.cpp` 只有 6 个入口
（create/finalize/retain/release/enqueue/getinfo）；`CommandBuffer::isSupported()`
仅 `EnableClKhrCommandBuffer == 1` 时返回真，且整套代码在 **LEO（OpenCL-on-Level-Zero）
驱动机**里。Tiger Lake 走的是经典 NEO OpenCL 路径，不会注册它——所以本机连这半套都拿不到。

### 附带确认：新驱动不改变 launch 开销

项目在 26.35 上**能编译、能跑**（`kernel_bench --op chain`，8192 WI × 1024 loop，N=200）：

```
per-dispatch: wall=18.69 us  host=14.04 us  gpu_busy=11.75 us  wall-busy=6.93 us
```

与 R32 在 23.17 上测得的 `clEnqueueNDRangeKernel ≈ 14–17 µs/次` 基本一致。
即：**换最新 runtime 不会降低 per-dispatch host 开销**，对 P2 没有直接帮助。

## 3. 结论与替代路线

1. **`cl_khr_command_buffer`：当前不可行，且不是「换个版本就行」。**
   - 本平台（TGL）未接线到 LEO，扩展不暴露；
   - 即便暴露（LEO 平台），上游录制入口尚未实现。
   - 跟进上游 `LeoCommandBuffer` / `clCommandNDRangeKernelKHR` 的进度，并确认 TGL 是否
     会切到 LEO；完成前不要基于它做设计。
2. **Level Zero 命令列表**：Intel 侧真正的「录制一次、重放多次」原语（NEO 的 command
   buffer 也只是把 L0 command list 翻译过去）。但项目现在**零 L0 依赖**，要用得从
   OpenCL C 迁到 SPIR-V/IGC，等于重写运行时后端；成本极大，暂不考虑。
3. **现实路线：继续 P2 host 开销**（见 [`benchmark.md`](benchmark.md) §2.4）。按实测分段
   （`clEnqueueNDRangeKernel` 入队 1.7–2.4 ms + 其余 host≈`setArg` ~2 ms），可做：
   参数缓存（`MEMORY_CHANGED` 式跳过 `setArg`）、减少 dispatch 数（融合 / 持久化 kernel）、
   去掉 profiling 下的逐节点同步、多 Session/队列。这些都不依赖新驱动特性。

## 4. 复现

> 本次用的是**临时文件，未入库**（按「保留 dev 镜像不动」的决定；上面的探针片段可
> 自行补成独立 `.c`）。步骤如下：

```bash
# 1) 建临时镜像：Ubuntu 24.04 + 26.35.39758.10 的 deb（GitHub release，经代理）
#    deb：intel-opencl-icd、libigdgmm12、libze-intel-gpu1（compute-runtime）
#         intel-igc-core-2、intel-igc-opencl-2（intel-graphics-compiler v2.41.5）
#    依赖：libc6>=2.38、libstdc++6>=13 → 必须 24.04（22.04 装不上）。
#    Dockerfile 大致为：FROM ubuntu:24.04 → 装 build/OpenCL 依赖 → dpkg -i *.deb
docker build -t infvino-dev-noble:test .   # 使用临时 Dockerfile

# 2) 跑探针（默认 + 调试键），探针源码见上文片段
docker run --rm --device=/dev/dri/renderD128 -v "$PWD":/t infvino-dev-noble:test bash -lc '
  gcc -O2 -o /tmp/p /t/cmdbuf_probe.c -lOpenCL && /tmp/p
  NEOReadDebugKeys=1 EnableClKhrCommandBuffer=1 /tmp/p'

# 3) 对比项目内 R32 原型（旧/新 runtime 都回退到普通 replay）
INFVINO_CMDBUF=1 ./build/kernel_run --plan models/yolov8n-pose/model.plan --report --iters 3
```

> 归档信息（2026-10-04）：
> - 实测最新版：`intel/compute-runtime` **26.35.39758.10**（2026-09-17），
>   IGC **2.41.5**，gmmlib 22.10.0。
> - 判定「不完整」的文档：`documentation/LEO_COMMAND_BUFFER.md`（该 tag）；
>   基础设施 commit `acdefe27c2`（2026-07-31）。
> - 临时产物（可删）：镜像 `infvino-dev-noble:test`、下载目录 `/tmp/opencode/neo/`。
