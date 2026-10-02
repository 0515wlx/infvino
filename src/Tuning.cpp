// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// Tuning 实现：OpSignature 编码 / TuningCache 读写 / expected_ops（中间标准）。
#include "infvino/Tuning.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gk
{

namespace
{
// 极简 JSON 辅助：本项目不引入第三方 JSON 库（与 onnx2plan 的 /tmp/opencode 脚本一致，
// 缓存结构扁平、字段固定）。写入用标准 JSON；读取用宽松的 "key": value 扫描，
// 对未知字段宽容（向前兼容）。
std::string jsonEscape(const std::string & s)
{
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

std::string jsonUnescape(const std::string & s)
{
  std::string o;
  o.reserve(s.size());
  for (size_t i = 0; i < s.size(); ++i) {
    if (s[i] == '\\' && i + 1 < s.size()) {
      char n = s[++i];
      o += (n == 'n') ? '\n' : (n == 't') ? '\t' : n;
    } else {
      o += s[i];
    }
  }
  return o;
}

// 在整段文本里找下一个 `"key"` 后的 `:`，返回其后的原始 token（到 , 或 } 或换行）。
std::string findField(const std::string & body, const std::string & key, bool * found)
{
  *found = false;
  const std::string pat = "\"" + key + "\"";
  size_t p = body.find(pat);
  if (p == std::string::npos) return {};
  p = body.find(':', p + pat.size());
  if (p == std::string::npos) return {};
  ++p;
  while (p < body.size() && (body[p] == ' ' || body[p] == '\t')) ++p;
  if (p < body.size() && body[p] == '"') {
    size_t e = p + 1;
    std::string raw;
    while (e < body.size() && body[e] != '"') {
      if (body[e] == '\\' && e + 1 < body.size()) { raw += body[e]; raw += body[e + 1]; e += 2; continue; }
      raw += body[e++];
    }
    *found = true;
    return jsonUnescape(raw);
  }
  size_t e = p;
  while (e < body.size() && body[e] != ',' && body[e] != '}' && body[e] != '\n') ++e;
  std::string t = body.substr(p, e - p);
  while (!t.empty() && (t.back() == ' ' || t.back() == '\t' || t.back() == '\r')) t.pop_back();
  *found = !t.empty();
  return t;
}
}  // namespace

std::string OpSignature::str() const
{
  std::ostringstream o;
  o << op;
  if (op == "gemm") {
    o << "|M" << M << "N" << N << "K" << K;
  } else if (op == "conv3x3") {
    o << "|W" << W << "H" << H << "s" << stride << "p" << pad << "_Cin" << Cin << "_Cout" << Cout;
  } else if (op == "conv_general" || op == "depthwise") {
    o << "|W" << W << "H" << H << "s" << stride << "p" << pad << "_Cin" << Cin << "_Cout" << Cout
      << "_K" << K << "_G" << groups;
  } else if (op == "conv1x1") {
    o << "|Cout" << Cout << "_N" << N << "_Cin" << Cin;
  } else if (op == "gap") {
    o << "|C" << Cin << "_HW" << N;
  } else {
    // Round 28: 通用小算子 —— params 是有序维度列表，保证签名稳定、可区分。
    o << "|";
    if (!params.empty()) {
      for (size_t i = 0; i < params.size(); ++i) { if (i) o << ","; o << params[i]; }
    } else {
      o << Cin << "x" << Cout << "@" << W << "x" << H;
    }
  }
  o << "_act" << act << "_" << dtype;
  if (batch > 1) o << "_b" << batch;
  return o.str();
}

OpSignature OpSignature::conv3x3(int Wout, int Hout, int stride, int pad, int Cin, int Cout, int act)
{
  OpSignature s;
  s.op = "conv3x3"; s.W = Wout; s.H = Hout; s.stride = stride; s.pad = pad;
  s.Cin = Cin; s.Cout = Cout; s.act = act;
  return s;
}
OpSignature OpSignature::convGeneral(int Wout, int Hout, int stride, int pad, int Cin, int Cout,
                                     int groups, int K, int act)
{
  OpSignature s;
  s.op = "conv_general"; s.W = Wout; s.H = Hout; s.stride = stride; s.pad = pad;
  s.Cin = Cin; s.Cout = Cout; s.groups = groups; s.K = K; s.act = act;
  return s;
}
OpSignature OpSignature::gemm(int M, int N, int K, int act)
{
  OpSignature s;
  s.op = "gemm"; s.M = M; s.N = N; s.K = K; s.Cin = K; s.Cout = M; s.act = act;
  return s;
}
OpSignature OpSignature::conv1x1(int Cout, int N, int Cin, int act, int res)
{
  OpSignature s;
  s.op = "conv1x1"; s.Cout = Cout; s.N = N; s.Cin = Cin; s.act = act;
  if (res) s.groups = 2;  // res 复用 groups 低 bit 仅用于签名区分
  return s;
}
OpSignature OpSignature::depthwise(int Wout, int Hout, int stride, int pad, int Cin, int K, int act)
{
  OpSignature s;
  s.op = "depthwise"; s.W = Wout; s.H = Hout; s.stride = stride; s.pad = pad;
  s.Cin = Cin; s.Cout = Cin; s.K = K; s.groups = Cin; s.act = act;
  return s;
}
OpSignature OpSignature::gap(int C, int HW)
{
  OpSignature s;
  s.op = "gap"; s.Cin = C; s.N = HW;
  return s;
}
OpSignature OpSignature::custom(const std::string & op, std::vector<int> params, int act)
{
  OpSignature s;
  s.op = op; s.params = std::move(params); s.act = act;
  return s;
}

// ---------------------------------------------------------------------------
// Device key
// ---------------------------------------------------------------------------

std::string TuningCache::deviceKey(const DeviceInfo & d)
{
  char buf[128];
  if (d.pci_device_id)
    std::snprintf(buf, sizeof(buf), "%04x:%04x_eu%u_clk%u",
                  d.pci_vendor_id, d.pci_device_id, d.eu, d.clock_mhz);
  else
    std::snprintf(buf, sizeof(buf), "%s_eu%u_clk%u", d.name.c_str(), d.eu, d.clock_mhz);
  return buf;
}

// ---------------------------------------------------------------------------
// TuningCache I/O
// ---------------------------------------------------------------------------

TuningCache TuningCache::load(const std::string & path)
{
  TuningCache c;
  if (path.empty() || path == "none") { c.enabled_ = false; return c; }
  std::ifstream f(path);
  if (!f) { c.enabled_ = true; return c; }  // 缺失 = 空缓存（回退启发式），不算禁用
  std::ostringstream ss;
  ss << f.rdbuf();
  const std::string txt = ss.str();

  const char * env = std::getenv("INFVINO_TUNING");
  if (env && std::string(env) == "off") { c.enabled_ = false; return c; }

  bool found = false;
  const std::string dev = findField(txt, "device_id", &found);
  if (found) c.device_id_ = dev;

  // 遍历 "entries" 对象里的每个 "key": { ... }。用大括号配对切分（扁平结构）。
  size_t ep = txt.find("\"entries\"");
  if (ep == std::string::npos) return c;
  ep = txt.find('{', ep);
  if (ep == std::string::npos) return c;
  size_t i = ep + 1;
  while (i < txt.size()) {
    while (i < txt.size() && (txt[i] == ' ' || txt[i] == '\t' || txt[i] == '\n' || txt[i] == '\r' || txt[i] == ',')) ++i;
    if (i >= txt.size() || txt[i] == '}') break;
    if (txt[i] != '"') break;
    // key
    size_t ke = i + 1;
    std::string keyRaw;
    while (ke < txt.size() && txt[ke] != '"') {
      if (txt[ke] == '\\' && ke + 1 < txt.size()) { keyRaw += txt[ke]; keyRaw += txt[ke + 1]; ke += 2; continue; }
      keyRaw += txt[ke++];
    }
    const std::string key = jsonUnescape(keyRaw);
    i = txt.find('{', ke);
    if (i == std::string::npos) break;
    size_t depth = 0, j = i;
    for (; j < txt.size(); ++j) {
      if (txt[j] == '{') ++depth;
      else if (txt[j] == '}') { if (--depth == 0) { ++j; break; } }
    }
    const std::string body = txt.substr(i + 1, j - i - 2);
    TuningEntry e;
    bool f1, f2, f3, f4, f5, f6, f7;
    e.kernel   = findField(body, "kernel", &f1);
    e.config   = findField(body, "config", &f2);
    e.options  = findField(body, "options", &f2);
    e.ms       = std::atof(findField(body, "ms", &f3).c_str());
    e.ops      = std::atof(findField(body, "ops_per_eu_cyc", &f4).c_str());
    e.expected = std::atof(findField(body, "expected_ops_per_eu_cyc", &f5).c_str());
    e.ratio    = std::atof(findField(body, "ratio", &f6).c_str());
    e.iters    = std::atoi(findField(body, "iters", &f7).c_str());
    e.device_id = findField(body, "device_id", &f7);
    e.source    = findField(body, "source", &f7);
    if (f1) c.entries_[key] = e;
    i = j;
  }
  return c;
}

TuningCache TuningCache::loadDefault()
{
  const char * env = std::getenv("INFVINO_TUNING_CACHE");
  if (env) return load(env);
  // 默认：与可执行文件/源码树并行的 config/tuning.json。libinfvino 编译期未注入仓库
  // 路径，因此优先 env，其次当前工作目录（生产部署常以仓库根为 cwd），再退到
  // INFVINO_KERNEL_DIR 的上级目录（源码树 kernels/../config）。
  {
    TuningCache c = load("config/tuning.json");
    if (c.size() > 0) return c;
  }
  {
    std::string kdir = INFVINO_KERNEL_DIR;
    const auto pos = kdir.find_last_of('/');
    if (pos != std::string::npos)
      return load(kdir.substr(0, pos) + "/config/tuning.json");
  }
  return TuningCache{};
}

bool TuningCache::save(const std::string & path) const
{
  if (path.empty() || path == "none") return false;
  std::ofstream f(path);
  if (!f) return false;
  f << "{\n";
  f << "  \"version\": 1,\n";
  f << "  \"device_id\": \"" << jsonEscape(device_id_) << "\",\n";
  f << "  \"entries\": {\n";
  size_t n = 0;
  for (const auto & kv : entries_) {
    const TuningEntry & e = kv.second;
    f << "    \"" << jsonEscape(kv.first) << "\": {"
      << "\"kernel\": \"" << jsonEscape(e.kernel) << "\", "
      << "\"config\": \"" << jsonEscape(e.config) << "\", "
      << "\"options\": \"" << jsonEscape(e.options) << "\", "
      << "\"ms\": " << e.ms << ", "
      << "\"ops_per_eu_cyc\": " << e.ops << ", "
      << "\"expected_ops_per_eu_cyc\": " << e.expected << ", "
      << "\"ratio\": " << e.ratio << ", "
      << "\"iters\": " << e.iters << ", "
      << "\"device_id\": \"" << jsonEscape(e.device_id) << "\", "
      << "\"source\": \"" << jsonEscape(e.source) << "\""
      << "}" << (++n < entries_.size() ? "," : "") << "\n";
  }
  f << "  }\n}\n";
  return true;
}

const TuningEntry * TuningCache::lookup(const OpSignature & sig) const
{
  if (!enabled_) return nullptr;
  auto it = entries_.find(sig.str());
  if (it == entries_.end()) return nullptr;
  // 设备键不匹配 = 该条目来自别的 GPU 型号，视为未命中（回退启发式）。
  if (!device_id_.empty() && !it->second.device_id.empty() && it->second.device_id != device_id_)
    return nullptr;
  return &it->second;
}

void TuningCache::put(const OpSignature & sig, const TuningEntry & e)
{
  entries_[sig.str()] = e;
}

// ---------------------------------------------------------------------------
// expected_ops —— 中间标准
// ---------------------------------------------------------------------------
//
// 模型与 docs/kernel.md R18.7 一致：
//   ops/EU/cyc = 32 × (mad 指令占发射槽比例) × (SIMD/16) × 网格占用修正
// 各分项系数全部来自 R13–R23 的实测结论，函数内逐一标注。
// 目标不是精确绝对值，而是「单调、可比」的期望标尺。

namespace
{
// 网格占用修正：R18.4 显示 <16 个 work-group 时 80 EU 严重饥饿，小空间层 +20–56%。
double gridFactor(long n_wg, int eu)
{
  const double target = std::max(1.0, static_cast<double>(eu) * 2.0);  // R18: ~2 WG/EU 才喂饱
  return std::min(1.0, static_cast<double>(n_wg) / target);
}

// ---------------------------------------------------------------------------
// Round 30：**剩下的 kernel**（非 conv/gemm）的中间标准。
//
// conv/gemm 的上限由「指令配额 = 32 × mad 占比」给出；但搬运/逐元素/归约类算子
// 的算术强度 ≈ 0.25–0.5 FLOP/byte，根本碰不到 FMA 发射，其物理上限是**内存 roofline**
// （bytes/EU/cyc），再叠加每个 dispatch 的固定开销。本机（Iris Xe / 单通道 LPDDR）
// R30 用 kernel_bench 实测的 copy(read+write) 带宽-足迹曲线为：
//   4KB 3.0 | 16KB 8.1 | 64KB 22.7 | 256KB 46.4 | 512KB 63.3 | 1MB 89.7
//   | 2MB 62.4 | 4MB 44.6 | 8MB 21.0 GB/s（DRAM memcpy 墙 = 19.4 GB/s）
// 每个 dispatch 的 launch floor ≈ 3.5 µs（membw 4KB copy = 0.003 ms）。
//
// 调优器对小算子传入的 `flops = 2 × 输出元素数`，因此这里的「期望」也用同一 proxy
// 量纲：expected = 2·out / (EU·clk·T)，T = launch + bytes/BW(footprint)。
// 它单调、按 shape 自洽，能直接读出「离物理上限多远」。
// ---------------------------------------------------------------------------
struct BwPoint { double bytes; double gbps; };
const BwPoint kBwCurve[] = {
  {4e3, 3.0}, {16e3, 8.1}, {64e3, 22.7}, {256e3, 46.4}, {512e3, 63.3},
  {1e6, 89.7}, {2e6, 62.4}, {4e6, 44.6}, {8e6, 21.0}};
constexpr double kSmallLaunchUs = 3.5;

// 由足迹插值 copy 带宽（log-log 线性；端点外取端点值）。
double copyBwGbps(double footprint)
{
  const int n = static_cast<int>(sizeof(kBwCurve) / sizeof(kBwCurve[0]));
  if (footprint <= kBwCurve[0].bytes) return kBwCurve[0].gbps;
  if (footprint >= kBwCurve[n - 1].bytes) return kBwCurve[n - 1].gbps;
  for (int i = 0; i + 1 < n; ++i)
  {
    if (footprint <= kBwCurve[i + 1].bytes)
    {
      const double lx = std::log(footprint / kBwCurve[i].bytes) /
                        std::log(kBwCurve[i + 1].bytes / kBwCurve[i].bytes);
      return kBwCurve[i].gbps * std::pow(kBwCurve[i + 1].gbps / kBwCurve[i].gbps, lx);
    }
  }
  return 20.0;
}

// 内存受限算子的期望 proxy（= 2·out 元素 / EU / cyc，与 autotune 的 `ops` 同量纲）。
double smallMemCeiling(double outElems, double bytes, int eu, double clkMhz)
{
  if (outElems <= 0.0 || bytes <= 0.0) return 0.0;
  const double bw = copyBwGbps(bytes) * 1e9;  // bytes/s
  const double t  = kSmallLaunchUs * 1e-6 + bytes / bw;
  return 2.0 * outElems / (static_cast<double>(eu) * clkMhz * 1e6 * t);
}

double paramAt(const std::vector<int> & v, size_t i, double dflt = 0.0)
{
  return i < v.size() ? static_cast<double>(v[i]) : dflt;
}
}  // namespace

double expectedOps(const OpSignature & s, const DeviceInfo & dev)
{
  const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;

  if (s.op == "conv3x3") {
    // R24 修正：**不再把上限定成实测**。移植的 OV 内循环 ISA 是 288 packed mad /
    // 453 指令（mad 占 63.6%，broadcast 已折进 mad 操作数），指令配额上限 ≈ 20.3，
    // 而不是 R20/R23 的「1 broadcast : 1 mad → 16」。
    //   expected = 32 × mad_frac × prologue_amort × grid_factor
    // 其中 prologue_amort 摊薄每 WG 的固定开销（清零/存储/激活/尾部），短层吃亏；
    // grid_factor 是 R18.4 的网格占用修正。缺口 = 尚未榨干的流水/延迟。
    const int obw = (s.stride == 2) ? 5 : 8;
    const int obh = (s.stride == 2) ? 4 : 2;
    const long n_wg = static_cast<long>((s.W + obw - 1) / obw) *
                      static_cast<long>((s.H + obh - 1) / obh) *
                      static_cast<long>(((s.Cout + 1) / 2 + 15) / 16);
    const double work = static_cast<double>(s.Cin) * 9.0 * obw * obh;   // mads / WG
    const double fixed = 24.0 + 6.0 * obw * obh;                        // 每 WG 固定指令
    const double amort = work / (work + fixed);
    double e = kPeakOpsPerEuCycle * kConvOvMadFraction * amort;
    e *= std::max(0.25, gridFactor(n_wg, eu));
    e = std::min(e, kConvOvIssueCeiling);
    return std::max(1.0, e);
  }

  if (s.op == "gemm") {
    // R13/R14: compute-only 17.7，整核 13.7（大网格）；寄存器墙让 TM·TN ≈ 32 封顶。
    const long grid = static_cast<long>((s.M + 127) / 128) * static_cast<long>((s.N + 63) / 64);
    double e = 13.7;  // BK32 SG16 大网格的最优（R14）
    e *= gridFactor(grid, eu);
    if (s.K < 64) e *= 0.7;      // 小 K：k-tile 流水 prologue/epilogue 亏（R9）
    if (s.M <= 64) e *= 0.8;     // BM=64 复用降低（R9）
    return std::max(0.5, e);
  }

  if (s.op == "conv1x1") {
    if (s.N == 1) {
      // split-K GEMV（R22）：lane 沿 K 归约，fp32 累加，受归约延迟 / K 长度限制。
      // 实测 5–27× 于旧 gemm；期望取「每 EU 一条 sub-group 归约链」的保守值。
      return std::max(1.0, 8.0 * gridFactor(static_cast<long>(s.Cout), eu));
    }
    return expectedOps(OpSignature::gemm(s.Cout, s.N, s.Cin, s.act), dev);
  }

  const double clk = dev.clock_mhz > 0 ? static_cast<double>(dev.clock_mhz) : 1300.0;

  if (s.op == "depthwise" || s.op == "conv_general") {
    // R30：depthwise 的物理墙**不是** FMA，而是地址/边界谓词指令。反汇编
    // depthwise_v（DW_TW=8,K=3）为 1485 条指令 / 128 mad（mad_frac=8.6%）→ 指令配额
    // 32×0.086 ≈ 2.8 ops/EU/cyc（FLOPs 已含 K×K，不再除）。grid 饥饿再乘 gridFactor。
    const long n = static_cast<long>(s.Cout) * s.W * s.H;  // gws（标量版 1 WI/输出）
    return std::max(0.3, 32.0 * 0.086 * gridFactor(n, eu));
  }

  // Round 30：**剩下的 kernel** —— 内存 roofline（见文件上方 kBwCurve 说明）。
  // 调优器对它们统一传 flops = 2·输出元素数，故 expected 用同一 proxy 量纲。
  const std::vector<int> & v = s.params;
  if (s.op == "bmm") {
    // 计算受限：FP32、K 循环 ISA mad_frac=50.8%（R30），配额 16×0.508 ≈ 8.1。
    // 期望 proxy = 配额 / K；网格饥饿（M/TM·N/TN·B）再打折。
    const double B0 = paramAt(v, 0, 1), B1 = paramAt(v, 1, 1), M = paramAt(v, 2, 1),
                 K = std::max(1.0, paramAt(v, 3, 1)), N = paramAt(v, 4, 1);
    const long   grid = static_cast<long>((N + 3) / 4) * static_cast<long>((M + 3) / 4) *
                      static_cast<long>(B0 * B1);
    const double e = (16.0 * 0.508 / K) * std::max(0.25, gridFactor(grid, eu));
    return std::max(0.01, e);
  }
  if (s.op == "ew_binary") {  // {n, op, b_scalar}
    const double n = paramAt(v, 0), bs = paramAt(v, 2);
    const double bytes = 2.0 * (2.0 * n + (bs > 0 ? 0.0 : n));
    return std::max(0.02, smallMemCeiling(n, bytes, eu, clk));
  }
  if (s.op == "ew_binary_bcast") {  // {n, op, b_scalar, C}
    const double n = paramAt(v, 0), C = paramAt(v, 3);
    const double bd = C > 0 ? C : n;  // 通道特化有 C 个不同值；通用版保守按 n 计
    return std::max(0.02, smallMemCeiling(n, 2.0 * (2.0 * n + bd), eu, clk));
  }
  if (s.op == "ew_unary") {  // {n, op}
    const double n = paramAt(v, 0);
    return std::max(0.02, smallMemCeiling(n, 4.0 * n, eu, clk));
  }
  if (s.op == "concat4") {  // {outer, inner, ca, cb, cc, cd}
    const double outer = paramAt(v, 0, 1), inner = paramAt(v, 1, 1);
    const double out = (paramAt(v, 2) + paramAt(v, 3) + paramAt(v, 4) + paramAt(v, 5)) * outer * inner;
    return std::max(0.02, smallMemCeiling(out, 4.0 * out, eu, clk));
  }
  if (s.op == "copy_c") {  // {HW, cnt}
    const double out = paramAt(v, 0) * paramAt(v, 1);
    return std::max(0.02, smallMemCeiling(out, 4.0 * out, eu, clk));
  }
  if (s.op == "slice_axis") {  // {outer, axdim, inner, start, len}
    const double out = paramAt(v, 0, 1) * paramAt(v, 4) * paramAt(v, 2, 1);
    return std::max(0.02, smallMemCeiling(out, 4.0 * out, eu, clk));
  }
  if (s.op == "permute_0213") {  // {D1, D2, I, mode}
    const double out = paramAt(v, 0, 1) * paramAt(v, 1, 1) * paramAt(v, 2, 1);
    return std::max(0.02, smallMemCeiling(out, 4.0 * out, eu, clk));
  }
  if (s.op == "resize_nn") {  // {C, H, W, S}
    const double C = paramAt(v, 0), H = paramAt(v, 1), W = paramAt(v, 2), S = paramAt(v, 3, 1);
    const double out = C * H * S * W * S;
    return std::max(0.02, smallMemCeiling(out, 2.0 * (C * H * W + out), eu, clk));
  }
  if (s.op == "maxpool") {  // {C, H, W, Hout, Wout, K, S, P}
    const double C = paramAt(v, 0), ho = paramAt(v, 3), wo = paramAt(v, 4),
                 K = paramAt(v, 5, 1);
    const double out = C * ho * wo;
    // 每个输出做 K×K 次带边界谓词的读（R30 ISA：0 mad，纯索引/比较指令）。
    return std::max(0.02, smallMemCeiling(out, 2.0 * (out * K * K + out), eu, clk));
  }
  if (s.op == "softmax_axis") {  // {outer, axdim, inner}
    const double n = paramAt(v, 0, 1) * paramAt(v, 1) * paramAt(v, 2, 1);
    // 三趟读（max / exp-sum / 写回读）+ 一趟写 = 8n 字节。
    return std::max(0.02, smallMemCeiling(n, 8.0 * n, eu, clk));
  }
  if (s.op == "gap") {  // 专用字段：Cin = C, N = HW
    const double C = static_cast<double>(s.Cin), HW = static_cast<double>(s.N);
    return std::max(0.01, smallMemCeiling(C, 2.0 * (C * HW + C), eu, clk));
  }
  if (s.op == "bias_add") return 1.0;

  return 1.0;
}

}  // namespace gk
