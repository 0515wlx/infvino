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

namespace infvino
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
  } else if (op == "conv1x1_cat4") {
    o << "|Cout" << Cout << "_N" << N << "_Cin" << Cin << "_cat";
    for (size_t i = 0; i < params.size(); ++i) { if (i) o << ","; o << params[i]; }
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
OpSignature OpSignature::conv1x1Cat4(int Cout, int N, int Cin, int ca, int cb, int cc, int cd,
                                     const int *off, int act, int res)
{
  OpSignature s;
  s.op = "conv1x1_cat4"; s.Cout = Cout; s.N = N; s.Cin = Cin; s.act = act;
  // 分段 + 每段的源内 offset（offset 只影响取址，不影响性能；不入签名以复用调优项）。
  s.params = {ca, cb, cc, cd};
  (void)off;
  if (res) s.groups = 2;
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

std::string TuningCache::deviceKey(const ClDeviceInfo & d)
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
  // R45 P1#7: kernel 源码指纹（数值/编译契约守卫）。缺失=旧缓存，视为未标注。
  {
    bool sf = false;
    const std::string sh = findField(txt, "kernel_src_hash", &sf);
    if (sf) c.source_hash_ = sh;
  }
  // kernel ABI 守卫：文件标注了 abi 且与当前不符 → 整份缓存作废（未命中），
  // 避免 kernel 宏语义变化后静默套用旧 options。未标注=0 视为兼容（接受）。
  {
    bool af = false;
    const std::string abi = findField(txt, "cache_abi", &af);
    if (af && !abi.empty()) {
      c.abi_ = std::atoi(abi.c_str());
      if (c.abi_ != kTuningCacheAbi) return c;   // enabled_ 仍为 true → 全部 lookup 未命中
    }
  }

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
    // R43: 硬上限字段（向后兼容：旧缓存没有则回退为软值）。
    e.hard_ceiling = std::atof(findField(body, "hard_ceiling_ops_per_eu_cyc", &f7).c_str());
    e.hard_ratio   = std::atof(findField(body, "hard_ratio", &f7).c_str());
    if (e.hard_ceiling <= 0.0) e.hard_ceiling = e.expected;
    if (e.hard_ratio <= 0.0 && e.hard_ceiling > 0.0) e.hard_ratio = e.ops / e.hard_ceiling;
    e.iters    = std::atoi(findField(body, "iters", &f7).c_str());
    e.device_id = findField(body, "device_id", &f7);
    e.source    = findField(body, "source", &f7);
    // R48 §3.2: 数值契约（向后兼容：旧缓存无此字段 -> 视为逐位一致）。
    {
      bool fe = false, ft = false;
      const std::string ex = findField(body, "numeric_exact", &fe);
      if (fe) e.exact = (ex == "false" || ex == "0") ? false : true;
      const std::string tl = findField(body, "numeric_tol", &ft);
      if (ft) e.tol = std::atof(tl.c_str());
    }
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
  f << "  \"cache_abi\": " << kTuningCacheAbi << ",\n";
  f << "  \"device_id\": \"" << jsonEscape(device_id_) << "\",\n";
  f << "  \"kernel_src_hash\": \"" << jsonEscape(source_hash_) << "\",\n";
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
      << "\"hard_ceiling_ops_per_eu_cyc\": " << e.hard_ceiling << ", "
      << "\"hard_ratio\": " << e.hard_ratio << ", "
      << "\"iters\": " << e.iters << ", "
      << "\"device_id\": \"" << jsonEscape(e.device_id) << "\", "
      << "\"source\": \"" << jsonEscape(e.source) << "\", "
      << "\"numeric_exact\": " << (e.exact ? "true" : "false") << ", "
      << "\"numeric_tol\": " << e.tol
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
// R47-L3: 锁频重测的 copy(read+write) 带宽-足迹曲线（本次会话；比 R30 旧曲线准且覆盖到 16MB）。
// 膝点在 3→4MB（L3≈3.75MB）：145 → 111 → **56** → 35 → 28 → 22 → 20 GB/s。
const BwPoint kBwCurve[] = {
  {4e3, 3.0}, {16e3, 8.1}, {64e3, 22.7}, {256e3, 46.4}, {512e3, 63.3},
  {1e6, 145.4}, {2e6, 138.8}, {3e6, 111.5}, {4e6, 56.1}, {5e6, 35.5},
  {6e6, 28.2}, {8e6, 22.2}, {12e6, 20.7}, {16e6, 20.3}};
constexpr double kSmallLaunchUsInternal = 3.5;
// R47-L3: 有效 L3 容量（由膝点推断；用于图级溢出估计）。
constexpr double kL3Bytes = 3.75e6;

// 由足迹插值 copy 带宽（log-log 线性；端点外取端点值）。
double copyBwGbpsImpl(double footprint)
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

// R47-L3: 通用内存 roofline（ops/EU/cyc 量纲，flops/(EU·clk·t)）。把「足迹跨 L3 断崖」
// 显式建模进上限：t = launch + bytes/BW(footprint)。比 single-BW roofline 更接近真实，
// 且在 L3 断崖处自然给出更低的「可达上限」。
// R66: `bwGbps` 由调用方给出——可带并发/容量二维因子（`effectiveBwGbps`）。
double memRooflineOps(double flops, double bytes, int eu, double clkMhz, double launchUs,
                      double bwGbps)
{
  if (flops <= 0.0 || bytes <= 0.0 || bwGbps <= 0.0) return 1e30;
  const double bw = bwGbps * 1e9;
  const double t = launchUs * 1e-6 + bytes / bw;
  if (t <= 0.0) return 1e30;
  return flops / (static_cast<double>(eu) * clkMhz * 1e6 * t);
}

// 内存受限算子的期望 proxy（= 2·out 元素 / EU / cyc，与 autotune 的 `ops` 同量纲）。
double smallMemCeiling(double outElems, double bytes, int eu, double clkMhz)
{
  if (outElems <= 0.0 || bytes <= 0.0) return 0.0;
  const double bw = copyBwGbpsImpl(bytes) * 1e9;  // bytes/s
  const double t  = kSmallLaunchUsInternal * 1e-6 + bytes / bw;
  return 2.0 * outElems / (static_cast<double>(eu) * clkMhz * 1e6 * t);
}

double paramAt(const std::vector<int> & v, size_t i, double dflt = 0.0)
{
  return i < v.size() ? static_cast<double>(v[i]) : dflt;
}
}  // namespace

// R47: 导出 copy 带宽插值（外层可见）：转发到匿名命名空间里的同名实现。
double copyBwGbps(double footprint) { return copyBwGbpsImpl(footprint); }

/** @brief R47 标定：L3 miss 折算（1/BW_DRAM − 1/BW_L3），ms/byte。
 *  R59：BW 用实测平台/峰值（DRAM 20 GB/s、L3 copy 峰值 145 GB/s）。*/
const double kL3SpillPerByteMs = (1.0 / 20e9 - 1.0 / 145e9) * 1e3;

// R62/R63: 内存层次锚点（见 Tuning.hpp）。
//   GPU 私有 L3 Data Cache = 8 bank × 480 KiB = 3.75 MiB = 512 set × 120 way × 64 B。
//   共享 LLC（= CPU sysfs index3）= 8 MiB（TGL-U 4 核；TGL-H 8 核 24 MiB）。
//   两级：GPU L3 → LLC → DRAM。跨算子热重用有效容量 ≈ L3 + LLC ≈ 12.3 MB（R63 别名实测 ~12MB）。
double l3PhysicalBytes() { return 3932160.0; }          // 3.75 MiB (GPU private L3)
double l3CpuL3Bytes() { return 8388608.0; }             // 8 MiB (shared LLC, sysfs)
double l3LlcBytes() { return 8388608.0; }               // 8 MiB (shared LLC)
double l3WarmCapBytes() { return l3PhysicalBytes() + l3LlcBytes(); }   // ≈12.3 MB cross-op
double l3PrivateCapBytes() { return 2.0e6; }
double l3DefaultCapBytes()
{
  const char * geom = std::getenv("INFVINO_L3_GEOM");
  return (geom && std::string(geom) != "0" && std::string(geom) != "") ? l3PhysicalBytes()
                                                                      : l3WarmCapBytes();
}
double l3DefaultAnchorBytes()
{
  const char * geom = std::getenv("INFVINO_L3_GEOM");
  return (geom && std::string(geom) != "0" && std::string(geom) != "") ? 1.0e6
                                                                      : l3PrivateCapBytes();
}
double l3DramBwGbps() { return 20.0; }
double l3SramBwGbps() { return 145.0; }

// R60: reorder 的**可加**成本模型。常数由本机 `kernel_bench --op reorder` 标定
// （Cin16 322² 等 shape 实测 ~58–60 GB/s 单趟；launch floor ≈ 3.5 µs）。见 docs/round60-*。
const double kReorderLaunchMs = 0.0035;
const double kReorderStreamBwGbps = 60.0;
// R60: 顺序 dispatch 的逐次间隔（锁频实测 ~12.4 µs，见 docs/round60-*）。事件计时的单趟
// `#reorder.ms` 不含它；整网把 N 趟相加会少算 N×间隙——这正是「reorder 不可加」的第一项。
const double kReorderDispatchGapMs = 0.0124;
double reorderCostMs(double read_bytes, double write_bytes, bool input_resident)
{
  const double total = std::max(0.0, read_bytes) + std::max(0.0, write_bytes);
  const double bw = input_resident ? l3SramBwGbps() : kReorderStreamBwGbps;
  return kReorderLaunchMs + (bw > 0.0 ? total / (bw * 1e6) : 0.0);
}

namespace
{
// R55: 占用统计的**单一真相源**（occupancyPressure / occupancyThreads 共用）。
// 出参 concurrent = min(总线程数, 8192)、perWg = 每 WG 触达字节；返回 false = 该 op 不计占用。
bool occupancyStats(const TuningEntry & e, const OpSignature & s, double & concurrent, double & perWg)
{
  auto optIntOf = [](const std::string & opts, const char * key, int def) -> int {
    const std::string k(key);
    const auto p = opts.find(k);
    if (p == std::string::npos) return def;
    return std::atoi(opts.c_str() + p + k.size());
  };
  double totalThreads = 1.0;
  perWg = 0.0;
  int    wgSize = 64;
  if (s.op == "conv3x3")
  {
    const int obw = optIntOf(e.options, "-DOBW=", s.stride == 2 ? 5 : 8);
    const int obh = optIntOf(e.options, "-DOBH=", s.stride == 2 ? 4 : 2);
    const int slm = std::max(1, optIntOf(e.options, "-DSLM_DIV=", 1));
    wgSize = 16 * slm;
    const double nwg = static_cast<double>((s.W + obw - 1) / obw) * ((s.H + obh - 1) / obh) *
                       ((s.Cout + 15) / 16);
    totalThreads = nwg * wgSize;
    perWg = 2.0 * s.Cin * (obw + 2.0) * (obh + 2.0);
  }
  else if (s.op == "conv1x1" && e.kernel == "conv1x1_blk")
  {
    const int xb = std::max(1, optIntOf(e.options, "-DX_BLOCK=", 4));
    const int yb = std::max(1, optIntOf(e.options, "-DY_BLOCK=", 1));
    const int slm = std::max(1, optIntOf(e.options, "-DSLM_DIV=", 1));
    wgSize = 16 * slm;
    const int tile = xb * yb;
    const double nwg = static_cast<double>((s.N + tile - 1) / tile) *
                       static_cast<double>((s.Cout + 15) / 16);
    totalThreads = nwg * wgSize;
    perWg = 2.0 * s.Cin * tile;
  }
  else if (s.op == "depthwise" && e.kernel == "depthwise_blk")
  {
    // R50: 与 depthwise_blk 的几何一致（SG=16 lane=通道、XB 列、YB 行）。签名 W/H 为输出空间。
    const int xb = std::max(1, optIntOf(e.options, "-DX_BLOCK=", 8));
    const int yb = std::max(1, optIntOf(e.options, "-DY_BLOCK=", 1));
    wgSize = 16;
    const double nwg = static_cast<double>((s.W + xb - 1) / xb) *
                       static_cast<double>((s.H + yb - 1) / yb) *
                       static_cast<double>((s.Cout + 15) / 16);
    totalThreads = nwg * wgSize;
    // 每 WG 触碰的输入 tile：16 通道 × 列 span × 行 span × 2B。
    const double colSpan = (xb - 1) * s.stride + s.K;
    const double rowSpan = (yb - 1) * s.stride + s.K;
    perWg = 2.0 * 16.0 * colSpan * rowSpan;
  }
  else if (s.op == "conv1x1" || s.op == "gemm" || s.op == "conv1x1_cat4")
  {
    const int M = (s.op == "gemm") ? s.M : s.Cout;
    const int N = s.N;
    const int bm = optIntOf(e.options, "-DBM=", 64);
    const int bn = optIntOf(e.options, "-DBN=", 64);
    const int bk = optIntOf(e.options, "-DBK=", 16);
    const int tn = optIntOf(e.options, "-DTN=", 4);
    wgSize = bn / tn;
    const double nwg = static_cast<double>((M + bm - 1) / bm) * ((N + bn - 1) / bn);
    totalThreads = nwg * wgSize;
    perWg = 2.0 * (static_cast<double>(bm) * bk + static_cast<double>(bn) * bk);
  }
  else
    return false;  // 小算子：占用并入其自身实测（launch/带宽受限）
  constexpr double kSatThreads = 8192.0;  // 实测：与 WG/寄存器无关的线程数上限
  concurrent = std::min(totalThreads, kSatThreads);
  return true;
}
}  // namespace

double occupancyPressure(const TuningEntry & e, const OpSignature & s)
{
  double c = 0.0, p = 0.0;
  if (!occupancyStats(e, s, c, p)) return 0.0;
  return c * p;
}

double occupancyThreads(const TuningEntry & e, const OpSignature & s)
{
  double c = 0.0, p = 0.0;
  if (!occupancyStats(e, s, c, p)) return 0.0;
  return c;
}

// R55 校准（`l3couple`）：并发/MLP 因子。有效行吞吐随线程数上升、约 ~1024 WI 饱和。
double l3MlpFactor(double threads)
{
  if (threads <= 0.0) return 0.05;
  constexpr double kMlpSatThreads = 1024.0;
  return std::max(0.05, std::min(1.0, threads / kMlpSatThreads));
}

// R55 校准（`l3couple`）：L3 容量因子 g(R)。私有 tile 形态：平台 ~2 MB，半坡 ~6 MB，
// DRAM 平台 ~0.04（见 docs/round55-l3-coupling-calibration.md §2.3）。
double l3CapacityFactor(double residentBytes)
{
  struct P { double bytes, f; };
  static const P kCurve[] = {
    {0.0, 1.0},    {2.0e6, 1.0},  {3.0e6, 0.80}, {4.0e6, 0.70},  {6.0e6, 0.55},
    {8.0e6, 0.48}, {12.0e6, 0.28}, {16.0e6, 0.15}, {24.0e6, 0.08}, {32.0e6, 0.05},
    {64.0e6, 0.04}};
  const int n = static_cast<int>(sizeof(kCurve) / sizeof(kCurve[0]));
  if (residentBytes <= kCurve[1].bytes) return kCurve[1].f;
  if (residentBytes >= kCurve[n - 1].bytes) return kCurve[n - 1].f;
  for (int i = 1; i + 1 < n; ++i)
    if (residentBytes <= kCurve[i + 1].bytes)
    {
      const double lx = std::log(residentBytes / kCurve[i].bytes) /
                        std::log(kCurve[i + 1].bytes / kCurve[i].bytes);
      return kCurve[i].f * std::pow(kCurve[i + 1].f / kCurve[i].f, lx);
    }
  return kCurve[n - 1].f;
}

double effectiveBwGbps(double bytes, double threads)
{
  return copyBwGbps(bytes) * l3MlpFactor(threads);
}

// R67: 单遍内存字节（与 expectedOps 足迹口径一致）。
double singlePassBytes(const OpSignature & s)
{
  if (s.op == "conv3x3")
  {
    const double span = s.stride >= 2 ? static_cast<double>(s.stride) : 1.0;
    return 2.0 * s.Cin * (s.H * span) * (s.W * span) + 2.0 * s.Cout * s.H * s.W +
           2.0 * s.Cout * s.Cin * 9.0;
  }
  if (s.op == "gemm")
    return 2.0 * (static_cast<double>(s.M) * s.K + static_cast<double>(s.K) * s.N +
                  static_cast<double>(s.M) * s.N);
  if (s.op == "conv1x1" || s.op == "conv1x1_cat4")
    return singlePassBytes(OpSignature::gemm(s.Cout, s.N, s.Cin, s.act));
  if (s.op == "depthwise" || s.op == "conv_general")
    return 2.0 * s.Cin * s.H * s.W + 2.0 * s.Cout * s.H * s.W + 2.0 * s.Cin * 9.0;
  return 0.0;
}

// R67: 容量感知内存惩罚（ms）——只补「有效容量不足导致的额外内存时间」。
double capacityMemPenaltyMs(const TuningEntry & e, const OpSignature & s)
{
  const double R = occupancyPressure(e, s);
  if (R <= 0.0) return 0.0;
  const double g = l3CapacityFactor(R);
  if (g >= 1.0) return 0.0;
  const double bytes = singlePassBytes(s);
  if (bytes <= 0.0) return 0.0;
  const double bw = effectiveBwGbps(bytes, occupancyThreads(e, s));  // GB/s
  if (bw <= 0.0) return 0.0;
  const double tMemMs = bytes / (bw * 1e6);
  return (1.0 / g - 1.0) * tMemMs;
}

// R47: 导出的 launch 地板（供 PlanModel 的小算子外溢估计使用）。
const double kSmallLaunchUs = 3.5;

double expectedOps(const OpSignature & s, const ClDeviceInfo & dev, const TuningEntry * e)
{
  const int eu = dev.eu > 0 ? static_cast<int>(dev.eu) : 80;
  const double clk = dev.clock_mhz > 0 ? static_cast<double>(dev.clock_mhz) : 1300.0;
  // R66: 内存 roofline 的带宽——带候选时用二维有效带宽（并发 MLP × 足迹容量因子）；
  // 否则退回 copy 曲线（旧行为，保证无候选的调用点不变）。并发为 0 的 op 也用 copy。
  auto bwAt = [&](double bytes) -> double {
    if (e && !std::getenv("INFVINO_NO_L3_ROOFLINE_2D"))
    {
      const double th = occupancyThreads(*e, s);
      if (th > 0.0) return effectiveBwGbps(bytes, th);
    }
    return copyBwGbps(bytes);
  };

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
    // R47-L3: 叠加内存 roofline（足迹跨 L3 断崖的层会被正确压到内存墙之下）。
    {
      const double flops = 2.0 * s.Cout * s.H * s.W * s.Cin * 9.0;
      const double span = s.stride >= 2 ? static_cast<double>(s.stride) : 1.0;
      const double inBytes = 2.0 * s.Cin * (s.H * span) * (s.W * span);
      const double outBytes = 2.0 * s.Cout * s.H * s.W;
      const double wBytes = 2.0 * s.Cout * s.Cin * 9.0;
      const double memBytes = inBytes + outBytes + wBytes;
      e = std::min(e, memRooflineOps(flops, memBytes, eu, clk, 0.0, bwAt(memBytes)));
    }
    return std::max(1.0, e);
  }

  if (s.op == "gemm") {
    // R32: 修正 GEMM 中间标准。旧式把 grid 按 BM=128 估（生产 kernel 用 BM=64），
    // 且对「小网格」套 gridFactor(grid/160) —— 但 GEMM 每个 WG 有 K 长度的归约工作，
    // 即便 grid 小，EU 流水仍能保持较高占用。旧期望系统性偏低 ~1.4–2.2×，导致
    // tuning.json 里一批 conv1x1 的 ratio > 1（对「上限」模型自相矛盾），无法用于热点定位。
    // 新式：grid 用实际 BM=64/BN=64；~64 WG 即可喂饱 80 EU；K 太短时 prologue/epilogue
    // 摊薄不足。上界仍为整核 13.7（R14），与 conv3x3 的 §7 模型同思路。
    const long grid = ((s.M + 63) / 64) * ((s.N + 63) / 64);
    double e = 13.7;  // 整核 compute+staging 上限（R14）
    e *= std::min(1.0, static_cast<double>(grid) / 64.0);   // 波量化：~64 WG 喂饱 80 EU
    e *= static_cast<double>(s.K) / (static_cast<double>(s.K) + 64.0);  // 短 K 的 prologue 亏
    // R47-L3: 叠加内存 roofline（大 spatial / 窄 K 的 GEMM 会撞 DRAM 墙）。
    {
      const double flops = 2.0 * s.M * s.N * s.K;
      const double bytes = 2.0 * (static_cast<double>(s.M) * s.K + static_cast<double>(s.K) * s.N +
                                  static_cast<double>(s.M) * s.N);
      e = std::min(e, memRooflineOps(flops, bytes, eu, clk, 0.0, bwAt(bytes)));
    }
    return std::max(0.5, e);
  }

  if (s.op == "conv1x1_cat4") {
    // R30c: 融合 concat；计算仍是 M=Cout,N=HW,K=Cin 的 GEMM（另省 concat 物化）。
    return expectedOps(OpSignature::gemm(s.Cout, s.N, s.Cin, s.act), dev, e);
  }

  if (s.op == "conv1x1") {
    if (s.N == 1) {
      // split-K GEMV（R22）：lane 沿 K 归约，fp32 累加，受归约延迟 / K 长度限制。
      // 实测 5–27× 于旧 gemm；期望取「每 EU 一条 sub-group 归约链」的保守值。
      return std::max(1.0, 8.0 * gridFactor(static_cast<long>(s.Cout), eu));
    }
    return expectedOps(OpSignature::gemm(s.Cout, s.N, s.Cin, s.act), dev, e);
  }

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

}  // namespace infvino
