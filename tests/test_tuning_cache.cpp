// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_tuning_cache —— TuningCache / OpSignature 的离线回归测试。
//
// 覆盖 R42–R48 反复出问题的地方：
//   * 签名稳定性 / 可区分性 / 字符串编码；
//   * 缓存**全字段** round-trip（含 R43 hard_ceiling、R48 数值契约 exact/tol、R45 source hash、ABI）；
//   * cache_abi 不符 -> 整份作废（R47 类静默套用旧 options 的护栏）；
//   * 设备键不匹配 -> 未命中；空设备键 -> 宽松命中；
//   * 禁用 / 缺失文件的回退语义；
//   * JSON 转义（options/config 里带引号与反斜杠）。
//
// 纯 CPU、不需要 GPU；临时文件写在 /tmp/opencode 下。
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "infvino/ClRuntime.hpp"
#include "infvino/Tuning.hpp"
#include "test_util.hpp"

using namespace infvino;

namespace
{
ClDeviceInfo dev()
{
  ClDeviceInfo d;
  d.name             = "TestGT";
  d.eu               = 80;
  d.clock_mhz        = 1300;
  d.subgroup_size    = 16;
  d.pci_vendor_id    = 0x8086;
  d.pci_device_id    = 0x9a49;
  return d;
}

ClDeviceInfo dev_no_pci()
{
  ClDeviceInfo d;
  d.name = "SomeSoftDevice";
  d.eu   = 16;
  d.clock_mhz = 300;
  return d;
}

const char * kPath = "/tmp/opencode/test_tuning_cache.json";
}  // namespace

static void run_tests()
{
  std::filesystem::create_directories("/tmp/opencode");

  // --- 签名稳定性 / 可区分性 ---
  {
    const auto a = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
    const auto b = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
    const auto c = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 0);
    CHECK(a.str() == b.str(), "same params -> same signature");
    CHECK(a.str() != c.str(), "different activation -> different signature");
    CHECK(a.str().find("W40H40") != std::string::npos, "signature encodes spatial dims");
    CHECK(OpSignature::gemm(128, 1600, 192, 0).str().find("M128N1600K192") != std::string::npos,
          "gemm signature encodes M/N/K");
    CHECK(OpSignature::custom("ew_binary", {409600, 0, 0}).str() ==
            OpSignature::custom("ew_binary", {409600, 0, 0}).str(),
          "custom signature stable");
    CHECK(OpSignature::custom("ew_binary", {409600, 0, 0}).str() !=
            OpSignature::custom("ew_binary", {204800, 0, 0}).str(),
          "custom signature encodes params");
  }

  // --- 设备键 ---
  {
    const std::string k1 = TuningCache::deviceKey(dev());
    const std::string k2 = TuningCache::deviceKey(dev_no_pci());
    CHECK(!k1.empty() && !k2.empty(), "device keys non-empty");
    CHECK(k1 != k2, "different devices -> different keys");
    CHECK(TuningCache::deviceKey(dev()) == k1, "device key deterministic");
    CHECK(k2.find("SomeSoftDevice") != std::string::npos, "no-PCI key falls back to name");
  }

  // --- 全字段 round-trip ---
  {
    TuningCache c;
    c.setDeviceId(TuningCache::deviceKey(dev()));
    c.setSourceHash("src-hash-abc");

    const auto sig = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
    TuningEntry e;
    e.kernel       = "conv3x3_ov";
    e.config       = "OBW=8,OBH=1";
    e.options      = "-DOBW=8 -DOBH=1";
    e.ms           = 0.123;
    e.ops          = 8.4;
    e.expected     = 12.0;
    e.ratio        = 0.70;
    e.hard_ceiling = 20.3;
    e.hard_ratio   = 8.4 / 20.3;
    e.iters        = 12;
    e.device_id    = c.deviceId();
    e.source       = "tuned";
    e.exact        = false;
    e.tol          = 0.05;
    c.put(sig, e);

    CHECK(c.save(kPath), "cache save");

    TuningCache in = TuningCache::load(kPath);
    CHECK(in.enabled(), "loaded cache enabled");
    CHECK_EQ(in.abi(), kTuningCacheAbi, "cache_abi round-trip");
    CHECK_EQ(in.sourceHash(), std::string("src-hash-abc"), "kernel source hash round-trip");
    CHECK_EQ(in.deviceId(), c.deviceId(), "device id round-trip");
    CHECK_EQ(in.size(), size_t(1), "entry count round-trip");

    const TuningEntry * le = in.lookup(sig);
    CHECK(le != nullptr, "lookup hit");
    if (le)
    {
      CHECK_EQ(le->kernel, std::string("conv3x3_ov"), "kernel round-trip");
      CHECK_EQ(le->config, std::string("OBW=8,OBH=1"), "config round-trip");
      CHECK_EQ(le->options, std::string("-DOBW=8 -DOBH=1"), "options round-trip");
      CHECK_NEAR(le->ms, 0.123, 1e-9, "ms round-trip");
      CHECK_NEAR(le->ops, 8.4, 1e-9, "ops round-trip");
      CHECK_NEAR(le->expected, 12.0, 1e-9, "expected round-trip");
      CHECK_NEAR(le->ratio, 0.70, 1e-9, "ratio round-trip");
      CHECK_NEAR(le->hard_ceiling, 20.3, 1e-9, "hard ceiling round-trip");
      CHECK_NEAR(le->hard_ratio, 8.4 / 20.3, 1e-6, "hard ratio round-trip");
      CHECK_EQ(le->iters, 12, "iters round-trip");
      CHECK_EQ(le->source, std::string("tuned"), "source round-trip");
      CHECK_EQ(le->exact, false, "R48 numeric contract exact=false round-trip");
      CHECK_NEAR(le->tol, 0.05, 1e-9, "R48 numeric tol round-trip");
    }

    // --- 设备键不匹配 -> 未命中（回退启发式）---
    in.setDeviceId("different_device");
    CHECK(in.lookup(sig) == nullptr, "device mismatch -> miss");

    // --- 空设备键 -> 宽松命中（向后兼容旧缓存）---
    in.setDeviceId("");
    CHECK(in.lookup(sig) != nullptr, "empty device id -> lenient hit");
  }

  // --- JSON 转义：config/options 里带引号与反斜杠 ---
  {
    TuningCache c;
    c.setDeviceId("dev");
    const auto sig = OpSignature::gemm(64, 64, 64, 0);
    TuningEntry e;
    e.kernel    = "gemm_f16";
    e.config    = "weird \"quoted\" cfg";
    e.options   = "-DA=\"x\" -DB=C:\\\\tmp";
    e.device_id = "dev";
    c.put(sig, e);
    CHECK(c.save(kPath), "escaped cache save");
    TuningCache in = TuningCache::load(kPath);
    const TuningEntry * le = in.lookup(sig);
    CHECK(le != nullptr, "escaped lookup hit");
    if (le)
    {
      CHECK_EQ(le->config, e.config, "quoted config round-trip");
      CHECK_EQ(le->options, e.options, "backslash options round-trip");
    }
  }

  // --- hard_ceiling 缺失 -> 回退为 expected（旧缓存兼容）---
  {
    const auto sig = OpSignature::conv1x1(256, 196, 256, 0, 0);
    std::ofstream f(kPath);
    f << "{\n  \"version\": 1,\n  \"cache_abi\": " << kTuningCacheAbi << ",\n"
      << "  \"device_id\": \"d\",\n  \"entries\": {\n"
      << "    \"" << sig.str() << "\": {\"kernel\": \"k\", \"config\": \"c\", \"options\": \"o\", "
         "\"ms\": 1.0, \"ops_per_eu_cyc\": 8.4, \"expected_ops_per_eu_cyc\": 12.0, \"ratio\": 0.7, "
         "\"iters\": 3, \"device_id\": \"d\", \"source\": \"tuned\"}\n"
      << "  }\n}\n";
    f.close();
    TuningCache in = TuningCache::load(kPath);
    const TuningEntry * le = in.lookup(sig);
    CHECK(le != nullptr, "old cache entry loads");
    if (le)
    {
      CHECK_NEAR(le->hard_ceiling, 12.0, 1e-9, "missing hard_ceiling falls back to expected");
      CHECK_NEAR(le->hard_ratio, 8.4 / 12.0, 1e-9, "missing hard_ratio recomputed from ops/expected");
    }
  }

  // --- cache_abi 不符 -> 整份作废（不是错误套用）---
  {
    const auto sig = OpSignature::conv3x3(20, 20, 1, 1, 32, 32, 0);
    std::ofstream f(kPath);
    f << "{\n  \"version\": 1,\n  \"cache_abi\": " << (kTuningCacheAbi + 1) << ",\n"
      << "  \"device_id\": \"d\",\n  \"entries\": {\n"
      << "    \"" << sig.str() << "\": {\"kernel\": \"k\", \"config\": \"c\", \"options\": \"o\", "
         "\"ms\": 1.0, \"ops_per_eu_cyc\": 1.0, \"expected_ops_per_eu_cyc\": 2.0, \"ratio\": 0.5, "
         "\"iters\": 3, \"device_id\": \"d\"}\n"
      << "  }\n}\n";
    f.close();
    TuningCache in = TuningCache::load(kPath);
    CHECK(in.enabled(), "abi-mismatch cache still enabled (fallback, not disabled)");
    CHECK_EQ(in.size(), size_t(0), "abi mismatch -> entries dropped");
    CHECK(in.lookup(sig) == nullptr, "abi mismatch -> miss");
  }

  // --- 缺失文件 / "none" / 环境开关 ---
  {
    TuningCache missing = TuningCache::load("/tmp/opencode/does_not_exist_12345.json");
    CHECK(missing.enabled(), "missing file -> enabled empty cache (fallback)");
    CHECK_EQ(missing.size(), size_t(0), "missing file -> empty");

    TuningCache none = TuningCache::load("none");
    CHECK(!none.enabled(), "'none' -> disabled cache");

    // 写一份有效缓存，再用 INFVINO_TUNING=off 加载
    TuningCache c;
    c.setDeviceId("d");
    c.put(OpSignature::gemm(32, 32, 32, 0), TuningEntry{});
    CHECK(c.save(kPath), "cache save for env test");
    setenv("INFVINO_TUNING", "off", 1);
    TuningCache off = TuningCache::load(kPath);
    unsetenv("INFVINO_TUNING");
    CHECK(!off.enabled(), "INFVINO_TUNING=off -> disabled");
  }

  // --- put 覆盖同 key + setEnabled(false) 永远 miss ---
  {
    TuningCache c;
    c.setDeviceId("d");
    const auto sig = OpSignature::custom("copy_c", {1024, 8});
    TuningEntry e1;
    e1.kernel = "copy_c";
    e1.device_id = "d";
    c.put(sig, e1);
    TuningEntry e2;
    e2.kernel = "copy_c2";
    e2.device_id = "d";
    c.put(sig, e2);
    CHECK_EQ(c.size(), size_t(1), "put overwrites same key");
    CHECK_EQ(c.lookup(sig)->kernel, std::string("copy_c2"), "put overwrote kernel");

    c.setEnabled(false);
    CHECK(c.lookup(sig) == nullptr, "disabled cache always misses");
  }
}

ITEST_MAIN("test_tuning_cache")
