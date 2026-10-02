// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// tuning_test —— 自动调优基础设施的离线自检（**不需要 GPU**）。
//
// 覆盖：签名稳定性、缓存 round-trip、设备键、expected_ops 的单调性/上界。
// 用法：./tuning_test            （成功返回 0）
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include "infvino/Tuning.hpp"

namespace
{
int g_fail = 0;
#define CHECK(cond, msg) do { if (!(cond)) { std::printf("FAIL: %s\n", msg); ++g_fail; } \
                              else std::printf("ok:   %s\n", msg); } while (0)

gk::DeviceInfo dev()
{
  gk::DeviceInfo d;
  d.name = "TestGT"; d.eu = 80; d.clock_mhz = 1300; d.subgroup_size = 16;
  d.pci_vendor_id = 0x8086; d.pci_device_id = 0x9a49;
  return d;
}
}  // namespace

int main()
{
  using namespace gk;

  // --- 签名稳定性/可区分性 ---
  const auto a = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
  const auto b = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 1);
  const auto c = OpSignature::conv3x3(40, 40, 1, 1, 64, 64, 0);
  CHECK(a.str() == b.str(), "same params -> same signature");
  CHECK(a.str() != c.str(), "different act -> different signature");
  CHECK(a.str().find("W40H40") != std::string::npos, "signature encodes spatial");
  CHECK(OpSignature::gemm(128, 1600, 192, 0).str().find("M128N1600K192") != std::string::npos,
        "gemm signature encodes M/N/K");

  // --- 缓存 round-trip ---
  std::filesystem::create_directories("/tmp/opencode");
  const std::string path = "/tmp/opencode/tuning_test_cache.json";
  TuningCache out;
  out.setDeviceId(TuningCache::deviceKey(dev()));
  TuningEntry e;
  e.kernel = "conv3x3_ov"; e.config = "OBW=8,OBH=1"; e.options = "-DOBW=8 -DOBH=1";
  e.ms = 0.123; e.ops = 8.4; e.expected = 12.0; e.ratio = 0.70; e.iters = 12;
  e.device_id = out.deviceId(); e.source = "tuned";
  out.put(a, e);
  CHECK(out.save(path), "cache save");
  TuningCache in = TuningCache::load(path);
  CHECK(in.size() == 1, "cache load count");
  const TuningEntry * le = in.lookup(a);
  CHECK(le != nullptr, "cache lookup hit");
  if (le) {
    CHECK(le->kernel == "conv3x3_ov", "cache kernel round-trip");
    CHECK(le->config == "OBW=8,OBH=1", "cache config round-trip");
    CHECK(le->options == "-DOBW=8 -DOBH=1", "cache options round-trip");
    CHECK(le->iters == 12, "cache iters round-trip");
    CHECK(le->ratio > 0.69 && le->ratio < 0.71, "cache ratio round-trip");
  }
  in.setDeviceId("different_device_eu16_clk300");
  CHECK(in.lookup(a) == nullptr, "device key mismatch -> miss（回退启发式）");

  // --- expected_ops：单调 + 上界 ---
  DeviceInfo d = dev();
  const double big = expectedOps(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1), d);
  const double small = expectedOps(OpSignature::conv3x3(20, 20, 1, 1, 64, 64, 1), d);
  CHECK(big > small, "larger grid -> higher expected (grid factor)");
  CHECK(big <= kConvStagingFreeCeiling + 1e-9, "conv expected <= staging-free ceiling");
  const double g = expectedOps(OpSignature::gemm(1024, 1024, 1024, 0), d);
  CHECK(g > 5.0 && g <= 17.7 + 1e-9, "gemm expected within (5, compute-only 17.7]");
  CHECK(expectedOps(c, d) >= 1.0, "expected has a positive floor");

  TuningCache disabled;
  disabled.setEnabled(false);
  CHECK(disabled.lookup(a) == nullptr, "disabled cache always misses");

  std::printf("\n%s (%d failures)\n", g_fail ? "TUNING TEST FAILED" : "TUNING TEST PASSED", g_fail);
  return g_fail ? 1 : 0;
}
