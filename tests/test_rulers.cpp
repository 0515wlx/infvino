// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_rulers —— 「中间标准 / 物理模型」的离线回归测试（R30/R47/R49）。
//
// 这些标尺驱动 autotune 排序与布局决策，公式一坏就静默地把选择带偏（R42 记账 bug 类）。
// 覆盖：
//   * expectedOps 对全部代表算子有限、为正；
//   * conv3x3 的网格单调性 / 指令配额上界；gemm / depthwise / bmm 的量级区间；
//   * copyBwGbps 足迹插值：端点钳制 + 膝点两侧单调；
//   * occupancyPressure：conv3x3 随 SLM 上升、gemm 为正、小算子为 0。
//
// 纯 CPU、不需要 GPU。
#include <cmath>
#include <string>
#include <vector>

#include "infvino/KernelFamily.hpp"
#include "infvino/L3Model.hpp"
#include "infvino/Tuning.hpp"
#include "test_util.hpp"

using namespace infvino;

namespace
{
ClDeviceInfo dev()
{
  ClDeviceInfo d;
  d.name = "TestGT";
  d.eu = 80;
  d.clock_mhz = 1300;
  d.subgroup_size = 16;
  d.pci_vendor_id = 0x8086;
  d.pci_device_id = 0x9a49;
  return d;
}
}  // namespace

static void run_tests()
{
  const ClDeviceInfo d = dev();

  // --- expectedOps：全体代表算子有限且为正 ---
  {
    std::vector<OpSignature> sigs;
    sigs.push_back(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1));
    sigs.push_back(OpSignature::conv3x3(160, 160, 2, 1, 3, 16, 1));
    sigs.push_back(OpSignature::gemm(1024, 1024, 1024, 0));
    sigs.push_back(OpSignature::conv1x1(256, 196, 256, 0, 0));
    sigs.push_back(OpSignature::conv1x1(1000, 1, 1024, 0, 0));
    sigs.push_back(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 1));
    sigs.push_back(OpSignature::gap(64, 784));
    sigs.push_back(OpSignature::custom("ew_binary", {409600, 2, 0}));
    sigs.push_back(OpSignature::custom("copy_c", {25600, 16}));
    sigs.push_back(OpSignature::custom("bmm", {1, 2, 64, 400, 400}));
    sigs.push_back(OpSignature::custom("softmax_axis", {1, 16, 33600}));
    sigs.push_back(OpSignature::custom("maxpool", {64, 80, 80, 40, 40, 2, 2, 0}));
    sigs.push_back(OpSignature::custom("resize_nn", {64, 40, 40, 2}));
    sigs.push_back(OpSignature::custom("permute_0213", {1, 64, 400, 0}));
    sigs.push_back(OpSignature::custom("concat4", {1, 64, 64, 64, 64, 64}));

    for (const auto & s : sigs)
    {
      const double e = expectedOps(s, d);
      const std::string msg = "expectedOps finite/positive for " + s.str();
      CHECK(std::isfinite(e) && e > 0.0, msg.c_str());
    }
  }

  // --- conv3x3：大网格 > 小网格；不超 OV 指令配额上界 ---
  {
    const double big = expectedOps(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1), d);
    const double small = expectedOps(OpSignature::conv3x3(20, 20, 1, 1, 64, 64, 1), d);
    CHECK(big > small, "conv3x3: larger grid -> higher expected (grid factor)");
    CHECK(big <= kConvOvIssueCeiling + 1e-9, "conv3x3: expected <= OV issue ceiling");
    CHECK(big > kConvStagingFreeCeiling, "conv3x3: middle standard anchored above old ~16");
  }

  // --- gemm / depthwise / bmm 量级区间 ---
  {
    const double g = expectedOps(OpSignature::gemm(1024, 1024, 1024, 0), d);
    CHECK(g > 5.0 && g <= 17.7 + 1e-9, "gemm expected within (5, compute-only 17.7]");
    const double dw = expectedOps(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 1), d);
    CHECK(dw > 0.5 && dw < 4.0, "depthwise expected = ISA quota (~2.8)");
    const double bm = expectedOps(OpSignature::custom("bmm", {1, 2, 64, 400, 400}), d);
    CHECK(bm > 0.0 && bm < 0.1, "bmm expected is K/grid bounded");
    const double eb = expectedOps(OpSignature::custom("ew_binary_bcast", {285600, 2, 0, 0}), d);
    CHECK(eb > 0.05 && eb < 1.0, "bcast expected in memory-roofline range");
  }

  // --- hard ceiling >= 软 expected（同一 shape）---
  {
    const OpSignature cs = OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1);
    const KernelFamily * ov = familyByName("conv3x3_ov");
    CHECK(ov != nullptr && static_cast<bool>(ov->hardCeiling), "conv3x3_ov declares a hard ceiling");
    if (ov && ov->hardCeiling)
    {
      const double hard = ov->hardCeiling(cs, d);
      CHECK(hard > 0.0 && expectedOps(cs, d) <= hard + 1e-9,
            "expected <= hard ceiling for conv3x3_ov");
      CHECK_NEAR(hard, kConvOvIssueCeiling, 1e-9, "conv3x3_ov hard ceiling == ISA quota");
    }
    const double best = bestFamilyCeiling(cs, d);
    CHECK(best > 0.0 && best <= kConvOvIssueCeiling + 1e-9,
          "bestFamilyCeiling <= conv3x3 OV issue ceiling");
  }

  // --- copyBwGbps：端点钳制 + 曲线单调 ---
  {
    CHECK_NEAR(copyBwGbps(1.0), 3.0, 1e-9, "copyBw below first point clamps to 3.0");
    CHECK_NEAR(copyBwGbps(4e3), 3.0, 1e-9, "copyBw at first point == 3.0");
    CHECK_NEAR(copyBwGbps(1e6), 145.4, 1e-6, "copyBw at 1MB peak == 145.4");
    CHECK_NEAR(copyBwGbps(16e6), 20.3, 1e-6, "copyBw at 16MB == 20.3");
    CHECK_NEAR(copyBwGbps(1e12), 20.3, 1e-9, "copyBw above last point clamps to 20.3");

    const double rising[] = {4e3, 16e3, 64e3, 256e3, 512e3, 1e6};
    for (size_t i = 1; i < sizeof(rising) / sizeof(rising[0]); ++i)
      CHECK(copyBwGbps(rising[i]) > copyBwGbps(rising[i - 1]),
            "copyBw rising段单调递增");

    const double falling[] = {1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 8e6, 12e6, 16e6};
    for (size_t i = 1; i < sizeof(falling) / sizeof(falling[0]); ++i)
      CHECK(copyBwGbps(falling[i]) < copyBwGbps(falling[i - 1]),
            "copyBw falling段单调递减");
  }

  // --- occupancyPressure ---
  {
    const OpSignature cs = OpSignature::conv3x3(16, 16, 1, 1, 8, 16, 1);
    TuningEntry e;
    e.kernel = "conv3x3_ov";
    e.options = "-DOBW=8 -DOBH=2 -DSLM_DIV=1";
    const double p1 = occupancyPressure(e, cs);
    e.options = "-DOBW=8 -DOBH=2 -DSLM_DIV=4";
    const double p4 = occupancyPressure(e, cs);
    CHECK(p1 > 0.0, "conv3x3 occupancy pressure positive");
    CHECK(p4 > p1, "conv3x3 occupancy pressure grows with SLM");

    TuningEntry g;
    g.kernel = "gemm_f16";
    g.options = "-DBM=64 -DBN=64 -DBK=16 -DTN=4";
    CHECK(occupancyPressure(g, OpSignature::gemm(512, 400, 256, 0)) > 0.0,
          "gemm occupancy pressure positive");

    TuningEntry s;
    s.kernel = "copy_c";
    CHECK_NEAR(occupancyPressure(s, OpSignature::custom("copy_c", {25600, 16})), 0.0, 1e-12,
               "small-op occupancy pressure is 0 (launch/bandwidth bounded)");
  }

  // --- R55: l3MlpFactor（并发/MLP 因子）---
  {
    CHECK_NEAR(l3MlpFactor(0.0), 0.05, 1e-12, "mlp factor floor at 0 threads");
    CHECK_NEAR(l3MlpFactor(1024.0), 1.0, 1e-12, "mlp factor saturates at 1024 threads");
    CHECK_NEAR(l3MlpFactor(8192.0), 1.0, 1e-12, "mlp factor capped at 1");
    const double rising[] = {64.0, 128.0, 256.0, 512.0, 1024.0, 2048.0};
    for (size_t i = 1; i < sizeof(rising) / sizeof(rising[0]); ++i)
      CHECK(l3MlpFactor(rising[i]) >= l3MlpFactor(rising[i - 1]), "mlp factor monotone nondecreasing");
    CHECK(l3MlpFactor(512.0) < l3MlpFactor(1024.0), "mlp factor rises before saturation");
  }

  // --- R55: l3CapacityFactor g(R) ---
  {
    CHECK_NEAR(l3CapacityFactor(0.0), 1.0, 1e-12, "capacity factor = 1 for tiny footprint");
    CHECK_NEAR(l3CapacityFactor(1.0e6), 1.0, 1e-12, "capacity factor plateau <= ~2MB");
    CHECK_NEAR(l3CapacityFactor(2.0e6), 1.0, 1e-12, "capacity factor plateau at 2MB");
    CHECK_NEAR(l3CapacityFactor(64e6), 0.04, 1e-9, "capacity factor clamps at DRAM plateau");
    const double falling[] = {2e6, 3e6, 4e6, 6e6, 8e6, 12e6, 16e6, 24e6, 32e6, 64e6};
    for (size_t i = 1; i < sizeof(falling) / sizeof(falling[0]); ++i)
      CHECK(l3CapacityFactor(falling[i]) <= l3CapacityFactor(falling[i - 1]) + 1e-12,
            "capacity factor monotone nonincreasing past plateau");
    CHECK(l3CapacityFactor(2e6) > l3CapacityFactor(8e6), "capacity factor drops past the knee");
    CHECK(l3CapacityFactor(8e6) > l3CapacityFactor(32e6), "capacity factor keeps dropping");
  }

  // --- R55: effectiveBwGbps = copy × mlp ---
  {
    const double bytes = 1e6, thr = 512.0;
    CHECK_NEAR(effectiveBwGbps(bytes, thr), copyBwGbps(bytes) * l3MlpFactor(thr), 1e-9,
               "effectiveBw == copyBw * mlpFactor");
    CHECK(effectiveBwGbps(bytes, thr) <= copyBwGbps(bytes) + 1e-9,
          "effectiveBw never exceeds the footprint curve");
    CHECK(effectiveBwGbps(bytes, 4096.0) > effectiveBwGbps(bytes, 256.0),
          "effectiveBw rises with concurrency");
  }

  // --- R62: L3 几何锚点（GPU L3 = 3.75 MiB，非 R59 误用的 CPU 8 MiB）---
  {
    CHECK_NEAR(l3PhysicalBytes(), 3932160.0, 1.0, "R62: GPU L3 = 3.75 MiB (8 bank x 480 KiB)");
    CHECK_NEAR(l3PhysicalBytes(), 512.0 * 120.0 * 64.0, 1.0, "R62: GPU L3 = 512 set x 120 way x 64B");
    CHECK_NEAR(l3CpuL3Bytes(), 8388608.0, 1.0, "R62: CPU L3 = 8 MiB (reference)");
    CHECK(l3PhysicalBytes() < l3CpuL3Bytes(), "R62: GPU L3 < CPU L3 (R59 had used the CPU value)");
    CHECK_NEAR(l3PrivateCapBytes(), 2.0e6, 1.0, "R62: private-tile cap knee = 2 MB (R55)");
    CHECK_NEAR(l3DefaultCapBytes(), l3PhysicalBytes(), 1.0, "R62: default cap == GPU L3 (geometry)");
    CHECK_NEAR(l3DefaultAnchorBytes(), l3PrivateCapBytes(), 1.0, "R62: default anchor == private cap");
    CHECK(l3SramBwGbps() > l3DramBwGbps(), "R62: SRAM BW > DRAM BW");
    // legacy 开关（A/B）：置 1 时回退 R59 的 CPU 值 8 MiB / 1 MiB。
    setenv("INFVINO_L3_LEGACY", "1", 1);
    CHECK_NEAR(l3DefaultCapBytes(), l3CpuL3Bytes(), 1.0, "R62: legacy cap = CPU L3 (8 MiB)");
    CHECK_NEAR(l3DefaultAnchorBytes(), 1.0e6, 1.0, "R62: legacy anchor = 1 MB");
    unsetenv("INFVINO_L3_LEGACY");
    CHECK_NEAR(l3DefaultCapBytes(), l3PhysicalBytes(), 1.0, "R62: back to geometry after unset");
  }

  // --- R60: reorder 可加成本 = launch floor + 传输/BW(state) + L3 策略解析 ---
  {
    CHECK_NEAR(reorderCostMs(0.0, 0.0, false), kReorderLaunchMs, 1e-12,
               "R60: reorder cost = launch floor at zero bytes");
    const double cold = reorderCostMs(2.0e6, 2.0e6, false);
    const double hot = reorderCostMs(2.0e6, 2.0e6, true);
    CHECK(cold > hot, "R60: L3-resident input is cheaper than cold input");
    CHECK_NEAR(cold, kReorderLaunchMs + 4.0e6 / (kReorderStreamBwGbps * 1e6), 1e-12,
               "R60: cold path = floor + bytes/BW_stream");
    CHECK(reorderCostMs(4.0e6, 4.0e6, false) > cold, "R60: reorder monotone in bytes");
    CHECK(kReorderStreamBwGbps > l3DramBwGbps(), "R60: reorder stream BW above DRAM platform");
    CHECK(kReorderDispatchGapMs > kReorderLaunchMs, "R60: measured inter-dispatch gap > single launch floor");
    // 策略解析：默认严格 LRU；INFVINO_L3_POLICY=nru 选 NRU 等价模型。
    unsetenv("INFVINO_L3_POLICY");
    CHECK(l3DefaultPolicy() == L3Policy::LRU, "R60: default L3 policy is strict LRU");
    setenv("INFVINO_L3_POLICY", "nru", 1);
    CHECK(l3DefaultPolicy() == L3Policy::NRU, "R60: INFVINO_L3_POLICY=nru selects NRU");
    unsetenv("INFVINO_L3_POLICY");
    CHECK(l3DefaultPolicy() == L3Policy::LRU, "R60: back to LRU after unset");
  }

  // --- R55: occupancyThreads 与 occupancyPressure 同源 ---
  {
    TuningEntry e;
    e.kernel = "conv3x3_ov";
    e.options = "-DOBW=8 -DOBH=2 -DSLM_DIV=1";
    const OpSignature cs = OpSignature::conv3x3(16, 16, 1, 1, 8, 16, 1);
    CHECK(occupancyThreads(e, cs) > 0.0, "occupancyThreads positive for conv3x3");
    CHECK(occupancyPressure(e, cs) > 0.0, "occupancyPressure positive (same op)");
    CHECK_NEAR(occupancyThreads(e, OpSignature::custom("copy_c", {25600, 16})), 0.0, 1e-12,
               "occupancyThreads 0 for small op");
  }
}

ITEST_MAIN("test_rulers")
