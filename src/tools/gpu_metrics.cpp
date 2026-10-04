// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// gpu_metrics —— 通过 Intel Metrics Discovery API (libigdmd) 读 GPU 硬件计数器。
//
// 目的（R42/R43）：给「哪堵墙限制了 kernel」提供**独立的硬件证据**，与
// kernel_bench 的 -DPROBE 剖面交叉验证。本机 sandbox 里 intel_gpu_top 因缺
// PMU 权限不可用（Failed to initialize PMU），但 MD API 可用（实测 rc=0）。
//
// 可用计数器（OA 组，本机）：GpuTime / EuActive / EuStall / EuThreadOccupancy /
// SlmBytesRead/Written / ShaderMemoryAccesses / L3ShaderThroughput / L3 bank 等。
//
// 用法:
//   gpu_metrics --list                       # 枚举 concurrent groups / metric sets / metrics
//   gpu_metrics --sample [SetName] [ms]      # 采样一个 metric set（默认 ComputeBasic，1000ms）
//
// 依赖：intel-metrics-discovery(+dev)（见 docker/Dockerfile）。
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <metrics_discovery_api.h>

using namespace MetricsDiscovery;

// The library exports these as C symbols; the header only provides typedefs
// (MD API is normally consumed via dlopen). Declare them explicitly.
extern "C" TCompletionCode OpenMetricsDevice(IMetricsDeviceLatest ** metricsDevice);
extern "C" TCompletionCode CloseMetricsDevice(IMetricsDeviceLatest * metricsDevice);

static int listMetrics(IMetricsDeviceLatest * dev)
{
  const auto * dp = dev->GetParams();
  for (uint32_t g = 0; g < dp->ConcurrentGroupsCount; ++g) {
    auto * cg = dev->GetConcurrentGroup(g);
    if (!cg) continue;
    const auto * gp = cg->GetParams();
    std::printf("[group %u] %-12s metricSets=%u\n", g, gp->SymbolName, gp->MetricSetsCount);
    for (uint32_t s = 0; s < gp->MetricSetsCount; ++s) {
      auto * ms = cg->GetMetricSet(s);
      if (!ms) continue;
      const auto * sp = ms->GetParams();
      std::printf("    set %u: %-24s metrics=%u\n", s, sp->SymbolName, sp->MetricsCount);
      for (uint32_t m = 0; m < sp->MetricsCount; ++m) {
        auto * mt = ms->GetMetric(m);
        if (!mt) continue;
        const auto * mp = mt->GetParams();
        std::printf("      %-34s %s\n", mp->SymbolName, mp->ShortName ? mp->ShortName : "");
      }
    }
  }
  return 0;
}

static int runSample(IMetricsDeviceLatest * dev, const std::string & setFilter, int ms)
{
  const auto * dp = dev->GetParams();
  for (uint32_t g = 0; g < dp->ConcurrentGroupsCount; ++g) {
    auto * cg = dev->GetConcurrentGroup(g);
    if (!cg) continue;
    const auto * gp = cg->GetParams();
    for (uint32_t s = 0; s < gp->MetricSetsCount; ++s) {
      auto * ms_ = cg->GetMetricSet(s);
      if (!ms_) continue;
      const auto * sp = ms_->GetParams();
      if (!setFilter.empty() && setFilter != sp->SymbolName) continue;

      std::printf("sampling group=%s set=%s (metrics=%u, rawReport=%u B) for %d ms\n",
                  gp->SymbolName, sp->SymbolName, sp->MetricsCount, sp->RawReportSize, ms);
      if (ms_->Activate() != CC_OK) { std::fprintf(stderr, "Activate failed\n"); return 1; }
      uint32_t periodNs = 1000000;   // 1 ms sampling period
      uint32_t oaBufSize = 0;
      TCompletionCode rc = cg->OpenIoStream(ms_, 0, &periodNs, &oaBufSize);
      if (rc != CC_OK) {
        std::fprintf(stderr,
          "OpenIoStream failed rc=%d (%s)\n", (int)rc,
          rc == CC_CONCURRENT_GROUP_LOCKED
            ? "OA concurrent group locked — i915 needs CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS=y "
              "and/or another consumer holds the OA unit"
            : "check /dev/dri access, perf_event_paranoid, and capabilities");
        ms_->Deactivate();
        return 1;
      }
      const uint32_t reportSize = sp->RawReportSize;
      std::vector<char> buf(oaBufSize > reportSize ? oaBufSize : reportSize * 64);
      const int frames = ms > 0 ? (ms * 1000000 / (periodNs ? periodNs : 1000000)) : 1;
      int got = 0;
      for (int f = 0; f < frames; ++f) {
        uint32_t n = 1;
        rc = cg->ReadIoStream(&n, buf.data(), 0);
        if (rc != CC_OK || n == 0) break;
        ++got;
        if (got == 1) {
          // Decode the raw OA report using the metric set's own equations.
          std::vector<TTypedValue_1_0> vals(sp->MetricsCount);
          uint32_t outReports = 0;
          TCompletionCode crc = ms_->CalculateMetrics(
              reinterpret_cast<const uint8_t *>(buf.data()), reportSize, vals.data(),
              static_cast<uint32_t>(sizeof(TTypedValue_1_0) * sp->MetricsCount),
              &outReports, false);
          if (crc == CC_OK) {
            for (uint32_t m = 0; m < sp->MetricsCount; ++m) {
              auto * mt = ms_->GetMetric(m);
              if (!mt) continue;
              const TTypedValue_1_0 & tv = vals[m];
              switch (tv.ValueType) {
                case VALUE_TYPE_UINT32:
                  std::printf("  %-34s %12u  %s\n", mt->GetParams()->SymbolName,
                              tv.ValueUInt32, mt->GetParams()->ShortName); break;
                case VALUE_TYPE_UINT64:
                  std::printf("  %-34s %12llu %s\n", mt->GetParams()->SymbolName,
                              (unsigned long long)tv.ValueUInt64, mt->GetParams()->ShortName); break;
                case VALUE_TYPE_FLOAT:
                  std::printf("  %-34s %12.4f %s\n", mt->GetParams()->SymbolName,
                              tv.ValueFloat, mt->GetParams()->ShortName); break;
                case VALUE_TYPE_BOOL:
                  std::printf("  %-34s %12d  %s\n", mt->GetParams()->SymbolName,
                              (int)tv.ValueBool, mt->GetParams()->ShortName); break;
                default: break;
              }
            }
          } else {
            std::fprintf(stderr, "CalculateMetrics failed rc=%d; raw report:\n", (int)crc);
            for (uint32_t b = 0; b < reportSize && b < 64; ++b)
              std::fprintf(stderr, "%02x", (unsigned char)buf[b]);
            std::fprintf(stderr, "\n");
          }
        }
      }
      std::printf("  read %d report(s)\n", got);
      cg->CloseIoStream();
      ms_->Deactivate();
      return 0;
    }
  }
  std::fprintf(stderr, "gpu_metrics: set '%s' not found\n", setFilter.c_str());
  return 1;
}

int main(int argc, char ** argv)
{
  bool list = false, sample = false;
  std::string sampleSet;
  int sampleMs = 1000;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--list")) list = true;
    else if (!std::strcmp(argv[i], "--sample")) {
      sample = true;
      if (i + 1 < argc && argv[i + 1][0] != '-') sampleSet = argv[++i];
      if (i + 1 < argc && argv[i + 1][0] != '-') sampleMs = std::atoi(argv[++i]);
    }
  }
  if (!list && !sample) { list = true; }

  IMetricsDeviceLatest * dev = nullptr;
  TCompletionCode rc = OpenMetricsDevice(&dev);
  if (rc != CC_OK || !dev) {
    std::fprintf(stderr, "gpu_metrics: OpenMetricsDevice failed rc=%d "
                         "(pass --device=/dev/dri/renderD128)\n", (int)rc);
    return 1;
  }
  const auto * dp = dev->GetParams();
  std::printf("device=%s  api=%u.%u  concurrentGroups=%u\n",
              dp->DeviceName, dp->Version.MajorNumber, dp->Version.MinorNumber,
              dp->ConcurrentGroupsCount);

  int r = 0;
  if (list) r = listMetrics(dev);
  else r = runSample(dev, sampleSet.empty() ? std::string("ComputeBasic") : sampleSet, sampleMs);
  CloseMetricsDevice(dev);
  return r;
}
