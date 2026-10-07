// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_bench —— 自研 kernel 基准：延迟(ms) + ops/EU/cycle（FP16 上限 32 = 16 packed FMA）。
//
//   kernel_bench --list-devices
//   kernel_bench --op gemm --shape 1024,1024,1024 --verify
//   kernel_bench --op gemm --shape 6400,64,64 --shape 400,256,256 --tiles 64,64,16,8,4
//
// 设计：每个 case 只计 kernel 自身时间（OpenCL event profiling），与
// infvino_bench 的 “net only” 口径一致，便于逐轮对比。
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"
#include "infvino/Half.hpp"
#include "infvino/Tiles.hpp"

namespace
{

struct Shape
{
  int M, N, K;
  std::string label;
};

std::vector<int> parseInts(const std::string & s)
{
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::atoi(tok.c_str()));
  return out;
}

void fillRandom(std::vector<uint16_t> & v, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dis(-0.5f, 0.5f);
  for (auto & x : v) x = infvino::f32_to_f16(dis(rng));
}

// 相对误差（不使用余弦）：mean_rel = mean|diff|/mean|ref|, max_rel = max|diff|/max|ref|.
struct Accuracy
{
  double mean_rel, max_rel, max_abs;
};

Accuracy verifyGemm(
  const std::vector<uint16_t> & A, const std::vector<uint16_t> & B,
  const std::vector<uint16_t> & C, int M, int N, int K)
{
  double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
  size_t n = 0;
  for (int r = 0; r < M; ++r) {
    for (int c = 0; c < N; ++c) {
      float acc = 0.f;
      for (int t = 0; t < K; ++t)
        acc += infvino::f16_to_f32(A[(size_t)r * K + t]) * infvino::f16_to_f32(B[(size_t)t * N + c]);
      const double got = infvino::f16_to_f32(C[(size_t)r * N + c]);
      const double d = std::fabs(got - acc);
      sumabs += d;
      sumref += std::fabs(static_cast<double>(acc));
      maxabs = std::max(maxabs, d);
      refmax = std::max(refmax, static_cast<double>(std::fabs(acc)));
      ++n;
    }
  }
  (void)n;
  return {sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs};
}

int benchGemm(infvino::ClRuntime & rt, const infvino::Tiles & t, const Shape & s, int iters, bool verify)
{
  const int M = s.M, N = s.N, K = s.K;
  if (M % t.BM || N % t.BN)
    std::fprintf(stderr, "[skip] %s: shape not multiple of block tile\n", s.label.c_str());

  cl_kernel k;
  try {
    k = rt.buildKernel("gemm", "gemm_f16", t.options());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }

  std::vector<uint16_t> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  fillRandom(hA, 1234);
  fillRandom(hB, 5678);

  cl_mem dA = rt.alloc((size_t)M * K * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)K * N * 2, CL_MEM_READ_ONLY);
  cl_mem dC = rt.alloc((size_t)M * N * 2, CL_MEM_WRITE_ONLY);
  rt.write(dA, (size_t)M * K * 2, hA.data());
  rt.write(dB, (size_t)K * N * 2, hB.data());

  clSetKernelArg(k, 0, sizeof(dA), &dA);
  clSetKernelArg(k, 1, sizeof(dB), &dB);
  clSetKernelArg(k, 2, sizeof(dC), &dC);
  clSetKernelArg(k, 3, sizeof(M), &M);
  clSetKernelArg(k, 4, sizeof(N), &N);
  clSetKernelArg(k, 5, sizeof(K), &K);

  const size_t lws[2] = {static_cast<size_t>(t.BN / t.TN), static_cast<size_t>(t.BM / t.TM)};
  const size_t gws[2] = {
    static_cast<size_t>((N + t.BN - 1) / t.BN) * lws[0],
    static_cast<size_t>((M + t.BM - 1) / t.BM) * lws[1]};

  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws); }, 3, iters);
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double ops = rt.opsPerEuCycle(flops, med);

  std::printf(
    "  gemm %-16s M=%-5d N=%-4d K=%-5d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), M, N, K, med, flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);

  if (verify) {
    rt.read(dC, (size_t)M * N * 2, hC.data());
    const Accuracy a = verifyGemm(hA, hB, hC, M, N, K);
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e", a.mean_rel, a.max_rel, a.max_abs);
  }
  std::printf("\n");

  clReleaseMemObject(dA);
  clReleaseMemObject(dB);
  clReleaseMemObject(dC);
  clReleaseKernel(k);
  return 0;
}

// Specialized small-GEMM: lane-over-K split-K with sub-group reduction (R32).
int benchGemmSk(infvino::ClRuntime & rt, const infvino::Conv1x1Cfg & c, const Shape & s, int iters, bool verify)
{
  const int M = s.M, N = s.N, K = s.K;
  const int TM = c.TM, TN = c.TN, SG = c.SG, UK = c.UNROLL;
  char opts[256];
  std::snprintf(opts, sizeof(opts),
                "-DSK_TM=%d -DSK_TN=%d -DSK_SG=%d -DSK_UK=%d -DACT=%d -DRES=%d "
                "-cl-mad-enable -cl-fast-relaxed-math", TM, TN, SG, UK, c.ACT, c.RES);
  cl_kernel k;
  try {
    k = rt.buildKernel("gemm_sk", "gemm_sk_f16", opts);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }
  std::vector<uint16_t> hA((size_t)M * K), hB((size_t)K * N), hC((size_t)M * N);
  fillRandom(hA, 1234);
  fillRandom(hB, 5678);
  cl_mem dA = rt.alloc((size_t)M * K * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)K * N * 2, CL_MEM_READ_ONLY);
  cl_mem dC = rt.alloc((size_t)M * N * 2, CL_MEM_WRITE_ONLY);
  rt.write(dA, (size_t)M * K * 2, hA.data());
  rt.write(dB, (size_t)K * N * 2, hB.data());
  cl_mem dNull = nullptr;
  clSetKernelArg(k, 0, sizeof(dA), &dA);
  clSetKernelArg(k, 1, sizeof(dB), &dB);
  clSetKernelArg(k, 2, sizeof(dC), &dC);
  clSetKernelArg(k, 3, sizeof(M), &M);
  clSetKernelArg(k, 4, sizeof(N), &N);
  clSetKernelArg(k, 5, sizeof(K), &K);
  clSetKernelArg(k, 6, sizeof(dNull), &dNull);
  clSetKernelArg(k, 7, sizeof(dNull), &dNull);
  const size_t lws[2] = {static_cast<size_t>(SG), 1};
  const size_t gws[2] = {
    static_cast<size_t>((N + TN - 1) / TN) * lws[0],
    static_cast<size_t>((M + TM - 1) / TM)};
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws); }, 3, iters);
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  gemm_sk %-14s M=%-5d N=%-5d K=%-5d TM%d TN%d u%d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), M, N, K, TM, TN, UK, med, flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);
  if (verify) {
    rt.read(dC, (size_t)M * N * 2, hC.data());
    const Accuracy a = verifyGemm(hA, hB, hC, M, N, K);
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e", a.mean_rel, a.max_rel, a.max_abs);
  }
  std::printf("\n");
  clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC); clReleaseKernel(k);
  return 0;
}

// R32: 纯 launch 开销探针 —— 连续入队 N 个极小 kernel（同一 buffer，in-order 队列
// 隐含依赖），分别量 host 入队、GPU busy 之和、以及 wall。用于回答「每 dispatch 的
// 非重叠开销到底在 host 还是 GPU 前端」。
int benchChain(infvino::ClRuntime & rt, int iters)
{
  if (iters < 1) iters = 1;
  const char * src = "__kernel void nop_k(__global int * a){ int x=get_global_id(0);"
                     " for(int i=0;i<1024;i++) x=x*1103515245+12345; a[get_global_id(0)]=(int)x; }";
  cl_kernel k;
  try {
    k = rt.buildFromSource(src, "nop_k");
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }
  const size_t N = 8192;
  cl_mem d = rt.alloc(N * 4, CL_MEM_READ_WRITE);
  std::vector<int> h(N, 0);
  rt.write(d, N * 4, h.data());
  clSetKernelArg(k, 0, sizeof(d), &d);
  const size_t gws[1] = {N};
  for (int i = 0; i < 5; ++i) {
    cl_event e = infvino::ClRuntime::enqueueND(rt.queue(), k, 1, gws, nullptr);
    clWaitForEvents(1, &e);
    clReleaseEvent(e);
  }
  std::vector<cl_event> evs;
  evs.reserve(static_cast<size_t>(iters));
  auto   t0      = std::chrono::steady_clock::now();
  double host_us = 0.0;
  for (int i = 0; i < iters; ++i) {
    const auto h0 = std::chrono::steady_clock::now();
    evs.push_back(infvino::ClRuntime::enqueueND(rt.queue(), k, 1, gws, nullptr));
    host_us += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - h0).count();
  }
  const auto t1 = std::chrono::steady_clock::now();
  clWaitForEvents(static_cast<cl_uint>(evs.size()), evs.data());
  const auto t2 = std::chrono::steady_clock::now();
  double busy_us = 0.0;
  for (cl_event e : evs) {
    cl_ulong s = 0, en = 0;
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr);
    clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr);
    busy_us += static_cast<double>(en - s) * 1e-3;
    clReleaseEvent(e);
  }
  const double wall_ms  = std::chrono::duration<double, std::milli>(t2 - t0).count();
  const double burst_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  const double host_ms  = host_us / 1e3;
  const double busy_ms  = busy_us / 1e3;
  std::printf("  chain N=%d (8192WI x1024loop): wall=%.3f ms  enqueue_phase=%.3f  host=%.3f  sum_busy=%.3f\n",
              iters, wall_ms, burst_ms, host_ms, busy_ms);
  std::printf("    per-dispatch: wall=%.2f us  host=%.2f us  gpu_busy=%.2f us  wall-busy=%.2f us\n",
              wall_ms * 1000 / iters, host_ms * 1000 / iters, busy_ms * 1000 / iters,
              (wall_ms - busy_ms) * 1000 / iters);
  clReleaseMemObject(d);
  clReleaseKernel(k);
  return 0;
}

struct ConvShape
{
  int Cin, Cout, H, W;
  std::string label;
};

int benchConv(infvino::ClRuntime & rt, const infvino::Conv3x3Cfg & c, const ConvShape & s, int iters, bool verify)
{
  const int Hout = (s.H + 2 * c.PAD - 3) / c.STRIDE + 1;
  const int Wout = (s.W + 2 * c.PAD - 3) / c.STRIDE + 1;
  int ovslm = 1;
  if (c.OV) {
    const char * e = std::getenv("OV_SLM");
    if (e) ovslm = std::atoi(e);
    if (ovslm < 1) ovslm = 1;
  }
  cl_kernel k;
  try {
    if (c.OV)
      k = rt.buildKernel("conv_ov", "conv3x3_ov",
                         c.options() + " -DSLM_DIV=" + std::to_string(ovslm));
    else if (c.SGK)
      k = rt.buildKernel("conv_sg", "conv3x3_sg", c.options());
    else if (c.OSV)
      k = rt.buildKernel("conv_osv", "conv3x3_osv", c.options());
    else if (c.RT)
      k = rt.buildKernel("conv", "conv3x3_rt", c.options());
    else if (c.DB)
      k = rt.buildKernel("conv", "conv3x3_db", c.options());
    else
      k = rt.buildKernel("conv", "conv3x3_f16", c.options());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }

  std::vector<uint16_t> hX((size_t)s.Cin * s.H * s.W), hB((size_t)s.Cout);
  std::vector<uint16_t> hWt((size_t)s.Cout * s.Cin * 9), hY((size_t)s.Cout * Hout * Wout);
  std::mt19937 rng123(11), rng456(22);
  std::uniform_real_distribution<float> dum(-0.5f, 0.5f), dud(-0.2f, 0.2f);
  for (auto & v : hX) v = infvino::f32_to_f16(dum(rng123));
  for (auto & v : hWt) v = infvino::f32_to_f16(dud(rng456));
  for (auto & v : hB) v = infvino::f32_to_f16(dud(rng456));

  const size_t wbytes = c.OV
    ? (size_t)((s.Cout + 31) / 32) * s.Cin * 9 * 32 * 2
    : (size_t)s.Cout * s.Cin * 9 * 2;
  cl_mem dX = rt.alloc((size_t)s.Cin * s.H * s.W * 2, CL_MEM_READ_ONLY);
  cl_mem dW = rt.alloc(wbytes, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)s.Cout * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc((size_t)s.Cout * Hout * Wout * 2, CL_MEM_WRITE_ONLY);
  rt.write(dX, (size_t)s.Cin * s.H * s.W * 2, hX.data());
  if (c.OV) {
    // OpenVINO os_iyx_osv32 weight layout: [ceil(Cout/32)][Cin][3][3][32] with
    // block position p = (o % 32) (see GET_FILTER_OS_IYX_OSV_INDEX with
    // sub_group_size=32). The strided block read gives lane l positions (l, l+16),
    // which the kernel maps to channels (fmg*32+l, fmg*32+16+l).
    const int fmgroups = (s.Cout + 31) / 32;
    std::vector<uint16_t> hWo((size_t)fmgroups * s.Cin * 9 * 32, 0);
    for (int fmg = 0; fmg < fmgroups; ++fmg)
      for (int ci = 0; ci < s.Cin; ++ci)
        for (int kk = 0; kk < 9; ++kk) {
          uint16_t *dst = &hWo[(((size_t)fmg * s.Cin + ci) * 9 + kk) * 32];
          for (int p = 0; p < 32; ++p) {
            const int oc = fmg * 32 + p;
            dst[p] = (oc < s.Cout) ? hWt[((size_t)oc * s.Cin + ci) * 9 + kk] : (uint16_t)0;
          }
        }
    rt.write(dW, hWo.size() * 2, hWo.data());
  } else if (c.WGL || c.SGK) {
    // Repack [Cout][Cin][KHKW] -> [Cin][KHKW][Cout] so the CB weights for one
    // (ci,kk) are contiguous. Round 19: also used by the lane=channel OV-style
    // kernel (conv3x3_sg), whose per-lane weight reads were otherwise strided by
    // Cin*KHKW -> 16 different cache lines per load.
    std::vector<uint16_t> hWg((size_t)s.Cin * 9 * s.Cout);
    for (int oc = 0; oc < s.Cout; ++oc)
      for (int ci = 0; ci < s.Cin; ++ci)
        for (int kk = 0; kk < 9; ++kk)
          hWg[((size_t)ci * 9 + kk) * s.Cout + oc] = hWt[((size_t)oc * s.Cin + ci) * 9 + kk];
    rt.write(dW, hWg.size() * 2, hWg.data());
  } else {
    rt.write(dW, (size_t)s.Cout * s.Cin * 9 * 2, hWt.data());
  }
  rt.write(dB, (size_t)s.Cout * 2, hB.data());
  clSetKernelArg(k, 0, sizeof(dX), &dX);
  clSetKernelArg(k, 1, sizeof(dW), &dW);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  int Cin = s.Cin, H = s.H, W = s.W, Cout = s.Cout, ho = Hout, wo = Wout;
  if (c.OV) {
    cl_mem dRes = nullptr;
    clSetKernelArg(k, 3, sizeof(dRes), &dRes);
    clSetKernelArg(k, 4, sizeof(dY), &dY);
    clSetKernelArg(k, 5, sizeof(Cin), &Cin);
    clSetKernelArg(k, 6, sizeof(H), &H);
    clSetKernelArg(k, 7, sizeof(W), &W);
    clSetKernelArg(k, 8, sizeof(Cout), &Cout);
    clSetKernelArg(k, 9, sizeof(ho), &ho);
    clSetKernelArg(k, 10, sizeof(wo), &wo);
  } else {
    clSetKernelArg(k, 3, sizeof(dY), &dY);
    clSetKernelArg(k, 4, sizeof(Cin), &Cin);
    clSetKernelArg(k, 5, sizeof(H), &H);
    clSetKernelArg(k, 6, sizeof(W), &W);
    clSetKernelArg(k, 7, sizeof(Cout), &Cout);
    clSetKernelArg(k, 8, sizeof(ho), &ho);
    clSetKernelArg(k, 9, sizeof(wo), &wo);
  }

  size_t lxThreads, lyThreads;
  if (c.OV) {
    lxThreads = 1;
    lyThreads = 1;
  } else if (c.SGK) {
    lxThreads = static_cast<size_t>(c.SG);
    lyThreads = 1;
  } else if (c.OSV) {
    lxThreads = static_cast<size_t>(c.TX / c.TM) * c.SG;
    lyThreads = static_cast<size_t>(c.TY) * (c.CB / (c.SG * c.VECO));
  } else {
    lxThreads = static_cast<size_t>(c.TX / c.TM);
    lyThreads = c.RT ? static_cast<size_t>(c.TY) * (c.CB / c.TN)
                     : static_cast<size_t>(c.TY);
  }
  const int ovSg = (c.SG > 0) ? c.SG : 16;
  const size_t lws[3] = {lxThreads, c.OV ? 1 : lyThreads,
                         c.OV ? static_cast<size_t>(ovSg * ovslm) : 1};
  const size_t gws[3] = {
    static_cast<size_t>((Wout + c.TX - 1) / c.TX) * lxThreads,
    static_cast<size_t>((Hout + c.TY - 1) / c.TY) * (c.OV ? 1 : lyThreads),
    c.OV ? static_cast<size_t>((((s.Cout + 1) / 2 + 15) / 16) * ovSg * ovslm)
         : static_cast<size_t>((Cout + c.CB - 1) / c.CB)};

  double tmin = 0.0, tp90 = 0.0;
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws); }, 3, iters,
    &tmin, &tp90);
  const double flops = 2.0 * Cout * Hout * Wout * Cin * 9;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  conv3x3 %-18s Cin=%-4d Cout=%-4d %dx%d s%d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)  [min %.3f p90 %.3f, spread %+.0f%%]",
    s.label.c_str(), Cin, Cout, H, W, c.STRIDE, med, flops / (med * 1e-3) / 1e9,
    ops, ops / 32 * 100, tmin, tp90,
    tp90 > 0 ? (tp90 / tmin - 1.0) * 100.0 : 0.0);

  if (verify) {
    rt.read(dY, (size_t)Cout * Hout * Wout * 2, hY.data());
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    // R41: apply the same activation as the kernel (canonical conv3x3 codes:
    // 1=SiLU, 3=HardSwish). The reference previously stored the raw conv, so any
    // ACT!=0 verify was wrong by construction.
    auto ref_act = [&](float f) -> double {
      if (c.ACT == 1) return f / (1.0 + std::exp(-(double)f));
      if (c.ACT == 3) return (double)(f * std::min(std::max(f + 3.0f, 0.0f), 6.0f) / 6.0f);
      return (double)f;
    };
    std::vector<float> ref((size_t)Cout * Hout * Wout);
    for (int oc = 0; oc < Cout; ++oc)
      for (int oy = 0; oy < Hout; ++oy)
        for (int ox = 0; ox < Wout; ++ox) {
          float acc = infvino::f16_to_f32(hB[oc]);
          for (int ci = 0; ci < Cin; ++ci)
            for (int kh = 0; kh < 3; ++kh)
              for (int kw = 0; kw < 3; ++kw) {
                int yy = oy * c.STRIDE - c.PAD + kh, xx = ox * c.STRIDE - c.PAD + kw;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                  acc += infvino::f16_to_f32(hX[((size_t)ci * H + yy) * W + xx]) *
                         infvino::f16_to_f32(hWt[((size_t)oc * Cin + ci) * 9 + kh * 3 + kw]);
              }
          ref[((size_t)oc * Hout + oy) * Wout + ox] = static_cast<float>(ref_act(acc));
        }
    for (size_t i = 0; i < ref.size(); ++i) {
      const double got = infvino::f16_to_f32(hY[i]);
      const double d = std::fabs(got - ref[i]);
      sumabs += d;
      sumref += std::fabs(static_cast<double>(ref[i]));
      maxabs = std::max(maxabs, d);
      refmax = std::max(refmax, static_cast<double>(std::fabs(ref[i])));
    }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dX); clReleaseMemObject(dW); clReleaseMemObject(dB); clReleaseMemObject(dY);
  clReleaseKernel(k);
  return 0;
}

// Round 25: blocked conv — OV `convolution_gpu_bfyx_f16` port (kernels/conv_blk.cl).
// Input is in b_fs_yx_fsv16 ([Cin/16][H][W][16]), weights in os_is_yx_isv16_osv16
// ([Cout/16][Cin/16][3][3][16 isv][16 osv]), output plain bfyx.  OBW = c.TX.
int benchConvBlk(infvino::ClRuntime & rt, const infvino::Conv3x3Cfg & c, const ConvShape & s, int iters, bool verify)
{
  const int Hout = (s.H + 2 * c.PAD - 3) / c.STRIDE + 1;
  const int Wout = (s.W + 2 * c.PAD - 3) / c.STRIDE + 1;
  const int Cin = s.Cin, Cout = s.Cout, OBW = c.TX;
  char oo[192];
  const int SLM_DIV = (c.TY >= 1) ? c.TY : 1;
  std::snprintf(oo, sizeof(oo),
                "-DOBW=%d -DSTRIDE=%d -DPAD=%d -DACT=%d -DSG=16 -DSLM_DIV=%d -DOUT_FSV16=%d "
                "-cl-mad-enable -cl-fast-relaxed-math",
                OBW, c.STRIDE, c.PAD, c.ACT, SLM_DIV, c.OUTFSV ? 1 : 0);
  // Mirror the registry's compile-time specialization so the bench matches the
  // production candidate binary exactly.
  {
    std::string add;
    if (Wout % OBW == 0) add += " -DFIT_WH=1";
    if (Cout % 16 == 0) add += " -DFIT_COUT=1";
    if (Cin % 16 == 0) add += " -DFIT_CIN=1";
    std::strncat(oo, add.c_str(), sizeof(oo) - std::strlen(oo) - 1);
  }
  cl_kernel k;
  try { k = rt.buildKernel("conv_blk", "conv3x3_blk", oo); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }

  const int icb = (Cin + 15) / 16, ocb = (Cout + 15) / 16;
  std::vector<uint16_t> hX((size_t)Cin * s.H * s.W), hB((size_t)Cout),
      hWt((size_t)Cout * Cin * 9), hY((size_t)Cout * Hout * Wout);
  std::mt19937 rngX(11), rngW(22);
  std::uniform_real_distribution<float> dum(-0.5f, 0.5f), dud(-0.2f, 0.2f);
  for (auto & v : hX) v = infvino::f32_to_f16(dum(rngX));
  for (auto & v : hWt) v = infvino::f32_to_f16(dud(rngW));
  for (auto & v : hB) v = infvino::f32_to_f16(dud(rngW));

  // input bfyx -> b_fs_yx_fsv16
  std::vector<uint16_t> hXb((size_t)icb * s.H * s.W * 16, 0);
  for (int ch = 0; ch < Cin; ++ch)
    for (int y = 0; y < s.H; ++y)
      for (int x = 0; x < s.W; ++x)
        hXb[(((size_t)(ch / 16) * s.H + y) * s.W + x) * 16 + (ch % 16)] =
          hX[((size_t)ch * s.H + y) * s.W + x];
  // weights bfyx -> os_is_yx_isv16_osv16
  std::vector<uint16_t> hWb((size_t)ocb * icb * 9 * 16 * 16, 0);
  for (int o = 0; o < Cout; ++o)
    for (int i = 0; i < Cin; ++i)
      for (int kh = 0; kh < 3; ++kh)
        for (int kw = 0; kw < 3; ++kw)
          hWb[((((size_t)(o / 16) * icb + (i / 16)) * 9 + kh * 3 + kw) * 16 + (i % 16)) * 16 + (o % 16)] =
              hWt[((size_t)o * Cin + i) * 9 + kh * 3 + kw];

  const size_t yelems = c.OUTFSV ? (size_t)ocb * Hout * Wout * 16 : (size_t)Cout * Hout * Wout;
  cl_mem dX = rt.alloc((size_t)icb * s.H * s.W * 16 * 2, CL_MEM_READ_ONLY);
  cl_mem dW = rt.alloc(hWb.size() * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)Cout * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc(yelems * 2, CL_MEM_WRITE_ONLY);
  rt.write(dX, hXb.size() * 2, hXb.data());
  rt.write(dW, hWb.size() * 2, hWb.data());
  rt.write(dB, (size_t)Cout * 2, hB.data());
  clSetKernelArg(k, 0, sizeof(dX), &dX);
  clSetKernelArg(k, 1, sizeof(dW), &dW);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dY), &dY);
  int CinA = Cin, HA = s.H, WA = s.W, CoutA = Cout, ho = Hout, wo = Wout;
  clSetKernelArg(k, 4, sizeof(CinA), &CinA);
  clSetKernelArg(k, 5, sizeof(HA), &HA);
  clSetKernelArg(k, 6, sizeof(WA), &WA);
  clSetKernelArg(k, 7, sizeof(CoutA), &CoutA);
  clSetKernelArg(k, 8, sizeof(ho), &ho);
  clSetKernelArg(k, 9, sizeof(wo), &wo);

  const size_t lws[3] = {1, static_cast<size_t>(16 * SLM_DIV), 1};
  const size_t gws[3] = {
    static_cast<size_t>(((Wout + OBW - 1) / OBW) * Hout),
    static_cast<size_t>(((Cout + 15) / 16) * 16 * SLM_DIV), 1};
  double btmin = 0.0, btp90 = 0.0;
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws); }, 3, iters,
    &btmin, &btp90);
  const double flops = 2.0 * Cout * Hout * Wout * Cin * 9;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  conv3x3blk %-15s Cin=%-4d Cout=%-4d %dx%d s%d OBW%d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)  [min %.3f p90 %.3f, spread %+.0f%%]",
    s.label.c_str(), Cin, Cout, s.H, s.W, c.STRIDE, OBW, med,
    flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100, btmin, btp90,
    btp90 > 0 ? (btp90 / btmin - 1.0) * 100.0 : 0.0);

  if (verify) {
    if (c.OUTFSV) {
      std::vector<uint16_t> hYb(yelems, 0);
      rt.read(dY, yelems * 2, hYb.data());
      for (int ch = 0; ch < Cout; ++ch)
        for (int y = 0; y < Hout; ++y)
          for (int x = 0; x < Wout; ++x)
            hY[((size_t)ch * Hout + y) * Wout + x] =
              hYb[(((size_t)(ch / 16) * Hout + y) * Wout + x) * 16 + (ch % 16)];
    } else {
      rt.read(dY, (size_t)Cout * Hout * Wout * 2, hY.data());
    }
    auto ref_act = [&](float f) -> double {
      if (c.ACT == 1) return f / (1.0 + std::exp(-(double)f));
      if (c.ACT == 3) return (double)(f * std::min(std::max(f + 3.0f, 0.0f), 6.0f) / 6.0f);
      return (double)f;
    };
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    for (int oc = 0; oc < Cout; ++oc)
      for (int oy = 0; oy < Hout; ++oy)
        for (int ox = 0; ox < Wout; ++ox) {
          float acc = infvino::f16_to_f32(hB[oc]);
          for (int ci = 0; ci < Cin; ++ci)
            for (int kh = 0; kh < 3; ++kh)
              for (int kw = 0; kw < 3; ++kw) {
                int yy = oy * c.STRIDE - c.PAD + kh, xx = ox * c.STRIDE - c.PAD + kw;
                if (yy >= 0 && yy < s.H && xx >= 0 && xx < s.W)
                  acc += infvino::f16_to_f32(hX[((size_t)ci * s.H + yy) * s.W + xx]) *
                         infvino::f16_to_f32(hWt[((size_t)oc * Cin + ci) * 9 + kh * 3 + kw]);
              }
          const double r = ref_act(acc);
          const size_t oidx = ((size_t)oc * Hout + oy) * Wout + ox;
          const double got = infvino::f16_to_f32(hY[oidx]);
          const double d = std::fabs(got - r);
          sumabs += d; sumref += std::fabs(r);
          maxabs = std::max(maxabs, d); refmax = std::max(refmax, std::fabs(r));
        }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dX); clReleaseMemObject(dW); clReleaseMemObject(dB); clReleaseMemObject(dY);
  clReleaseKernel(k);
  return 0;
}

// R52: standalone bfyx -> b_fs_yx_fsv16 reorder (the per-layer blocked-chain tax).
// Shape = Cin,_,H,W; measures GB/s (bytes = read + write, incl. channel padding).
int benchReorder(infvino::ClRuntime & rt, const ConvShape & s, int iters)
{
  const int C = s.Cin, H = s.H, W = s.W, cb = (C + 15) / 16;
  cl_kernel k;
  try { k = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", ""); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
  const size_t ne = (size_t)C * H * W, no = (size_t)cb * H * W * 16;
  cl_mem dI = rt.alloc(ne * 2, CL_MEM_READ_ONLY);
  cl_mem dO = rt.alloc(no * 2, CL_MEM_WRITE_ONLY);
  std::vector<uint16_t> h(ne, 0x3c00);   // 1.0
  rt.write(dI, ne * 2, h.data());
  int Ca = C, Ha = H, Wa = W;
  clSetKernelArg(k, 0, sizeof(dI), &dI);
  clSetKernelArg(k, 1, sizeof(dO), &dO);
  clSetKernelArg(k, 2, sizeof(Ca), &Ca);
  clSetKernelArg(k, 3, sizeof(Ha), &Ha);
  clSetKernelArg(k, 4, sizeof(Wa), &Wa);
  const size_t gws[3] = {(size_t)W, (size_t)H, (size_t)C};
  double btmin = 0.0, btp90 = 0.0;
  double med = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, nullptr); },
                         3, iters, &btmin, &btp90);
  const double bytes = 2.0 * (ne + no) * 2.0;
  std::printf("  reorder      Cin=%-4d %dx%d bytes=%.2fMB  %8.4f ms  %7.1f GB/s  [min %.4f p90 %.4f spread %+.0f%%]\n",
              C, H, W, bytes / 1e6, med, bytes / (med * 1e-3) / 1e9, btmin, btp90,
              btp90 > 0 ? (btp90 / btmin - 1.0) * 100.0 : 0.0);
  clReleaseKernel(k);

  // R54: OV-style SLM transpose variant.
  try {
    cl_kernel ks = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16_slm", "");
    cl_int e0 = clSetKernelArg(ks, 0, sizeof(dI), &dI);
    cl_int e1 = clSetKernelArg(ks, 1, sizeof(dO), &dO);
    cl_int e2 = clSetKernelArg(ks, 2, sizeof(Ca), &Ca);
    cl_int e3 = clSetKernelArg(ks, 3, sizeof(Ha), &Ha);
    cl_int e4 = clSetKernelArg(ks, 4, sizeof(Wa), &Wa);
    if (e0|e1|e2|e3|e4)
      std::fprintf(stderr, "  [reorder_slm] setArg errs %d %d %d %d %d\n", e0,e1,e2,e3,e4);
    const size_t g2[3] = {(size_t)((W + 15) / 16) * 16, (size_t)H, (size_t)cb};
    const size_t l2[3] = {16, 1, 1};
    double mn = 0.0, p90 = 0.0;
    double m2 = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), ks, 3, g2, l2); },
                          3, iters, &mn, &p90);
    std::printf("  reorder_slm  Cin=%-4d %dx%d bytes=%.2fMB  %8.4f ms  %7.1f GB/s  [min %.4f p90 %.4f spread %+.0f%%]\n",
                C, H, W, bytes / 1e6, m2, bytes / (m2 * 1e-3) / 1e9, mn, p90,
                p90 > 0 ? (p90 / mn - 1.0) * 100.0 : 0.0);
    // correctness: old vs slm must be bit-identical (same fp16 values, only order).
    {
      cl_kernel ko = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
      clSetKernelArg(ko, 0, sizeof(dI), &dI);
      clSetKernelArg(ko, 1, sizeof(dO), &dO);
      clSetKernelArg(ko, 2, sizeof(Ca), &Ca);
      clSetKernelArg(ko, 3, sizeof(Ha), &Ha);
      clSetKernelArg(ko, 4, sizeof(Wa), &Wa);
      std::vector<uint16_t> a(no), b(no, 0xd0d0), zeros(no, 0);
      rt.write(dO, no * 2, zeros.data());
      infvino::ClRuntime::enqueueND(rt.queue(), ko, 3, gws, nullptr);
      rt.read(dO, no * 2, a.data());
      rt.write(dO, no * 2, zeros.data());
      infvino::ClRuntime::enqueueND(rt.queue(), ks, 3, g2, l2);
      rt.read(dO, no * 2, b.data());
      int bad = 0;
      for (size_t i = 0; i < no; ++i) if (a[i] != b[i]) ++bad;
      std::printf("  reorder_slm  Cin=%-4d %dx%d verify: %s (%d/%zu mismatches)\n",
                  C, H, W, bad == 0 ? "PASS" : "FAIL", bad, no);
      clReleaseKernel(ko);
    }
    clReleaseKernel(ks);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "  [reorder_slm] build/run failed: %s\n", e.what());
  }
  clReleaseMemObject(dI); clReleaseMemObject(dO);
  return 0;
}

// R60: reorder additivity probe. N reorders back-to-back.
//   mode bit0: 1 = shared input buffer (L3-hot after 1st), 0 = N distinct inputs.
//   mode bit1: 1 = shared output buffer,                    0 = N distinct outputs.
//   (0,0) all cold; (1,1) all hot; the difference attributes the additivity gap to
//   the input vs output working set (vs L3 capacity).
int benchReorderSeq(infvino::ClRuntime & rt, const ConvShape & s, int nseq, int mode, int iters)
{
  const int C = s.Cin, H = s.H, W = s.W, cb = (C + 15) / 16;
  cl_kernel k;
  try { k = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", ""); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
  const size_t ne = (size_t)C * H * W, oe = (size_t)cb * H * W * 16;
  if (nseq < 1) nseq = 1;
  const bool sharedIn = (mode & 1) != 0, sharedOut = (mode & 2) != 0;
  const int nbuf = sharedIn ? 1 : nseq, nout = sharedOut ? 1 : nseq;
  std::vector<cl_mem> dI(nbuf), dO(nout);
  std::vector<uint16_t> h(ne, 0x3c00);
  for (int i = 0; i < nbuf; ++i) { dI[i] = rt.alloc(ne * 2, CL_MEM_READ_ONLY); rt.write(dI[i], ne * 2, h.data()); }
  for (int i = 0; i < nout; ++i) dO[i] = rt.alloc(oe * 2, CL_MEM_WRITE_ONLY);
  int Ca = C, Ha = H, Wa = W;
  const size_t gws[3] = {(size_t)W, (size_t)H, (size_t)C};
  auto enqueueOne = [&](cl_mem in, cl_mem out) {
    clSetKernelArg(k, 0, sizeof(in), &in);
    clSetKernelArg(k, 1, sizeof(out), &out);
    clSetKernelArg(k, 2, sizeof(Ca), &Ca);
    clSetKernelArg(k, 3, sizeof(Ha), &Ha);
    clSetKernelArg(k, 4, sizeof(Wa), &Wa);
    return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, nullptr);
  };
  const double s1 = rt.timeMs([&] { return enqueueOne(dI[0], dO[0]); }, 3, iters);
  // GPU-spanning time: first event's START -> last event's END (excludes host enqueue bubbles).
  auto timeBatch = [&]() -> double {
    auto once = [&](bool timed, double * out) {
      cl_event first = nullptr, last = nullptr, prev = nullptr;
      for (int i = 0; i < nseq; ++i)
      {
        cl_event e = enqueueOne(sharedIn ? dI[0] : dI[i], sharedOut ? dO[0] : dO[i]);
        if (i == 0) first = e;
        else if (prev && prev != first) clReleaseEvent(prev);
        prev = e;
        last = e;
      }
      clWaitForEvents(1, &last);
      if (timed)
      {
        cl_ulong t0 = 0, t1 = 0;
        clGetEventProfilingInfo(first, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
        clGetEventProfilingInfo(last, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
        *out = static_cast<double>(t1 - t0) * 1e-6;
      }
      if (last != first) clReleaseEvent(last);
      clReleaseEvent(first);
    };
    for (int w = 0; w < 3; ++w) once(false, nullptr);
    std::vector<double> ts;
    for (int it = 0; it < iters; ++it) { double v = 0.0; once(true, &v); ts.push_back(v); }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size() / 2];
  };
  const double stot = timeBatch();
  const double bytes = 2.0 * (ne + oe) * 2.0;
  std::printf("  reorder_seq  Cin=%-4d %dx%d n=%-3d in=%s out=%s  bytes/pass=%.2f MB\n", C, H, W, nseq,
              sharedIn ? "shared" : "distinct", sharedOut ? "shared" : "distinct", bytes / 1e6);
  std::printf("    single=%8.4f ms  total=%8.4f ms  per=%8.4f ms  additivity(N x single)=%.3f\n",
              s1, stot, stot / nseq, s1 > 0 ? (stot / nseq) / s1 : 0.0);
  for (cl_mem m : dI) clReleaseMemObject(m);
  for (cl_mem m : dO) clReleaseMemObject(m);
  clReleaseKernel(k);
  return 0;
}

// Specialized 1x1 conv (pointwise) kernel: fused bias + activation.
int benchConv1x1(infvino::ClRuntime & rt, const infvino::Conv1x1Cfg & c, const ConvShape & s, int iters, bool verify)
{
  const int Cin = s.Cin, Cout = s.Cout, HW = s.H * s.W;
  cl_kernel k;
  try {
    k = rt.buildKernel("conv1x1", "conv1x1_f16", c.options());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }
  std::vector<uint16_t> hW((size_t)Cout * Cin), hX((size_t)Cin * HW), hB((size_t)Cout), hY((size_t)Cout * HW);
  std::mt19937 rngW(11), rngX(22), rngB(33);
  std::uniform_real_distribution<float> disW(-0.5f, 0.5f), disX(-0.5f, 0.5f), disB(-0.2f, 0.2f);
  for (auto & v : hW) v = infvino::f32_to_f16(disW(rngW));
  for (auto & v : hX) v = infvino::f32_to_f16(disX(rngX));
  for (auto & v : hB) v = infvino::f32_to_f16(disB(rngB));

  cl_mem dW = rt.alloc((size_t)Cout * Cin * 2, CL_MEM_READ_ONLY);
  cl_mem dX = rt.alloc((size_t)Cin * HW * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)Cout * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc((size_t)Cout * HW * 2, CL_MEM_WRITE_ONLY);
  {  // repack W [Cout][Cin] -> [Cin][Cout] (contiguous output-channel reads)
    std::vector<uint16_t> hWt((size_t)Cin * Cout);
    for (int oc = 0; oc < Cout; ++oc)
      for (int ci = 0; ci < Cin; ++ci) hWt[(size_t)ci * Cout + oc] = hW[(size_t)oc * Cin + ci];
    rt.write(dW, (size_t)Cout * Cin * 2, hWt.data());
  }
  rt.write(dX, (size_t)Cin * HW * 2, hX.data());
  rt.write(dB, (size_t)Cout * 2, hB.data());

  cl_mem dRes = nullptr;
  clSetKernelArg(k, 0, sizeof(dW), &dW);
  clSetKernelArg(k, 1, sizeof(dX), &dX);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dRes), &dRes);
  clSetKernelArg(k, 4, sizeof(dY), &dY);
  int CinA = Cin, CoutA = Cout, HWA = HW;
  clSetKernelArg(k, 5, sizeof(CinA), &CinA);
  clSetKernelArg(k, 6, sizeof(CoutA), &CoutA);
  clSetKernelArg(k, 7, sizeof(HWA), &HWA);

  const size_t lws[2] = {16, 1};
  const size_t gws[2] = {
    static_cast<size_t>(((HW + c.TN - 1) / c.TN + lws[0] - 1) / lws[0] * lws[0]),
    static_cast<size_t>((Cout + c.TM - 1) / c.TM)};

  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws); }, 3, iters);
  const double flops = 2.0 * Cout * Cin * (double)HW;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  conv1x1 %-20s Cin=%-4d Cout=%-4d HW=%-6d %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), Cin, Cout, HW, med, flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);

  if (verify) {
    rt.read(dY, (size_t)Cout * HW * 2, hY.data());
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    for (int oc = 0; oc < Cout; ++oc)
      for (int n = 0; n < HW; ++n) {
        float acc = infvino::f16_to_f32(hB[oc]);
        for (int ci = 0; ci < Cin; ++ci)
          acc += infvino::f16_to_f32(hW[(size_t)oc * Cin + ci]) * infvino::f16_to_f32(hX[(size_t)ci * HW + n]);
        const double got = infvino::f16_to_f32(hY[(size_t)oc * HW + n]);
        const double d = std::fabs(got - acc);
        sumabs += d; sumref += std::fabs((double)acc);
        maxabs = std::max(maxabs, d); refmax = std::max(refmax, std::fabs((double)acc));
      }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dW); clReleaseMemObject(dX); clReleaseMemObject(dB);
  clReleaseMemObject(dY); clReleaseKernel(k);
  return 0;
}

// Specialized split-K GEMV for 1x1 conv / fc with HW==1 (fused bias + activation).
int benchConv1x1Gemv(infvino::ClRuntime & rt, const infvino::Conv1x1Cfg & c, const ConvShape & s, int iters, bool verify)
{
  const int Cin = s.Cin, Cout = s.Cout;
  cl_kernel k;
  try {
    k = rt.buildKernel("conv1x1", "conv1x1_gemv_f16", c.options());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }
  std::vector<uint16_t> hW((size_t)Cout * Cin), hX((size_t)Cin), hB((size_t)Cout), hY((size_t)Cout);
  std::mt19937 rngW(11), rngX(22), rngB(33);
  std::uniform_real_distribution<float> disW(-0.5f, 0.5f), disX(-0.5f, 0.5f), disB(-0.2f, 0.2f);
  for (auto & v : hW) v = infvino::f32_to_f16(disW(rngW));
  for (auto & v : hX) v = infvino::f32_to_f16(disX(rngX));
  for (auto & v : hB) v = infvino::f32_to_f16(disB(rngB));

  cl_mem dW = rt.alloc((size_t)Cout * Cin * 2, CL_MEM_READ_ONLY);
  cl_mem dX = rt.alloc((size_t)Cin * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)Cout * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc((size_t)Cout * 2, CL_MEM_WRITE_ONLY);
  rt.write(dW, (size_t)Cout * Cin * 2, hW.data());
  rt.write(dX, (size_t)Cin * 2, hX.data());
  rt.write(dB, (size_t)Cout * 2, hB.data());

  cl_mem dRes = nullptr;
  clSetKernelArg(k, 0, sizeof(dW), &dW);
  clSetKernelArg(k, 1, sizeof(dX), &dX);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dRes), &dRes);
  clSetKernelArg(k, 4, sizeof(dY), &dY);
  int CinA = Cin, CoutA = Cout;
  clSetKernelArg(k, 5, sizeof(CinA), &CinA);
  clSetKernelArg(k, 6, sizeof(CoutA), &CoutA);

  const size_t lws[1] = {16};
  const size_t gws[1] = {static_cast<size_t>((Cout + c.GEMV_TM - 1) / c.GEMV_TM) * 16};
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, gws, lws); }, 3, iters);
  const double flops = 2.0 * Cout * Cin;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  gemv    %-20s Cin=%-4d Cout=%-4d           %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), Cin, Cout, med, flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);
  if (verify) {
    rt.read(dY, (size_t)Cout * 2, hY.data());
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    for (int oc = 0; oc < Cout; ++oc) {
      float acc = infvino::f16_to_f32(hB[oc]);
      for (int ci = 0; ci < Cin; ++ci)
        acc += infvino::f16_to_f32(hW[(size_t)oc * Cin + ci]) * infvino::f16_to_f32(hX[ci]);
      const double got = infvino::f16_to_f32(hY[oc]);
      const double d = std::fabs(got - acc);
      sumabs += d; sumref += std::fabs((double)acc);
      maxabs = std::max(maxabs, d); refmax = std::max(refmax, std::fabs((double)acc));
    }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dW); clReleaseMemObject(dX); clReleaseMemObject(dB);
  clReleaseMemObject(dY); clReleaseKernel(k);
  return 0;
}

// Blocked (b_fs_yx_fsv16) 1x1 conv — port of OV `convolution_gpu_bfyx_f16_1x1`.
// Lane=output channel, os_is_yx_isv16_osv16 weights, X_BLOCK output columns/lane.
int benchConv1x1Blk(infvino::ClRuntime & rt, const infvino::Conv1x1BlkCfg & c,
                    const ConvShape & s, int iters, bool verify)
{
  const int Cin = s.Cin, Cout = s.Cout, H = s.H, W = s.W, HW = H * W;
  const std::string oo = c.options();
  cl_kernel k;
  try { k = rt.buildKernel("conv1x1_blk", "conv1x1_blk", oo); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }

  const int ocb = (Cout + 15) / 16, icb = (Cin + 15) / 16;
  std::vector<uint16_t> hW((size_t)Cout * Cin), hX((size_t)Cin * HW), hB((size_t)Cout),
      hY((size_t)Cout * HW), hRes((size_t)Cout * HW);
  std::mt19937 rngW(11), rngX(22), rngB(33);
  std::uniform_real_distribution<float> disW(-0.5f, 0.5f), disX(-0.5f, 0.5f), disB(-0.2f, 0.2f);
  for (auto & v : hW) v = infvino::f32_to_f16(disW(rngW));
  for (auto & v : hX) v = infvino::f32_to_f16(disX(rngX));
  for (auto & v : hB) v = infvino::f32_to_f16(disB(rngB));
  for (auto & v : hRes) v = infvino::f32_to_f16(disX(rngX));

  // weights -> os_is_yx_isv16_osv16 [Cout/16][Cin/16][isv16][osv16]
  std::vector<uint16_t> hWb((size_t)ocb * icb * 16 * 16, 0);
  for (int oc = 0; oc < Cout; ++oc)
    for (int ic = 0; ic < Cin; ++ic)
      hWb[(((size_t)(oc / 16) * icb + (ic / 16)) * 16 + (ic % 16)) * 16 + (oc % 16)] =
          hW[(size_t)oc * Cin + ic];
  // input -> b_fs_yx_fsv16 [Cin/16][H][W][16]（IN_NCHW=1 时直接喂 NCHW [Cin][H][W]）
  std::vector<uint16_t> hXb((size_t)icb * HW * 16, 0);
  if (!c.IN_NCHW)
    for (int ch = 0; ch < Cin; ++ch)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          hXb[(((size_t)(ch / 16) * H + y) * W + x) * 16 + (ch % 16)] =
              hX[(size_t)ch * HW + y * W + x];

  cl_mem dW = rt.alloc(hWb.size() * 2, CL_MEM_READ_ONLY);
  cl_mem dX = c.IN_NCHW ? rt.alloc((size_t)Cin * HW * 2, CL_MEM_READ_ONLY)
                        : rt.alloc(hXb.size() * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)Cout * 2, CL_MEM_READ_ONLY);
  const size_t ybytes = c.OUT_FSV16 ? (size_t)ocb * HW * 16 * 2 : (size_t)Cout * HW * 2;
  cl_mem dY = rt.alloc(ybytes, CL_MEM_READ_WRITE);
  rt.write(dW, hWb.size() * 2, hWb.data());
  rt.write(dX, c.IN_NCHW ? (size_t)Cin * HW * 2 : hXb.size() * 2,
           c.IN_NCHW ? hX.data() : hXb.data());
  rt.write(dB, (size_t)Cout * 2, hB.data());

  cl_mem dRes = nullptr;
  if (c.RES) {
    dRes = rt.alloc((size_t)Cout * HW * 2, CL_MEM_READ_ONLY);
    rt.write(dRes, (size_t)Cout * HW * 2, hRes.data());
  }

  clSetKernelArg(k, 0, sizeof(dX), &dX);
  clSetKernelArg(k, 1, sizeof(dW), &dW);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dY), &dY);
  clSetKernelArg(k, 4, sizeof(dRes), &dRes);
  int CinA = Cin, HA = H, WA = W, CoutA = Cout;
  clSetKernelArg(k, 5, sizeof(CinA), &CinA);
  clSetKernelArg(k, 6, sizeof(HA), &HA);
  clSetKernelArg(k, 7, sizeof(WA), &WA);
  clSetKernelArg(k, 8, sizeof(CoutA), &CoutA);

  const int yb = c.YB > 0 ? c.YB : 1;   // R48 D1
  const size_t lws[3] = {1, static_cast<size_t>(16 * c.SLM_DIV), 1};
  const size_t gws[3] = {
    static_cast<size_t>(((W + c.XB - 1) / c.XB) * ((H + yb - 1) / yb)),
    static_cast<size_t>(((Cout + 15) / 16) * lws[1]), 1};
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws); }, 3, iters);
  const double flops = 2.0 * Cout * Cin * (double)HW;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  conv1x1blk %-16s Cin=%-4d Cout=%-4d %dx%d XB%d YB%d slm%d%s  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), Cin, Cout, H, W, c.XB, c.YB, c.SLM_DIV, c.IN_NCHW ? " innchw" : "", med,
    flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);

  if (verify) {
    std::vector<uint16_t> raw((size_t)ocb * HW * 16);
    rt.read(dY, std::min<size_t>(raw.size() * 2, ybytes), raw.data());
    auto act = [&](double f) -> double {
      switch (c.ACT) {
        case 1: return f / (1.0 + std::exp(-f));
        case 2: return std::max(f, 0.0);
        case 3: return f * std::min(std::max(f + 3.0, 0.0), 6.0) / 6.0;
        case 4: return std::min(std::max(f + 3.0, 0.0), 6.0) / 6.0;
        case 5: return 1.0 / (1.0 + std::exp(-f));
        default: return f;
      }
    };
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    for (int oc = 0; oc < Cout; ++oc)
      for (int n = 0; n < HW; ++n) {
        float acc = infvino::f16_to_f32(hB[oc]);
        for (int ic = 0; ic < Cin; ++ic)
          acc += infvino::f16_to_f32(hW[(size_t)oc * Cin + ic]) * infvino::f16_to_f32(hX[(size_t)ic * HW + n]);
        const size_t oidx = c.OUT_FSV16
          ? ((((size_t)(oc / 16) * H + n / W) * W + n % W) * 16 + (oc % 16))
          : ((size_t)oc * HW + n);
        const double got = infvino::f16_to_f32(
            raw[oidx]);  // raw is fsv16-sized; bfyx index still within it
        const double r = act(acc) + (c.RES ? infvino::f16_to_f32(hRes[(size_t)oc * HW + n]) : 0.0);
        const double d = std::fabs(got - r);
        sumabs += d; sumref += std::fabs(r);
        maxabs = std::max(maxabs, d); refmax = std::max(refmax, std::fabs(r));
      }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dW); clReleaseMemObject(dX); clReleaseMemObject(dB);
  clReleaseMemObject(dY); if (dRes) clReleaseMemObject(dRes); clReleaseKernel(k);
  return 0;
}

// Blocked (b_fs_yx_fsv16) depthwise — port of OV `convolution_gpu_bfyx_f16_depthwise`.
int benchDepthwiseBlk(infvino::ClRuntime & rt, const infvino::DepthwiseBlkCfg & c,
                      const ConvShape & s, int iters, bool verify)
{
  const int C = s.Cin, H = s.H, W = s.W, K = c.K, S = c.S, P = c.P;
  const int Ho = (H + 2 * P - K) / S + 1, Wo = (W + 2 * P - K) / S + 1;
  cl_kernel k;
  try { k = rt.buildKernel("depthwise_blk", "depthwise_blk", c.options()); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }

  const int cb = (C + 15) / 16;
  std::vector<uint16_t> hW((size_t)C * K * K), hX((size_t)C * H * W), hB((size_t)C),
      hY((size_t)C * Ho * Wo);
  std::mt19937 rngW(11), rngX(22), rngB(33);
  std::uniform_real_distribution<float> disW(-0.5f, 0.5f), disX(-0.5f, 0.5f), disB(-0.2f, 0.2f);
  for (auto & v : hW) v = infvino::f32_to_f16(disW(rngW));
  for (auto & v : hX) v = infvino::f32_to_f16(disX(rngX));
  for (auto & v : hB) v = infvino::f32_to_f16(disB(rngB));

  // weights -> [C/16][K][K][16]
  std::vector<uint16_t> hWb((size_t)cb * K * K * 16, 0);
  for (int ch = 0; ch < C; ++ch)
    for (int kk = 0; kk < K * K; ++kk)
      hWb[((size_t)(ch / 16) * K * K + kk) * 16 + (ch % 16)] = hW[(size_t)ch * K * K + kk];
  // input -> [C/16][H][W][16]
  std::vector<uint16_t> hXb((size_t)cb * H * W * 16, 0);
  for (int ch = 0; ch < C; ++ch)
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x)
        hXb[(((size_t)(ch / 16) * H + y) * W + x) * 16 + (ch % 16)] =
            hX[((size_t)ch * H + y) * W + x];

  const size_t ybytes = c.OUT_FSV16 ? (size_t)cb * Ho * Wo * 16 * 2 : (size_t)C * Ho * Wo * 2;
  cl_mem dW = rt.alloc(hWb.size() * 2, CL_MEM_READ_ONLY);
  cl_mem dX = rt.alloc(hXb.size() * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)C * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc(ybytes, CL_MEM_READ_WRITE);
  rt.write(dW, hWb.size() * 2, hWb.data());
  rt.write(dX, hXb.size() * 2, hXb.data());
  rt.write(dB, (size_t)C * 2, hB.data());

  clSetKernelArg(k, 0, sizeof(dX), &dX);
  clSetKernelArg(k, 1, sizeof(dW), &dW);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dY), &dY);
  int CA = C, HA = H, WA = W, ho = Ho, wo = Wo;
  clSetKernelArg(k, 4, sizeof(CA), &CA);
  clSetKernelArg(k, 5, sizeof(HA), &HA);
  clSetKernelArg(k, 6, sizeof(WA), &WA);
  clSetKernelArg(k, 7, sizeof(ho), &ho);
  clSetKernelArg(k, 8, sizeof(wo), &wo);

  const size_t lws[3] = {1, 16, 1};
  const int yb = c.YB > 0 ? c.YB : 1;
  const size_t gws[3] = {static_cast<size_t>(((Wo + c.XB - 1) / c.XB) * ((Ho + yb - 1) / yb)),
                         static_cast<size_t>(((C + 15) / 16) * 16), 1};
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws); }, 3, iters);
  const double flops = 2.0 * C * Ho * Wo * K * K;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  depthwiseblk %-14s C=%-4d %dx%d K%d s%d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 32)",
    s.label.c_str(), C, H, W, K, S, med, flops / (med * 1e-3) / 1e9, ops, ops / 32 * 100);

  if (verify) {
    std::vector<uint16_t> raw((size_t)cb * Ho * Wo * 16);
    rt.read(dY, std::min<size_t>(raw.size() * 2, ybytes), raw.data());
    auto act = [&](double f) -> double {
      // 规范激活码（与 depthwise_blk.cl 的 dwblk_activate 一致）：1=SiLU 2=ReLU 3=HardSwish
      // 4=HardSigmoid。旧版把 2/3 写反（2=HardSwish,3=ReLU），使 --verify 在 act3 假 FAIL。
      switch (c.ACT) {
        case 1: return f / (1.0 + std::exp(-f));
        case 2: return std::max(f, 0.0);
        case 3: return f * std::min(std::max(f + 3.0, 0.0), 6.0) / 6.0;
        case 4: return std::min(std::max(f + 3.0, 0.0), 6.0) / 6.0;
        default: return f;
      }
    };
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    for (int ch = 0; ch < C; ++ch)
      for (int oy = 0; oy < Ho; ++oy)
        for (int ox = 0; ox < Wo; ++ox) {
          float acc = infvino::f16_to_f32(hB[ch]);
          for (int kh = 0; kh < K; ++kh)
            for (int kw = 0; kw < K; ++kw) {
              const int yy = oy * S - P + kh, xx = ox * S - P + kw;
              if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                acc += infvino::f16_to_f32(hX[((size_t)ch * H + yy) * W + xx]) *
                       infvino::f16_to_f32(hW[((size_t)ch * K + kh) * K + kw]);
            }
          const size_t oidx = c.OUT_FSV16
            ? ((((size_t)(ch / 16) * Ho + oy) * Wo + ox) * 16 + (ch % 16))
            : (((size_t)ch * Ho + oy) * Wo + ox);
          const double got = infvino::f16_to_f32(raw[oidx]);
          const double r = act(acc);
          const double d = std::fabs(got - r);
          sumabs += d; sumref += std::fabs(r);
          maxabs = std::max(maxabs, d); refmax = std::max(refmax, std::fabs(r));
        }
    std::printf("  mean_rel=%.3e max_rel(amax)=%.3e max_abs=%.2e",
      sumabs / (sumref + 1e-12), maxabs / (refmax + 1e-12), maxabs);
  }
  std::printf("\n");
  clReleaseMemObject(dW); clReleaseMemObject(dX); clReleaseMemObject(dB);
  clReleaseMemObject(dY); clReleaseKernel(k);
  return 0;
}

int benchBandwidth(infvino::ClRuntime & rt, size_t mb, int iters)
{
  // 安全上限：iGPU 共享 host 内存，in+out=2x，再加 host 侧缓冲，过大直接 OOM/死机。
  if (mb > 256) {
    std::fprintf(stderr, "[bandwidth] mb=%zu too large (iGPU shares host RAM); clamp to 256\n", mb);
    mb = 256;
  }
  const size_t n = mb * 1024 * 1024 / 4;  // uint elements
  cl_kernel kcopy = rt.buildKernel("stream", "copy_u32");
  cl_kernel kread = rt.buildKernel("stream", "readonly_u32");
  cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
  cl_mem out = rt.alloc(n * 4, CL_MEM_WRITE_ONLY);
  {
    std::vector<uint32_t> z(n, 1);
    rt.write(in, n * 4, z.data());
  }
  const uint nn = static_cast<uint>(n);
  clSetKernelArg(kcopy, 0, sizeof(in), &in);
  clSetKernelArg(kcopy, 1, sizeof(out), &out);
  clSetKernelArg(kcopy, 2, sizeof(nn), &nn);
  clSetKernelArg(kread, 0, sizeof(in), &in);
  clSetKernelArg(kread, 1, sizeof(out), &out);
  clSetKernelArg(kread, 2, sizeof(nn), &nn);
  const size_t gws = n, lws = 256;

  const double tc = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), kcopy, 1, &gws, &lws); }, 3, iters);
  const double tr = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), kread, 1, &gws, &lws); }, 3, iters);
  const double gb = static_cast<double>(n) * 4 / 1e9;
  std::printf("  copy : %.3f ms -> %.1f GB/s (read+write)\n", tc, 2 * gb / (tc * 1e-3));
  std::printf("  read : %.3f ms -> %.1f GB/s (read only)\n", tr, gb / (tr * 1e-3));
  clReleaseMemObject(in);
  clReleaseMemObject(out);
  clReleaseKernel(kcopy);
  clReleaseKernel(kread);
  return 0;
}

// ---------------------------------------------------------------------------
// Round 10 micro-benchmarks: ALU latency/throughput + cache hierarchy.
// ---------------------------------------------------------------------------

// Report per-dependent-FMA latency. For DEPTH=1 the loop is one chain, so
// cycles/iter is the FMA latency; for large DEPTH it approaches 1 (throughput).
int benchFma(infvino::ClRuntime & rt, const std::string & width, int depth, int iters, int fi, int sg)
{
  std::string kname;
  int lanes = 1;
  if (width == "f32") { kname = "fma_f32_lat"; lanes = 1; }
  else if (width == "h1") { kname = "fma_h1_lat"; lanes = 1; }
  else if (width == "h2") { kname = "fma_h2_lat"; lanes = 2; }
  else if (width == "h4") { kname = "fma_h4_lat"; lanes = 4; }
  else if (width == "h8") { kname = "fma_h8_lat"; lanes = 8; }
  else if (width == "h8_1op") { kname = "fma_h8_1op"; lanes = 8; }
  else if (width == "add") { kname = "add_lat"; lanes = 1; }
  else { std::fprintf(stderr, "unknown width %s\n", width.c_str()); return 2; }
  const bool is_add = (width == "add");

  const int ITERS = 4096;
  std::string kname2 = is_add ? std::string("add_lat") : kname;
  std::string opts = "-DDEPTH=" + std::to_string(depth) + " -DITERS=" + std::to_string(ITERS) +
                     " -DSG=" + std::to_string(sg) +
                     " -cl-mad-enable -cl-fast-relaxed-math";
  cl_kernel k;
  try { k = rt.buildKernel("micro", kname2, opts); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }

  cl_mem out = rt.alloc(16 * 8, CL_MEM_WRITE_ONLY);
  const float a = 1.0001f, b = 1e-4f;
  if (is_add) {
    const int ia = 3;
    clSetKernelArg(k, 0, sizeof(out), &out);
    clSetKernelArg(k, 1, sizeof(ia), &ia);
  } else if (width == "f32") {
    clSetKernelArg(k, 0, sizeof(out), &out);
    clSetKernelArg(k, 1, sizeof(a), &a);
    clSetKernelArg(k, 2, sizeof(b), &b);
  } else {
    const uint16_t ah = infvino::f32_to_f16(a), bh = infvino::f32_to_f16(b);
    clSetKernelArg(k, 0, sizeof(out), &out);
    clSetKernelArg(k, 1, sizeof(ah), &ah);
    clSetKernelArg(k, 2, sizeof(bh), &bh);
  }

  const size_t lws = 64;
  const size_t gws = 64 * 512;  // plenty of work-items to fill the EUs
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, fi);
  // Aggregate FPU throughput of the whole device (the number comparable to the
  // gemm ops/EU/cyc): FLOPs = work-items x iters x depth x lanes x 2.
  const double flops = 2.0 * static_cast<double>(gws) * ITERS * depth * lanes;
  const double ops = rt.opsPerEuCycle(flops, med);
  // Per-work-item dependent-chain latency (with loop overhead): total cycles of
  // the kernel is the per-work-item chain time (all work-items run in parallel).
  const double total_cyc = med * 1e-3 * rt.info().clock_mhz * 1e6;
  const double cyc_per_iter = total_cyc / ITERS;
  const double cyc_per_dep_fma = cyc_per_iter / depth;
  std::printf("  fma %-6s depth=%-4d lanes=%-2d sg=%-2d %8.3f ms  ops/EU/cyc=%5.2f  "
              "chain=%.1f cyc/iter  %.2f cyc/dep-%s-FMA\n",
              width.c_str(), depth, lanes, sg, med, ops, cyc_per_iter, cyc_per_dep_fma,
              width.c_str());
  clReleaseMemObject(out);
  clReleaseKernel(k);
  return 0;
}

// Round 27: occupancy / register-pressure probe (see micro.cl:fma_cyc). Sweeps
// the grid size and reports aggregate ops/EU/cyc and per-mad cycles; the plateau
// is 32*T_res/L, the knee is where the EUs fill up.
int benchOcc(infvino::ClRuntime & rt, int depth, int sg, const std::string & width)
{
  const bool hi = (width == "h8");            // high-ILP variant
  const int eff_depth = hi ? 8 : depth;
  const int ITERS = hi ? 2048 : 256;
  const char * kname = hi ? "fma_h8_lat" : "fma_cyc";
  std::string opts = "-DDEPTH=" + std::to_string(eff_depth) + " -DITERS=" + std::to_string(ITERS) +
                     " -DSG=" + std::to_string(sg) + " -cl-mad-enable -cl-fast-relaxed-math";
  cl_kernel k;
  try { k = rt.buildKernel("micro", kname, opts); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
  cl_mem out = rt.alloc(16 * 8, CL_MEM_WRITE_ONLY);
  const uint16_t ah = infvino::f32_to_f16(1.0001f), bh = infvino::f32_to_f16(1e-4f);
  clSetKernelArg(k, 0, sizeof(out), &out);
  clSetKernelArg(k, 1, sizeof(ah), &ah);
  clSetKernelArg(k, 2, sizeof(bh), &bh);
  std::printf("  occ %s depth=%-3d sg=%-2d ITERS=%d\n", kname, eff_depth, sg, ITERS);
  for (int nwg : {8, 16, 24, 32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512}) {
    const size_t lws = 64, gws = static_cast<size_t>(nwg) * 64;
    const double med = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, 7);
    // h8: each "mad" is a half8 vector (8 packed halfs/lane) = 8x the FLOPs.
    const double fl = hi ? 8.0 : 1.0;
    const double flops = 2.0 * static_cast<double>(gws) * ITERS * eff_depth * fl;
    const double ops = rt.opsPerEuCycle(flops, med);
    const long subgrp = static_cast<long>(gws / 16);
    std::printf("    nwg=%-3d subgrp=%-5ld  %7.3f ms  ops/EU/cyc=%5.2f\n",
                nwg, subgrp, med, ops);
  }
  clReleaseMemObject(out);
  clReleaseKernel(k);
  return 0;
}

// R47: sweep work-group size + count to understand the EU saturation threshold
// (is it 80 raw threads/EU, or a fixed #work-groups / scheduler slots?). Uses
// fma_wgs built with -DWGS=<n> so the hardware WG-size is truthful.
int benchOccWgs(infvino::ClRuntime & rt, int depth, int sg)
{
  std::printf("[occwgs] DEPTH=%d sg=%d : ops/EU/cyc vs (nwg, WG size)\n", depth, sg);
  const int ITERS = 256;
  for (int wg : {16, 32, 64, 128}) {
    std::string opts = "-DDEPTH=" + std::to_string(depth) + " -DITERS=" + std::to_string(ITERS) +
                       " -DWGS=" + std::to_string(wg) + " -cl-mad-enable -cl-fast-relaxed-math";
    cl_kernel k;
    try { k = rt.buildKernel("micro", "fma_wgs", opts); }
    catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
    cl_mem out = rt.alloc(16, CL_MEM_WRITE_ONLY);
    const uint16_t ah = infvino::f32_to_f16(1.0001f), bh = infvino::f32_to_f16(1e-4f);
    clSetKernelArg(k, 0, sizeof(out), &out);
    clSetKernelArg(k, 1, sizeof(ah), &ah);
    clSetKernelArg(k, 2, sizeof(bh), &bh);
    std::printf("  WG=%d:\n", wg);
    for (int nwg : {16, 32, 48, 64, 80, 96, 112, 128, 160, 256, 512, 1024}) {
      const size_t lws = static_cast<size_t>(wg);
      const size_t gws = static_cast<size_t>(nwg) * lws;
      const double med = rt.timeMs(
        [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, 5);
      const double flops = 2.0 * static_cast<double>(gws) * ITERS * depth;
      const double ops = rt.opsPerEuCycle(flops, med);
      std::printf("    nwg=%-4d threads=%-6zu  %7.3f ms  ops/EU/cyc=%5.2f  (thr/EU=%.1f)\n",
                  nwg, gws, med, ops, static_cast<double>(gws) / 80.0);
    }
    clReleaseMemObject(out);
    clReleaseKernel(k);
  }
  return 0;
}

// Pointer chase: latency per access vs working-set size (bytes).
int benchMemLat(infvino::ClRuntime & rt, const std::vector<size_t> & sizes_kb, int fi)
{
  cl_kernel k = rt.buildKernel("micro", "mem_chase", "-cl-mad-enable");
  for (size_t kb : sizes_kb) {
    const size_t n = kb * 1024 / 4;  // uint elements
    std::vector<uint32_t> h(n);
    // stride permutation: idx -> (idx + step) with step coprime to n, so the
    // walk visits every cache line. step in elements; pick ~ n/2 rounded odd.
    uint32_t step = static_cast<uint32_t>(n / 2) | 1u;
    for (size_t i = 0; i < n; ++i) h[i] = static_cast<uint32_t>((i + step) % n);
    cl_mem tbl = rt.alloc(n * 4, CL_MEM_READ_ONLY);
    cl_mem sink = rt.alloc(65536 * 4, CL_MEM_WRITE_ONLY);
    rt.write(tbl, n * 4, h.data());
    const uint nn = static_cast<uint>(n);
    const uint iters = 4096;
    clSetKernelArg(k, 0, sizeof(tbl), &tbl);
    clSetKernelArg(k, 1, sizeof(sink), &sink);
    clSetKernelArg(k, 2, sizeof(nn), &nn);
    clSetKernelArg(k, 3, sizeof(iters), &iters);
    // Keep the launch latency-bound: few enough work-items that the EUs are not
    // over-subscribed (each work-item is a serial dependent chain).
    const size_t lws = 64, gws = 64 * 4;
    const double med = rt.timeMs(
      [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 1, fi);
    const double ns_per_acc = med * 1e6 / iters;
    const double cyc = ns_per_acc * rt.info().clock_mhz * 1e-3;
    std::printf("  memlat %6zu KB  %8.3f ms  %6.2f ns/access  %6.1f cyc/access\n",
                kb, med, ns_per_acc, cyc);
    clReleaseMemObject(tbl);
    clReleaseMemObject(sink);
  }
  clReleaseKernel(k);
  return 0;
}

// Streaming copy bandwidth vs footprint. A full copy touches every byte, so the
// traffic is exactly 2 x footprint per pass (1 read + 1 write). Sweeping the
// footprint reveals the cache tiers: fast while it fits in a level, then it
// drops to the next level's bandwidth.
int benchMemBw(infvino::ClRuntime & rt, const std::vector<size_t> & sizes_kb, int fi)
{
  cl_kernel k = rt.buildKernel("stream", "copy_u32", "-cl-mad-enable");
  for (size_t kb : sizes_kb) {
    const size_t n = kb * 1024 / 4;  // uint elements
    cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
    cl_mem out = rt.alloc(n * 4, CL_MEM_WRITE_ONLY);
    {
      std::vector<uint32_t> z(n, 1);
      rt.write(in, n * 4, z.data());
    }
    const uint nn = static_cast<uint>(n);
    clSetKernelArg(k, 0, sizeof(in), &in);
    clSetKernelArg(k, 1, sizeof(out), &out);
    clSetKernelArg(k, 2, sizeof(nn), &nn);
    const size_t lws = 256;
    const size_t gws = ((n + lws - 1) / lws) * lws;
    const double med = rt.timeMs(
      [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, fi);
    const double bytes = static_cast<double>(n) * 4.0 * 2.0;  // read + write
    std::printf("  membw %6zu KB  %8.3f ms  %7.1f GB/s (copy: read+write)\n",
                kb, med, bytes / (med * 1e-3) / 1e9);
    clReleaseMemObject(in);
    clReleaseMemObject(out);
  }
  clReleaseKernel(k);
  return 0;
}

// R30c: read-vs-write asymmetry at a given footprint (DRAM when large).
int benchReadWrite(infvino::ClRuntime & rt, size_t mb, int fi)
{
  const size_t n = mb * 1024 * 1024 / 4;  // float elements
  cl_kernel kw = rt.buildKernel("stream_rw", "wr_only", "-cl-mad-enable");
  cl_kernel kr = rt.buildKernel("stream_rw", "rd_only", "-cl-mad-enable");
  cl_kernel ks = rt.buildKernel("stream_rw", "rd_sum", "-cl-mad-enable");
  cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
  cl_mem out = rt.alloc(n * 4, CL_MEM_WRITE_ONLY);
  { std::vector<float> z(n, 1.0f); rt.write(in, n * 4, z.data()); }
  const uint nn = static_cast<uint>(n), C = 8;
  const size_t lws = 256, gws = ((n + lws - 1) / lws) * lws;
  const float c = 1.0f;
  clSetKernelArg(kw, 0, sizeof(out), &out); clSetKernelArg(kw, 1, sizeof(nn), &nn);
  clSetKernelArg(kw, 2, sizeof(c), &c);
  clSetKernelArg(kr, 0, sizeof(in), &in); clSetKernelArg(kr, 1, sizeof(out), &out);
  clSetKernelArg(kr, 2, sizeof(nn), &nn);
  const uint nout = static_cast<uint>(n / C);
  const size_t gs = ((nout + lws - 1) / lws) * lws;
  clSetKernelArg(ks, 0, sizeof(in), &in); clSetKernelArg(ks, 1, sizeof(out), &out);
  clSetKernelArg(ks, 2, sizeof(nout), &nout); clSetKernelArg(ks, 3, sizeof(C), &C);
  const double tw = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), kw, 1, &gws, &lws); }, 2, fi);
  const double tr = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), kr, 1, &gws, &lws); }, 2, fi);
  const double ts = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), ks, 1, &gs, &lws); }, 2, fi);
  const double wb = (double)n * 4 / 1e9;
  std::printf("  rw %zu MB  write-only %.3f ms %6.1f GB/s | read-only %.3f ms %6.1f GB/s | "
              "rd_sum(C=%u) %.3f ms (read %6.1f + write %4.1f GB/s)\n",
              mb, tw, wb / (tw * 1e-3), tr, wb / (tr * 1e-3), C, ts,
              wb / (ts * 1e-3), (wb / C) / (ts * 1e-3));
  clReleaseMemObject(in); clReleaseMemObject(out);
  clReleaseKernel(kw); clReleaseKernel(kr); clReleaseKernel(ks);
  return 0;
}

// Read-only bandwidth vs footprint with an internal pass loop, so even small
// footprints keep the launch long enough to be bandwidth-bound (L1/LLC tiers).
int benchScanBw(infvino::ClRuntime & rt, const std::vector<size_t> & sizes_kb, int fi)
{
  cl_kernel k = rt.buildKernel("micro", "scan_rep", "-cl-mad-enable");
  for (size_t kb : sizes_kb) {
    const size_t n = kb * 1024 / 4;  // uint elements
    cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
    cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
    {
      std::vector<uint32_t> z(n, 1);
      rt.write(in, n * 4, z.data());
    }
    const uint nn = static_cast<uint>(n), step = 1, passes = 64;
    clSetKernelArg(k, 0, sizeof(in), &in);
    clSetKernelArg(k, 1, sizeof(sink), &sink);
    clSetKernelArg(k, 2, sizeof(nn), &nn);
    clSetKernelArg(k, 3, sizeof(step), &step);
    clSetKernelArg(k, 4, sizeof(passes), &passes);
    const size_t lws = 256, gws = 256 * 64;  // more work-items -> more loads in flight
    const double med = rt.timeMs(
      [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, fi);
    const double bytes = static_cast<double>(n) * 4.0 * passes;
    std::printf("  scanbw %6zu KB  %8.3f ms  %7.1f GB/s (read only)\n",
                kb, med, bytes / (med * 1e-3) / 1e9);
    clReleaseMemObject(in);
    clReleaseMemObject(sink);
  }
  clReleaseKernel(k);
  return 0;
}

// SLM (on-die scratchpad) read bandwidth at a given per-WG allocation.
int benchSlmBw(infvino::ClRuntime & rt, int slm_kb, int iters, int fi, const std::string & width,
               int mode, int nwg)
{
  std::string ksrc = (width == "v4") ? "slm_bw_v4" : "slm_bw";
  std::string opts = "-DSLM_KB=" + std::to_string(slm_kb) + " -cl-mad-enable";
  if (width == "conf") {
    ksrc = "slm_conf";
    opts += " -DMODE=" + std::to_string(mode);
  }
  cl_kernel k;
  try { k = rt.buildKernel("micro", ksrc, opts); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
  cl_mem out = rt.alloc(4096 * 4, CL_MEM_WRITE_ONLY);
  const uint it = static_cast<uint>(iters);
  clSetKernelArg(k, 0, sizeof(out), &out);
  clSetKernelArg(k, 1, sizeof(it), &it);
  const size_t lws = 64, gws = 64 * static_cast<size_t>(nwg);
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, fi);
  // Bytes per work-item per iteration: base kernels do 4 x 4B, conf-mode does
  // 4B per load; MODE 3/4 read one uint4 = 16B. Count them.
  double bpw = 16.0;
  if (width == "conf") bpw = (mode == 3 || mode == 4) ? 16.0 : 4.0;
  const double bytes = bpw * gws * iters;
  std::printf("  slmbw %3d KB/WG %-4s mode=%d nwg=%-3d %8.3f ms  %7.1f GB/s (SLM read)\n",
              slm_kb, width.c_str(), mode, nwg, med, bytes / (med * 1e-3) / 1e9);
  clReleaseMemObject(out);
  clReleaseKernel(k);
  return 0;
}

// Work-group barrier + SLM round-trip cost (Round 12).
int benchBarrier(infvino::ClRuntime & rt, int wg, int nwg, int iters, int mode)
{
  cl_kernel k;
  try { k = rt.buildKernel("micro", "barrier_cost",
      "-DBWG=" + std::to_string(wg) + " -DBMODE=" + std::to_string(mode) + " -cl-mad-enable"); }
  catch (const std::exception & e) { std::fprintf(stderr, "[build-fail] %s\n", e.what()); return 1; }
  cl_mem out = rt.alloc(4096 * 4, CL_MEM_WRITE_ONLY);
  const uint it = static_cast<uint>(iters);
  clSetKernelArg(k, 0, sizeof(out), &out);
  clSetKernelArg(k, 1, sizeof(it), &it);
  const size_t lws = static_cast<size_t>(wg), gws = static_cast<size_t>(wg) * static_cast<size_t>(nwg);
  const double med = rt.timeMs(
    [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, 3);
  const double total_cyc = med * 1e-3 * rt.info().clock_mhz * 1e6;
  std::printf("  barrier wg=%-4d nwg=%-3d mode=%d  %8.3f ms  %6.1f cyc/iter\n",
              wg, nwg, mode, med, total_cyc / iters);
  clReleaseMemObject(out);
  clReleaseKernel(k);
  return 0;
}

// R47 calibration: occupancy footprint scan. Each WI keeps WS*4 bytes hot; the
// resident footprint = gws * WS*4 is swept by --nwg (WIs) to find the knee where
// bandwidth collapses (L3 capacity under occupancy). Also reports the linear
// scaling (bytes/(WI*pass)) to calibrate perWG_bytes.
int benchFootprintBw(infvino::ClRuntime & rt, int ws, int passes)
{
  const int maxWi = 4096;
  const int PASSES = 64;  // fixed scan length per WI (constant work; residency swept by WI count)
  (void)passes;
  const size_t n = static_cast<size_t>(maxWi) * ws * 16;  // cache-line strided
  cl_kernel k = rt.buildKernel("micro", "footprint_scan", "-cl-mad-enable");
  cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
  cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
  { std::vector<uint32_t> z(n, 1); rt.write(in, n * 4, z.data()); }
  const uint nn = static_cast<uint>(n), pp = static_cast<uint>(PASSES), wsu = static_cast<uint>(ws);
  clSetKernelArg(k, 0, sizeof(in), &in);
  clSetKernelArg(k, 1, sizeof(sink), &sink);
  clSetKernelArg(k, 2, sizeof(nn), &nn);
  clSetKernelArg(k, 3, sizeof(pp), &pp);
  clSetKernelArg(k, 4, sizeof(wsu), &wsu);
  const size_t lws = 64;
  std::printf("[footprint] ws=%d uints (%.1f KB/WI), passes=%d\n", ws, ws * 4.0 / 1024.0, PASSES);
  for (int wi : {64, 128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096}) {
    const size_t gws = static_cast<size_t>(wi);
    if (gws > static_cast<size_t>(maxWi)) break;
    const double med = rt.timeMs(
      [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, 7);
    const double bytes = static_cast<double>(wi) * ws * 4.0 * PASSES;
    const double gbps = bytes / (med * 1e-3) / 1e9;
    const double footMB = static_cast<double>(wi) * ws * 4.0 / 1e6;
    std::printf("  footprint wi=%-4d WGs=%-4zu residency=%6.2f MB  %7.3f ms  %7.1f GB/s\n",
                wi, gws / 64, footMB, med, gbps);
  }
  clReleaseMemObject(in);
  clReleaseMemObject(sink);
  clReleaseKernel(k);
  return 0;
}

// R55 L3 coupling calibration: 2D sweep (concurrency C=gws × per-WI window fp).
// Reports effective line throughput (touches×64B/time) at each (C, fp) so that
//   * fixed C, varying fp  -> pure **residency/miss** effect (R = C·fp·64);
//   * fixed R = C·fp, varying (C, fp) -> pure **concurrency/MLP** effect.
// passes=1 => streaming (no reuse); passes>1 => the per-WI window is reused.
int benchL3Couple(infvino::ClRuntime & rt, const std::vector<int> & wi_list,
                  const std::vector<int> & fp_list, int passes, int fi)
{
  const int maxWi = 8192, maxFp = 256;
  const size_t n = static_cast<size_t>(maxWi) * maxFp * 16;  // uint elements (line-strided)
  cl_kernel k = rt.buildKernel("micro", "l3_probe", "-cl-mad-enable");
  cl_mem in = rt.alloc(n * 4, CL_MEM_READ_ONLY);
  cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
  { std::vector<uint32_t> z(n, 1); rt.write(in, n * 4, z.data()); }
  const size_t lws = 64;
  std::printf("[l3couple] passes=%d  metric = effective line throughput (touch=64B)\n", passes);
  std::printf("        C\\fp");
  for (int fp : fp_list) std::printf("   fp=%-3d", fp);
  std::printf("\n");
  const uint pu = static_cast<uint>(passes);
  for (int w : wi_list)
  {
    if (w <= 0 || w > maxWi) continue;
    const size_t gws = static_cast<size_t>(w);
    std::printf("  C=%-6d", w);
    for (int fp : fp_list)
    {
      if (fp <= 0 || fp > maxFp) { std::printf("   %7s", "-"); continue; }
      const uint fpu = static_cast<uint>(fp);
      clSetKernelArg(k, 0, sizeof(in), &in);
      clSetKernelArg(k, 1, sizeof(sink), &sink);
      clSetKernelArg(k, 2, sizeof(fpu), &fpu);
      clSetKernelArg(k, 3, sizeof(pu), &pu);
      double med = 0.0;
      try {
        med = rt.timeMs(
          [&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, fi);
      } catch (const std::exception & e) { std::printf("   %7s", "ERR"); continue; }
      const double touches = static_cast<double>(w) * fp * passes;
      const double gbps = touches * 64.0 / (med * 1e-3) / 1e9;   // line-granular throughput
      std::printf(" %6.1f", gbps);
    }
    std::printf("   (R = C·fp·64)\n");
  }
  clReleaseMemObject(in);
  clReleaseMemObject(sink);
  clReleaseKernel(k);
  return 0;
}

// R55 two-tenant pollution probe: run an aggressor stream (private footprint
// R_A = agg_wi·agg_fp·64 B) immediately before cooperatively re-reading a SHARED
// hot set of `hot_lines` lines. timeMs() profiles the hot-read event only (the
// aggressor is enqueued on the same in-order queue just before, untimed), so the
// hot-read's effective throughput reveals whether the aggressor evicted the hot
// set. Baseline = no aggressor (fully L3-resident).
int benchL3Pollute(infvino::ClRuntime & rt, int hot_lines, int hot_gws, int hiters,
                   const std::vector<int> & agg_wi, const std::vector<int> & agg_fp, int fi)
{
  const int maxWi = 8192, maxFp = 256;
  const size_t hn = static_cast<size_t>(hot_lines) * 16 + 16;
  const size_t an = static_cast<size_t>(maxWi) * maxFp * 16;
  cl_mem hot = rt.alloc(hn * 4, CL_MEM_READ_ONLY);
  cl_mem agg = rt.alloc(an * 4, CL_MEM_READ_ONLY);
  cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
  { std::vector<uint32_t> z(hn, 1); rt.write(hot, hn * 4, z.data()); }
  { std::vector<uint32_t> z(an, 1); rt.write(agg, an * 4, z.data()); }
  cl_kernel khot = rt.buildKernel("micro", "l3_hot_read", "-cl-mad-enable");
  cl_kernel kagg = rt.buildKernel("micro", "l3_probe", "-cl-mad-enable");
  const size_t lws = 64;
  const uint hl = static_cast<uint>(hot_lines), hi = static_cast<uint>(hiters), one = 1u, zero = 0u;
  clSetKernelArg(khot, 0, sizeof(hot), &hot);
  clSetKernelArg(khot, 1, sizeof(sink), &sink);
  clSetKernelArg(khot, 2, sizeof(hl), &hl);
  clSetKernelArg(khot, 3, sizeof(hi), &hi);
  clSetKernelArg(khot, 4, sizeof(zero), &zero);
  const size_t hotGws = static_cast<size_t>(hot_gws);
  const double hotBytes = static_cast<double>(hot_lines) * hiters * 64.0;
  auto measure = [&](int agws, int afp) -> double {
    auto closure = [&]() -> cl_event {
      cl_event aev = nullptr;
      if (agws > 0)
      {
        const uint af = static_cast<uint>(afp);
        clSetKernelArg(kagg, 0, sizeof(agg), &agg);
        clSetKernelArg(kagg, 1, sizeof(sink), &sink);
        clSetKernelArg(kagg, 2, sizeof(af), &af);
        clSetKernelArg(kagg, 3, sizeof(one), &one);   // passes=1: pure stream
        size_t ag = static_cast<size_t>(agws);
        aev = infvino::ClRuntime::enqueueND(rt.queue(), kagg, 1, &ag, &lws);
      }
      cl_event hev = infvino::ClRuntime::enqueueND(rt.queue(), khot, 1, &hotGws, &lws);
      if (aev) clReleaseEvent(aev);
      return hev;
    };
    try { return rt.timeMs(closure, 2, fi); } catch (const std::exception &) { return 0.0; }
  };
  std::printf("[l3pollute] hot=%d lines (%.0f KB shared), hot_gws=%d, hiters=%d, "
              "hot_bytes=%.0f KB\n", hot_lines, hot_lines * 64.0 / 1024.0, hot_gws, hiters,
              hotBytes / 1024.0);
  std::printf("  aggressor            R_A       hot_read_ms   hot_GB/s   vs ref\n");
  // 参考点用**平台区中段**（R_A=1 MB，确定无污染）。两点伪影要避开：(a)「无 aggressor」
  // 背靠背单独 kernel 的 event 采样；(b) 首次测量的冷启动。先丢弃一次，再取参考。
  (void)measure(1024, 4);
  const double base = measure(4096, 4);
  const double baseGbs = base > 0 ? hotBytes / (base * 1e-3) / 1e9 : 0.0;
  std::printf("  %-16s %8s   %10.4f   %8.1f   1.00\n", "ref(R_A=1MB)", "1.05 MB", base, baseGbs);
  struct Row { double R; int wi, fp; double ms; };
  std::vector<Row> rows;
  for (int wi : agg_wi)
    for (int fp : agg_fp)
    {
      if (wi <= 0 || wi > maxWi || fp <= 0 || fp > maxFp) continue;
      const double ms = measure(wi, fp);
      rows.push_back({static_cast<double>(wi) * fp * 64.0, wi, fp, ms});
    }
  std::sort(rows.begin(), rows.end(), [](const Row & a, const Row & b) { return a.R < b.R; });
  for (const auto & r : rows)
  {
    const double gbs = r.ms > 0 ? hotBytes / (r.ms * 1e-3) / 1e9 : 0.0;
    std::printf("  wi=%-5d fp=%-3d  %8.2f MB   %10.4f   %8.1f   %.2f\n",
                r.wi, r.fp, r.R / 1e6, r.ms, gbs, base > 0 ? r.ms / base : 0.0);
  }
  clReleaseMemObject(hot);
  clReleaseMemObject(agg);
  clReleaseMemObject(sink);
  clReleaseKernel(khot);
  clReleaseKernel(kagg);
  return 0;
}

// R60: retention sweep — self-calibration infrastructure for the L3 replacement model.
// Machine-readable CSV: iters,R_A_MB,hot_ms,hot_gbps,retained (retained = ref_ms/hot_ms).
// Sweeps the *reuse count* (hot_iters) too, so a fitter can tell whether heavily-reused
// data resists streaming pollution (pLRU/NRU-like) or not (strict LRU).
int benchL3Retain(infvino::ClRuntime & rt, int hot_lines, int hot_gws, int hot_off,
                  const std::vector<int> & iters_list, const std::vector<int> & agg_wi,
                  const std::vector<int> & agg_fp, int fi)
{
  const int maxWi = 8192, maxFp = 256;
  const size_t hn = (static_cast<size_t>(hot_off) + hot_lines) * 16 + 16;
  const size_t an = static_cast<size_t>(maxWi) * maxFp * 16;
  cl_mem hot = rt.alloc(hn * 4, CL_MEM_READ_ONLY);
  cl_mem agg = rt.alloc(an * 4, CL_MEM_READ_ONLY);
  cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
  { std::vector<uint32_t> z(hn, 1); rt.write(hot, hn * 4, z.data()); }
  { std::vector<uint32_t> z(an, 1); rt.write(agg, an * 4, z.data()); }
  cl_kernel khot = rt.buildKernel("micro", "l3_hot_read", "-cl-mad-enable");
  cl_kernel kagg = rt.buildKernel("micro", "l3_probe", "-cl-mad-enable");
  const size_t lws = 64;
  const uint hl = static_cast<uint>(hot_lines), one = 1u, off = static_cast<uint>(hot_off);
  clSetKernelArg(khot, 0, sizeof(hot), &hot);
  clSetKernelArg(khot, 1, sizeof(sink), &sink);
  clSetKernelArg(khot, 2, sizeof(hl), &hl);
  clSetKernelArg(khot, 4, sizeof(off), &off);
  const size_t hotGws = static_cast<size_t>(hot_gws);
  std::printf("# l3retain hot=%d lines (%d KB) hot_gws=%d hot_off=%d  (self-calibration; no vendor docs)\n",
              hot_lines, hot_lines * 64 / 1024, hot_gws, hot_off);
  std::printf("iter,off,R_A_MB,hot_ms,hot_gbps,retained\n");
  for (int hi : iters_list)
  {
    if (hi <= 0) continue;
    const uint hiu = static_cast<uint>(hi);
    clSetKernelArg(khot, 3, sizeof(hiu), &hiu);
    const double hotBytes = static_cast<double>(hot_lines) * hi * 64.0;
    auto measure = [&](int agws, int afp) -> double {
      auto closure = [&]() -> cl_event {
        cl_event aev = nullptr;
        if (agws > 0)
        {
          const uint af = static_cast<uint>(afp);
          clSetKernelArg(kagg, 0, sizeof(agg), &agg);
          clSetKernelArg(kagg, 1, sizeof(sink), &sink);
          clSetKernelArg(kagg, 2, sizeof(af), &af);
          clSetKernelArg(kagg, 3, sizeof(one), &one);
          size_t ag = static_cast<size_t>(agws);
          aev = infvino::ClRuntime::enqueueND(rt.queue(), kagg, 1, &ag, &lws);
        }
        cl_event hev = infvino::ClRuntime::enqueueND(rt.queue(), khot, 1, &hotGws, &lws);
        if (aev) clReleaseEvent(aev);
        return hev;
      };
      try { return rt.timeMs(closure, 2, fi); } catch (const std::exception &) { return 0.0; }
    };
    (void)measure(1024, 4);                        // discard first (cold)
    const double base = measure(4096, 4);          // ref: R_A = 1 MB (no pollution)
    const double baseGbs = base > 0 ? hotBytes / (base * 1e-3) / 1e9 : 0.0;
    std::printf("%d,%d,1.05,%.6f,%.2f,1.000\n", hi, hot_off, base, baseGbs);
    struct Row { double R; int wi, fp; double ms; };
    std::vector<Row> rows;
    for (int wi : agg_wi)
      for (int fp : agg_fp)
      {
        if (wi <= 0 || wi > maxWi || fp <= 0 || fp > maxFp) continue;
        rows.push_back({static_cast<double>(wi) * fp * 64.0, wi, fp, measure(wi, fp)});
      }
    std::sort(rows.begin(), rows.end(), [](const Row & a, const Row & b) { return a.R < b.R; });
    for (const auto & r : rows)
    {
      const double gbs = r.ms > 0 ? hotBytes / (r.ms * 1e-3) / 1e9 : 0.0;
      std::printf("%d,%d,%.2f,%.6f,%.2f,%.3f\n", hi, hot_off, r.R / 1e6, r.ms, gbs,
                  base > 0 && r.ms > 0 ? base / r.ms : 0.0);
    }
  }
  clReleaseMemObject(hot);
  clReleaseMemObject(agg);
  clReleaseMemObject(sink);
  clReleaseKernel(khot);
  clReleaseKernel(kagg);
  return 0;
}

// R61: set/associativity geometry probe. Touch `nlines` lines at a fixed line stride,
// repeatedly; a stride sharing factors with the set count collapses the touched lines
// onto few sets -> associativity-limited thrash -> bandwidth drop.
int benchL3Conflict(infvino::ClRuntime & rt, const std::vector<int> & strides, int nlines,
                    int iters, int wi)
{
  const int maxStride = 8192;
  size_t maxUsed = 16;
  for (int s : strides) maxUsed = std::max(maxUsed, static_cast<size_t>(std::max(1, s)) * nlines);
  maxUsed += 16;
  const size_t kMaxLines = 64ull * 1024 * 1024;   // 4 GiB cap (uint32 elements)
  if (maxUsed > kMaxLines)
  {
    std::fprintf(stderr, "[l3conflict] refusing: span %zu lines exceeds cap\n", maxUsed);
    return 1;
  }
  cl_mem in = rt.alloc(maxUsed * 64, CL_MEM_READ_ONLY);
  cl_mem sink = rt.alloc(4, CL_MEM_WRITE_ONLY);
  { std::vector<uint32_t> z(maxUsed * 16, 1); rt.write(in, maxUsed * 64, z.data()); }
  cl_kernel k = rt.buildKernel("micro", "l3_stride", "-cl-mad-enable");
  const size_t lws = 64;
  std::printf("[l3conflict] nlines=%d wi=%d iters=%d\n", nlines, wi, iters);
  std::printf("  stride_lines  span_MB    ms     GB/s\n");
  for (int s : strides)
  {
    if (s < 1 || s > maxStride) continue;
    const uint nl = static_cast<uint>(nlines), sl = static_cast<uint>(s), it = static_cast<uint>(iters);
    clSetKernelArg(k, 0, sizeof(in), &in);
    clSetKernelArg(k, 1, sizeof(sink), &sink);
    clSetKernelArg(k, 2, sizeof(nl), &nl);
    clSetKernelArg(k, 3, sizeof(sl), &sl);
    clSetKernelArg(k, 4, sizeof(it), &it);
    const size_t gws = static_cast<size_t>(wi);
    double ms = 0.0;
    try { ms = rt.timeMs([&] { return infvino::ClRuntime::enqueueND(rt.queue(), k, 1, &gws, &lws); }, 2, 5); }
    catch (const std::exception &) { ms = 0.0; }
    const double acc = static_cast<double>(iters) * static_cast<double>(wi);
    const double bytes = acc * 64.0;   // each thread does one 64B line touch per iter
    std::printf("  %-13d %7.2f  %7.4f  %7.1f\n", s, static_cast<double>(nlines) * s * 64.0 / 1e6,
                ms, ms > 0 ? bytes / (ms * 1e-3) / 1e9 : 0.0);
  }
  clReleaseMemObject(in);
  clReleaseMemObject(sink);
  clReleaseKernel(k);
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string op = "gemm", kernel_dir = INFVINO_KERNEL_DIR;
  infvino::Tiles tiles;
  infvino::Conv3x3Cfg conv;
  infvino::Conv1x1Cfg c1x1;
  infvino::Conv1x1BlkCfg c1x1blk;
  infvino::DepthwiseBlkCfg dwblk;
  std::vector<ConvShape> conv_shapes;
  std::vector<Shape> shapes;
  int iters = 100;
  size_t mb = 256;
  bool verify = false;
  std::string width = "h1";
  int depth = 1;
  std::vector<size_t> sizes_kb;
  int slm_kb = 16, mode = 0, nwg = 64, wg = 256, sg = 0;
  std::vector<int> wi_list, fp_list;
  int passes = 32;
  int hot_lines = 16384, hot_gws = 4096, hot_iters = 1;
  int nseq = 8, seq_mode = 0;
  int hot_off = 0, nlines = 128;
  std::vector<int> hot_iters_list;
  std::vector<int> stride_list;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--list-devices") {
      for (const auto & d : infvino::ClRuntime::enumerate()) std::printf("  %s\n", d.describe().c_str());
      return 0;
    } else if (a == "--op") {
      op = next();
    } else if (a == "--kernel-dir") {
      kernel_dir = next();
    } else if (a == "--iters") {
      iters = std::atoi(next().c_str());
    } else if (a == "--mb") {
      mb = static_cast<size_t>(std::atoi(next().c_str()));
    } else if (a == "--verify") {
      verify = true;
    } else if (a == "--tiles") {
      tiles = infvino::parseTiles(next());
    } else if (a == "--conv") {
      conv = infvino::parseConv(next());
    } else if (a == "--conv1x1") {
      c1x1 = infvino::parseConv1x1(next());
    } else if (a == "--conv1x1blk") {
      c1x1blk = infvino::parseConv1x1Blk(next());
    } else if (a == "--depthwiseblk") {
      dwblk = infvino::parseDepthwiseBlk(next());
    } else if (a == "--conv-shape") {
      auto v = parseInts(next());
      if (v.size() != 4) { std::fprintf(stderr, "--conv-shape needs Cin,Cout,H,W\n"); return 2; }
      conv_shapes.push_back({v[0], v[1], v[2], v[3], "conv"});
    } else if (a == "--stride") {
      conv.STRIDE = std::atoi(next().c_str());
    } else if (a == "--pad") {
      conv.PAD = std::atoi(next().c_str());
    } else if (a == "--act") {
      conv.ACT = std::atoi(next().c_str());
    } else if (a == "--shape") {
      auto v = parseInts(next());
      if (v.size() != 3) { std::fprintf(stderr, "--shape needs M,N,K\n"); return 2; }
      shapes.push_back({v[0], v[1], v[2], "MxNxK"});
    } else if (a == "--width") {
      width = next();
    } else if (a == "--depth") {
      depth = std::atoi(next().c_str());
    } else if (a == "--sg") {
      sg = std::atoi(next().c_str());
      conv.SG = sg;
    } else if (a == "--sizes") {
      for (int v : parseInts(next())) sizes_kb.push_back(static_cast<size_t>(v));
    } else if (a == "--wi") {
      wi_list = parseInts(next());
    } else if (a == "--fp") {
      fp_list = parseInts(next());
    } else if (a == "--passes") {
      passes = std::atoi(next().c_str());
    } else if (a == "--hot-lines") {
      hot_lines = std::atoi(next().c_str());
    } else if (a == "--hot-gws") {
      hot_gws = std::atoi(next().c_str());
    } else if (a == "--hot-iters") {
      hot_iters = std::atoi(next().c_str());
    } else if (a == "--hot-sweep") {
      hot_iters_list = parseInts(next());
    } else if (a == "--hot-offset") {
      hot_off = std::atoi(next().c_str());
    } else if (a == "--nlines") {
      nlines = std::atoi(next().c_str());
    } else if (a == "--line-stride") {
      stride_list = parseInts(next());
    } else if (a == "--nseq") {
      nseq = std::atoi(next().c_str());
    } else if (a == "--seq-mode") {
      seq_mode = std::atoi(next().c_str());
    } else if (a == "--slm-kb") {
      slm_kb = std::atoi(next().c_str());
    } else if (a == "--mode") {
      mode = std::atoi(next().c_str());
    } else if (a == "--nwg") {
      nwg = std::atoi(next().c_str());
    } else if (a == "--wg") {
      wg = std::atoi(next().c_str());
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  infvino::ClRuntime rt(kernel_dir);
  std::printf("device     : %s\n", rt.info().describe().c_str());
  std::printf("peak FP16  : %.1f GFLOP/s (EU x clk x 32)\n\n", rt.info().peak_fp16_gflops);

  if (shapes.empty()) shapes = {{1024, 1024, 1024, "square"}, {512, 512, 512, "square"}};

  int rc = 0;
  if (op == "gemm") {
    std::printf("[gemm] tiles %s\n", tiles.label().c_str());
    for (const auto & s : shapes) rc |= benchGemm(rt, tiles, s, iters, verify);
  } else if (op == "gemm_sk") {
    std::printf("[gemm_sk] split-K %s\n", c1x1.label().c_str());
    for (const auto & s : shapes) rc |= benchGemmSk(rt, c1x1, s, iters, verify);
  } else if (op == "chain") {
    rc |= benchChain(rt, iters);
  } else if (op == "conv1x1") {
    std::printf("[conv1x1] (== gemm: M=Cout, N=H*W, K=Cin) %s\n", tiles.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{64, 64, 80, 80, "c1x1"}, {256, 256, 20, 20, "c1x1"}};
    for (const auto & s : conv_shapes) {
      Shape g{s.Cout, s.H * s.W, s.Cin, "conv1x1"};
      rc |= benchGemm(rt, tiles, g, iters, verify);
    }
  } else if (op == "conv1x1k") {
    std::printf("[conv1x1k] %s\n", c1x1.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{64, 64, 80, 80, "c1x1"}, {96, 576, 7, 7, "c1x1-mb"}};
    for (const auto & s : conv_shapes) rc |= benchConv1x1(rt, c1x1, s, iters, verify);
  } else if (op == "conv1x1g") {    std::printf("[conv1x1g] gemv %s\n", c1x1.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{576, 1024, 1, 1, "fc"}, {1024, 1000, 1, 1, "cls"}};
    for (const auto & s : conv_shapes) rc |= benchConv1x1Gemv(rt, c1x1, s, iters, verify);
  } else if (op == "conv1x1blk") {
    std::printf("[conv1x1blk] fsv16 %s\n", c1x1blk.label().c_str());
    if (conv_shapes.empty())
      conv_shapes = {{576, 96, 7, 7, "mb-exp"}, {96, 576, 7, 7, "mb-proj"},
                     {240, 40, 14, 14, "mb"}, {64, 64, 80, 80, "big"}};
    for (const auto & s : conv_shapes) rc |= benchConv1x1Blk(rt, c1x1blk, s, iters, verify);
  } else if (op == "depthwiseblk") {
    std::printf("[depthwiseblk] fsv16 %s\n", dwblk.label().c_str());
    if (conv_shapes.empty())
      conv_shapes = {{16, 16, 112, 112, "mb-s2"}, {96, 96, 56, 56, "mb"},
                     {240, 240, 28, 28, "mb"}, {576, 576, 14, 14, "mb"}};
    for (const auto & s : conv_shapes) rc |= benchDepthwiseBlk(rt, dwblk, s, iters, verify);
  } else if (op == "reorder") {
    std::printf("[reorder] bfyx -> b_fs_yx_fsv16\n");
    if (conv_shapes.empty())
      conv_shapes = {{64, 0, 82, 82, "80s1_64"}, {16, 0, 162, 162, "160s1_16"},
                     {16, 0, 322, 322, "160s2_16"}, {3, 0, 322, 322, "stem"}};
    for (const auto & s : conv_shapes) rc |= benchReorder(rt, s, iters);
  } else if (op == "reorder_seq") {
    std::printf("[reorder_seq] additivity probe (--nseq N --seq-mode 0|cold 1|hot)\n");
    if (conv_shapes.empty())
      conv_shapes = {{64, 0, 82, 82, "80s1_64"}, {16, 0, 162, 162, "160s1_16"}};
    for (const auto & s : conv_shapes) rc |= benchReorderSeq(rt, s, nseq, seq_mode, iters);
  } else if (op == "conv3x3" || op == "conv3x3rt" || op == "conv3x3osv" || op == "conv3x3sg" || op == "conv3x3db" || op == "conv3x3ov" || op == "conv3x3blk") {
    if (op == "conv3x3rt") conv.RT = 1;
    if (op == "conv3x3osv") conv.OSV = 1;
    if (op == "conv3x3sg") conv.SGK = 1;
    if (op == "conv3x3db") conv.DB = 1;
    if (op == "conv3x3ov") conv.OV = 1;
    if (op == "conv3x3blk") conv.BLK = 1;
    std::printf("[%s] %s\n", op.c_str(), conv.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{64, 64, 80, 80, "p3-3x3"}, {64, 64, 40, 40, "p4-3x3"}};
    for (const auto & s : conv_shapes) {
      if (conv.BLK) rc |= benchConvBlk(rt, conv, s, iters, verify);
      else rc |= benchConv(rt, conv, s, iters, verify);
    }
  } else if (op == "bandwidth") {
    std::printf("[bandwidth] buffer=%zu MB\n", mb);
    rc = benchBandwidth(rt, mb, iters);
  } else if (op == "fmalat") {
    std::printf("[fmalat] width=%s depth=%d (depth=1 -> latency, large -> throughput)\n",
                width.c_str(), depth);
    rc = benchFma(rt, width, depth, iters, 3, sg);
  } else if (op == "occ") {
    std::printf("[occ] occupancy / register-pressure probe (--depth D --sg 16 --width h8|cyc)\n");
    if (sg == 0) sg = 16;
    rc = benchOcc(rt, depth, sg, width);
  } else if (op == "occwgs") {
    rc = benchOccWgs(rt, depth, sg == 0 ? 16 : sg);
  } else if (op == "memlat") {
    std::printf("[memlat] pointer-chase latency vs working set\n");
    if (sizes_kb.empty()) sizes_kb = {4, 16, 64, 256, 1024, 4096, 16384};
    rc = benchMemLat(rt, sizes_kb, 2);
  } else if (op == "membw") {
    std::printf("[membw] streaming read bandwidth vs footprint\n");
    if (sizes_kb.empty()) sizes_kb = {4, 16, 64, 256, 1024, 4096, 16384};
    rc = benchMemBw(rt, sizes_kb, 2);
  } else if (op == "rwbw") {
    std::printf("[rwbw] write vs read vs rd_sum asymmetry (footprint = --mb)\n");
    rc = benchReadWrite(rt, mb, 2);
  } else if (op == "scanbw") {
    std::printf("[scanbw] read-only bandwidth vs footprint (internal passes)\n");
    if (sizes_kb.empty()) sizes_kb = {4, 16, 64, 256, 512, 1024, 2048};
    rc = benchScanBw(rt, sizes_kb, 2);
  } else if (op == "slmbw") {
    std::printf("[slmbw] SLM read bandwidth\n");
    rc = benchSlmBw(rt, slm_kb, iters, 2, width, mode, nwg);
  } else if (op == "barrier") {
    std::printf("[barrier] work-group barrier decomposition (--wg --nwg --mode)\n");
    rc = benchBarrier(rt, wg, nwg, iters, mode);
  } else if (op == "footprint") {
    rc = benchFootprintBw(rt, depth, iters);
  } else if (op == "l3couple") {
    if (wi_list.empty()) wi_list = {256, 512, 1024, 2048, 4096, 8192};
    if (fp_list.empty()) fp_list = {1, 2, 4, 8, 16, 32, 64};
    rc = benchL3Couple(rt, wi_list, fp_list, passes, 2);
  } else if (op == "l3pollute") {
    if (wi_list.empty()) wi_list = {1024, 2048, 4096, 8192};
    if (fp_list.empty()) fp_list = {1, 2, 4, 8, 16, 32};
    rc = benchL3Pollute(rt, hot_lines, hot_gws, hot_iters, wi_list, fp_list, 5);
  } else if (op == "l3retain") {
    if (hot_iters_list.empty()) hot_iters_list = {1, 4, 16};
    if (wi_list.empty()) wi_list = {1024, 2048, 4096, 8192};
    if (fp_list.empty()) fp_list = {4, 8, 16, 24, 32, 48};
    rc = benchL3Retain(rt, hot_lines, hot_gws, hot_off, hot_iters_list, wi_list, fp_list, 5);
  } else if (op == "l3conflict") {
    if (stride_list.empty())
      stride_list = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048};
    rc = benchL3Conflict(rt, stride_list, nlines, passes, wi_list.empty() ? 1024 : wi_list[0]);
  } else {
    std::fprintf(stderr, "unknown op: %s\n", op.c_str());
    return 2;
  }
  return rc;
}
