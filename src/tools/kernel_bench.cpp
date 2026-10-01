// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_bench —— 自研 kernel 基准：延迟(ms) + ops/EU/cycle（FP16 上限 16）。
//
//   kernel_bench --list-devices
//   kernel_bench --op gemm --shape 1024,1024,1024 --verify
//   kernel_bench --op gemm --shape 6400,64,64 --shape 400,256,256 --tiles 64,64,16,8,4
//
// 设计：每个 case 只计 kernel 自身时间（OpenCL event profiling），与
// infvino_bench 的 “net only” 口径一致，便于逐轮对比。
#include <algorithm>
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
  for (auto & x : v) x = gk::f32_to_f16(dis(rng));
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
        acc += gk::f16_to_f32(A[(size_t)r * K + t]) * gk::f16_to_f32(B[(size_t)t * N + c]);
      const double got = gk::f16_to_f32(C[(size_t)r * N + c]);
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

int benchGemm(gk::ClRuntime & rt, const gk::Tiles & t, const Shape & s, int iters, bool verify)
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
    [&] { return gk::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws); }, 3, iters);
  const double flops = 2.0 * static_cast<double>(M) * N * K;
  const double ops = rt.opsPerEuCycle(flops, med);

  std::printf(
    "  gemm %-16s M=%-5d N=%-4d K=%-5d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 16)",
    s.label.c_str(), M, N, K, med, flops / (med * 1e-3) / 1e9, ops, ops / 16 * 100);

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

struct ConvShape
{
  int Cin, Cout, H, W;
  std::string label;
};

int benchConv(gk::ClRuntime & rt, const gk::Conv3x3Cfg & c, const ConvShape & s, int iters, bool verify)
{
  const int Hout = (s.H + 2 * c.PAD - 3) / c.STRIDE + 1;
  const int Wout = (s.W + 2 * c.PAD - 3) / c.STRIDE + 1;
  cl_kernel k;
  try {
    k = rt.buildKernel("conv", "conv3x3_f16", c.options());
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[build-fail] %s\n", e.what());
    return 1;
  }

  std::vector<uint16_t> hX((size_t)s.Cin * s.H * s.W), hB((size_t)s.Cout);
  std::vector<uint16_t> hWt((size_t)s.Cout * s.Cin * 9), hY((size_t)s.Cout * Hout * Wout);
  std::mt19937 rng123(11), rng456(22);
  std::uniform_real_distribution<float> dum(-0.5f, 0.5f), dud(-0.2f, 0.2f);
  for (auto & v : hX) v = gk::f32_to_f16(dum(rng123));
  for (auto & v : hWt) v = gk::f32_to_f16(dud(rng456));
  for (auto & v : hB) v = gk::f32_to_f16(dud(rng456));

  cl_mem dX = rt.alloc((size_t)s.Cin * s.H * s.W * 2, CL_MEM_READ_ONLY);
  cl_mem dW = rt.alloc((size_t)s.Cout * s.Cin * 9 * 2, CL_MEM_READ_ONLY);
  cl_mem dB = rt.alloc((size_t)s.Cout * 2, CL_MEM_READ_ONLY);
  cl_mem dY = rt.alloc((size_t)s.Cout * Hout * Wout * 2, CL_MEM_WRITE_ONLY);
  rt.write(dX, (size_t)s.Cin * s.H * s.W * 2, hX.data());
  rt.write(dW, (size_t)s.Cout * s.Cin * 9 * 2, hWt.data());
  rt.write(dB, (size_t)s.Cout * 2, hB.data());
  clSetKernelArg(k, 0, sizeof(dX), &dX);
  clSetKernelArg(k, 1, sizeof(dW), &dW);
  clSetKernelArg(k, 2, sizeof(dB), &dB);
  clSetKernelArg(k, 3, sizeof(dY), &dY);
  int Cin = s.Cin, H = s.H, W = s.W, Cout = s.Cout, ho = Hout, wo = Wout;
  clSetKernelArg(k, 4, sizeof(Cin), &Cin);
  clSetKernelArg(k, 5, sizeof(H), &H);
  clSetKernelArg(k, 6, sizeof(W), &W);
  clSetKernelArg(k, 7, sizeof(Cout), &Cout);
  clSetKernelArg(k, 8, sizeof(ho), &ho);
  clSetKernelArg(k, 9, sizeof(wo), &wo);

  const size_t lws[3] = {static_cast<size_t>(c.TX / c.TM), static_cast<size_t>(c.TY), 1};
  const size_t gws[3] = {
    static_cast<size_t>((Wout + c.TX - 1) / c.TX) * lws[0],
    static_cast<size_t>((Hout + c.TY - 1) / c.TY) * lws[1],
    static_cast<size_t>((Cout + c.CB - 1) / c.CB)};

  const double med = rt.timeMs(
    [&] { return gk::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws); }, 3, iters);
  const double flops = 2.0 * Cout * Hout * Wout * Cin * 9;
  const double ops = rt.opsPerEuCycle(flops, med);
  std::printf(
    "  conv3x3 %-18s Cin=%-4d Cout=%-4d %dx%d s%d  %8.3f ms  %7.1f GFLOP/s  "
    "ops/EU/cyc=%5.2f (%5.1f%% of 16)",
    s.label.c_str(), Cin, Cout, H, W, c.STRIDE, med, flops / (med * 1e-3) / 1e9,
    ops, ops / 16 * 100);

  if (verify) {
    rt.read(dY, (size_t)Cout * Hout * Wout * 2, hY.data());
    double sumabs = 0, sumref = 0, maxabs = 0, refmax = 0;
    std::vector<float> ref((size_t)Cout * Hout * Wout);
    for (int oc = 0; oc < Cout; ++oc)
      for (int oy = 0; oy < Hout; ++oy)
        for (int ox = 0; ox < Wout; ++ox) {
          float acc = gk::f16_to_f32(hB[oc]);
          for (int ci = 0; ci < Cin; ++ci)
            for (int kh = 0; kh < 3; ++kh)
              for (int kw = 0; kw < 3; ++kw) {
                int yy = oy * c.STRIDE - c.PAD + kh, xx = ox * c.STRIDE - c.PAD + kw;
                if (yy >= 0 && yy < H && xx >= 0 && xx < W)
                  acc += gk::f16_to_f32(hX[((size_t)ci * H + yy) * W + xx]) *
                         gk::f16_to_f32(hWt[((size_t)oc * Cin + ci) * 9 + kh * 3 + kw]);
              }
          ref[((size_t)oc * Hout + oy) * Wout + ox] = acc;
        }
    for (size_t i = 0; i < ref.size(); ++i) {
      const double got = gk::f16_to_f32(hY[i]);
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

int benchBandwidth(gk::ClRuntime & rt, size_t mb, int iters)
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

  const double tc = rt.timeMs([&] { return gk::ClRuntime::enqueueND(rt.queue(), kcopy, 1, &gws, &lws); }, 3, iters);
  const double tr = rt.timeMs([&] { return gk::ClRuntime::enqueueND(rt.queue(), kread, 1, &gws, &lws); }, 3, iters);
  const double gb = static_cast<double>(n) * 4 / 1e9;
  std::printf("  copy : %.3f ms -> %.1f GB/s (read+write)\n", tc, 2 * gb / (tc * 1e-3));
  std::printf("  read : %.3f ms -> %.1f GB/s (read only)\n", tr, gb / (tr * 1e-3));
  clReleaseMemObject(in);
  clReleaseMemObject(out);
  clReleaseKernel(kcopy);
  clReleaseKernel(kread);
  return 0;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string op = "gemm", kernel_dir = INFVINO_KERNEL_DIR;
  gk::Tiles tiles;
  gk::Conv3x3Cfg conv;
  std::vector<ConvShape> conv_shapes;
  std::vector<Shape> shapes;
  int iters = 100;
  size_t mb = 256;
  bool verify = false;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--list-devices") {
      for (const auto & d : gk::ClRuntime::enumerate()) std::printf("  %s\n", d.describe().c_str());
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
      tiles = gk::parseTiles(next());
    } else if (a == "--conv") {
      conv = gk::parseConv(next());
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
    } else {
      std::fprintf(stderr, "unknown arg: %s\n", a.c_str());
      return 2;
    }
  }

  gk::ClRuntime rt(kernel_dir);
  std::printf("device     : %s\n", rt.info().describe().c_str());
  std::printf("peak FP16  : %.1f GFLOP/s (EU x clk x 16)\n\n", rt.info().peak_fp16_gflops);

  if (shapes.empty()) shapes = {{1024, 1024, 1024, "square"}, {512, 512, 512, "square"}};

  int rc = 0;
  if (op == "gemm") {
    std::printf("[gemm] tiles %s\n", tiles.label().c_str());
    for (const auto & s : shapes) rc |= benchGemm(rt, tiles, s, iters, verify);
  } else if (op == "conv1x1") {
    std::printf("[conv1x1] (== gemm: M=Cout, N=H*W, K=Cin) %s\n", tiles.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{64, 64, 80, 80, "c1x1"}, {256, 256, 20, 20, "c1x1"}};
    for (const auto & s : conv_shapes) {
      Shape g{s.Cout, s.H * s.W, s.Cin, "conv1x1"};
      rc |= benchGemm(rt, tiles, g, iters, verify);
    }
  } else if (op == "conv3x3") {
    std::printf("[conv3x3] %s\n", conv.label().c_str());
    if (conv_shapes.empty()) conv_shapes = {{64, 64, 80, 80, "p3-3x3"}, {64, 64, 40, 40, "p4-3x3"}};
    for (const auto & s : conv_shapes) rc |= benchConv(rt, conv, s, iters, verify);
  } else if (op == "bandwidth") {
    std::printf("[bandwidth] buffer=%zu MB\n", mb);
    rc = benchBandwidth(rt, mb, iters);
  } else {
    std::fprintf(stderr, "unknown op: %s\n", op.c_str());
    return 2;
  }
  return rc;
}
