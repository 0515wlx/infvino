// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_numtest —— 自研 kernel 数值检验：对确定性输入跑自定义算子并 dump 原始输出，
// 供 scripts/numerical_check_kernel.py 与 onnxruntime/numpy 参考逐元素比对。
//
//   kernel_numtest --op gemm --m 64 --n 64 --k 64
//       --input-a /work/a.bin --input-b /work/b.bin --dump /work/c.bin
//
// 约定：输入/输出均为 fp16 little-endian 裸数据（*.bin），row-major。
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "infvino/ClRuntime.hpp"
#include "infvino/Half.hpp"
#include "infvino/Tiles.hpp"

namespace
{
std::vector<uint16_t> readBin(const std::string & path, size_t expected)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  f.seekg(0, std::ios::end);
  const size_t bytes = static_cast<size_t>(f.tellg());
  f.seekg(0);
  if (expected && bytes != expected * 2)
    throw std::runtime_error("size mismatch for " + path + ": got " + std::to_string(bytes) +
      " expected " + std::to_string(expected * 2));
  std::vector<uint16_t> v(bytes / 2);
  f.read(reinterpret_cast<char *>(v.data()), static_cast<std::streamsize>(bytes));
  return v;
}

void writeBin(const std::string & path, const std::vector<uint16_t> & v)
{
  std::ofstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot write " + path);
  f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * 2));
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string op = "gemm", kernel_dir = INFVINO_KERNEL_DIR;
  std::string in_a, in_b, dump, in_x, in_w, in_bias;
  std::string in_scale;
  std::string knl, opts;
  int M = 0, N = 0, K = 0, iters = 1;
  int Cin = 0, Cout = 0, H = 0, W = 0, stride = -1, pad = -1;
  int B0 = 1, B1 = 1, outer = 0, axdim = 0, inner = 1;
  int dwK = 3, act = 0;
  infvino::Tiles tiles;
  infvino::Conv3x3Cfg conv;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--op") op = next();
    else if (a == "--kernel") knl = next();
    else if (a == "--opts") opts = next();
    else if (a == "--kernel-dir") kernel_dir = next();
    else if (a == "--m") M = std::atoi(next().c_str());
    else if (a == "--n") N = std::atoi(next().c_str());
    else if (a == "--k") K = std::atoi(next().c_str());
    else if (a == "--b0") B0 = std::atoi(next().c_str());
    else if (a == "--b1") B1 = std::atoi(next().c_str());
    else if (a == "--outer") outer = std::atoi(next().c_str());
    else if (a == "--axdim") axdim = std::atoi(next().c_str());
    else if (a == "--inner") inner = std::atoi(next().c_str());
    else if (a == "--dw-k") dwK = std::atoi(next().c_str());
    else if (a == "--act") act = std::atoi(next().c_str());
    else if (a == "--cin") Cin = std::atoi(next().c_str());
    else if (a == "--cout") Cout = std::atoi(next().c_str());
    else if (a == "--h") H = std::atoi(next().c_str());
    else if (a == "--w") W = std::atoi(next().c_str());
    else if (a == "--stride") stride = std::atoi(next().c_str());
    else if (a == "--pad") pad = std::atoi(next().c_str());
    else if (a == "--iters") iters = std::atoi(next().c_str());
    else if (a == "--input-a") in_a = next();
    else if (a == "--input-b") in_b = next();
    else if (a == "--input-x") in_x = next();
    else if (a == "--input-w") in_w = next();
    else if (a == "--input-bias") in_bias = next();
    else if (a == "--input-scale") in_scale = next();
    else if (a == "--dump") dump = next();
    else if (a == "--tiles") tiles = infvino::parseTiles(next());
    else if (a == "--conv") conv = infvino::parseConv(next());
    else if (a == "--ov") conv.OV = 1;
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  if (stride >= 0) conv.STRIDE = stride;
  if (pad >= 0) conv.PAD = pad;

  try {
    infvino::ClRuntime rt(kernel_dir);

    if (op == "gemm") {
      if (M <= 0 || N <= 0 || K <= 0) throw std::runtime_error("need --m --n --k");
      auto hA = readBin(in_a, static_cast<size_t>(M) * K);
      auto hB = readBin(in_b, static_cast<size_t>(K) * N);

      cl_kernel k = rt.buildKernel("gemm", "gemm_f16", tiles.options());
      cl_mem dA = rt.alloc(static_cast<size_t>(M) * K * 2, CL_MEM_READ_ONLY);
      cl_mem dB = rt.alloc(static_cast<size_t>(K) * N * 2, CL_MEM_READ_ONLY);
      cl_mem dC = rt.alloc(static_cast<size_t>(M) * N * 2, CL_MEM_WRITE_ONLY);
      rt.write(dA, static_cast<size_t>(M) * K * 2, hA.data());
      rt.write(dB, static_cast<size_t>(K) * N * 2, hB.data());
      clSetKernelArg(k, 0, sizeof(dA), &dA);
      clSetKernelArg(k, 1, sizeof(dB), &dB);
      clSetKernelArg(k, 2, sizeof(dC), &dC);
      clSetKernelArg(k, 3, sizeof(M), &M);
      clSetKernelArg(k, 4, sizeof(N), &N);
      clSetKernelArg(k, 5, sizeof(K), &K);
      const size_t lws[2] = {tiles.localX(), tiles.localY()};
      const size_t gws[2] = {
        static_cast<size_t>((N + tiles.BN - 1) / tiles.BN) * lws[0],
        static_cast<size_t>((M + tiles.BM - 1) / tiles.BM) * lws[1]};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hC(static_cast<size_t>(M) * N);
        rt.read(dC, static_cast<size_t>(M) * N * 2, hC.data());
        writeBin(dump, hC);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (%dx%d fp16)\n", dump.c_str(), M, N);
      }
      clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC); clReleaseKernel(k);
    } else if (op == "conv1x1") {
      // 1x1 conv == GEMM: C[Cout, HW] = W[Cout,Cin] * X[Cin, HW], with the fused
      // bias+activation epilogue used in production (EPI=1, bias=null here).
      if (Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::runtime_error("need --cin --cout --h --w");
      const int M = Cout, N = H * W, K = Cin;
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin);
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);

      tiles.EPI = 1;
      tiles.ACT = 0;
      cl_kernel k = rt.buildKernel("gemm", "gemm_f16", tiles.options());
      cl_mem dA = rt.alloc(static_cast<size_t>(M) * K * 2, CL_MEM_READ_ONLY);
      cl_mem dB = rt.alloc(static_cast<size_t>(K) * N * 2, CL_MEM_READ_ONLY);
      cl_mem dC = rt.alloc(static_cast<size_t>(M) * N * 2, CL_MEM_WRITE_ONLY);
      cl_mem dNull = nullptr;
      rt.write(dA, static_cast<size_t>(M) * K * 2, hW.data());
      rt.write(dB, static_cast<size_t>(K) * N * 2, hX.data());
      clSetKernelArg(k, 0, sizeof(dA), &dA);
      clSetKernelArg(k, 1, sizeof(dB), &dB);
      clSetKernelArg(k, 2, sizeof(dC), &dC);
      clSetKernelArg(k, 3, sizeof(M), &M);
      clSetKernelArg(k, 4, sizeof(N), &N);
      clSetKernelArg(k, 5, sizeof(K), &K);
      clSetKernelArg(k, 6, sizeof(dNull), &dNull);
      clSetKernelArg(k, 7, sizeof(dNull), &dNull);
      const size_t lws[2] = {tiles.localX(), tiles.localY()};
      const size_t gws[2] = {
        static_cast<size_t>((N + tiles.BN - 1) / tiles.BN) * lws[0],
        static_cast<size_t>((M + tiles.BM - 1) / tiles.BM) * lws[1]};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hC(static_cast<size_t>(M) * N);
        rt.read(dC, static_cast<size_t>(M) * N * 2, hC.data());
        writeBin(dump, hC);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (conv1x1 %dx%d fp16)\n",
          dump.c_str(), Cout, N);
      }
      clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC); clReleaseKernel(k);
    } else if (op == "conv1x1g") {
      // HW==1 split-K GEMV (production path for N=1 pointwise / fc).
      if (Cin <= 0 || Cout <= 0) throw std::runtime_error("need --cin --cout");
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin);
      auto hX = readBin(in_x, static_cast<size_t>(Cin));
      // R48 §4bis: 允许 --opts 覆盖（用于验证 GEMV_TM 多输出候选）。
      const std::string gopts =
          opts.empty() ? std::string("-DACT=0 -DRES=0 -DSG=16 -cl-mad-enable -cl-fast-relaxed-math")
                       : opts;
      cl_kernel k = rt.buildKernel("conv1x1", "conv1x1_gemv_f16", gopts);
      cl_mem dW = rt.alloc(static_cast<size_t>(Cout) * Cin * 2, CL_MEM_READ_ONLY);
      cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * 2, CL_MEM_READ_ONLY);
      cl_mem dC = rt.alloc(static_cast<size_t>(Cout) * 2, CL_MEM_WRITE_ONLY);
      cl_mem dNull = nullptr;
      rt.write(dW, static_cast<size_t>(Cout) * Cin * 2, hW.data());
      rt.write(dX, static_cast<size_t>(Cin) * 2, hX.data());
      clSetKernelArg(k, 0, sizeof(dW), &dW);
      clSetKernelArg(k, 1, sizeof(dX), &dX);
      clSetKernelArg(k, 2, sizeof(dNull), &dNull);
      clSetKernelArg(k, 3, sizeof(dNull), &dNull);
      clSetKernelArg(k, 4, sizeof(dC), &dC);
      clSetKernelArg(k, 5, sizeof(Cin), &Cin);
      clSetKernelArg(k, 6, sizeof(Cout), &Cout);
      const auto gt = gopts.find("-DGEMV_TM=");
      const int tm = gt == std::string::npos ? 1 : std::max(1, std::atoi(gopts.c_str() + gt + 10));
      const size_t lws[1] = {16};
      const size_t gws[1] = {static_cast<size_t>((Cout + tm - 1) / tm) * 16};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 1, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hC(static_cast<size_t>(Cout));
        rt.read(dC, static_cast<size_t>(Cout) * 2, hC.data());
        writeBin(dump, hC);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (conv1x1g %d fp16)\n", dump.c_str(), Cout);
      }
      clReleaseMemObject(dW); clReleaseMemObject(dX); clReleaseMemObject(dC); clReleaseKernel(k);
    } else if (op == "conv3x3") {
      if (Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::runtime_error("need --cin --cout --h --w");
      const int Hout = (H + 2 * conv.PAD - 3) / conv.STRIDE + 1;
      const int Wout = (W + 2 * conv.PAD - 3) / conv.STRIDE + 1;
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin * 9);
      std::vector<uint16_t> hB;
      if (!in_bias.empty()) hB = readBin(in_bias, static_cast<size_t>(Cout));

      if (conv.OV) {
        // OpenVINO os_iyx_osv32 port (kernels/conv_ov.cl), OSV-swizzled weights.
        const int obw = (conv.STRIDE == 2) ? 5 : 8;
        const int obh = (conv.STRIDE == 2) ? 4 : 2;
        char oo[192];
        std::snprintf(oo, sizeof(oo),
                      "-DOBW=%d -DOBH=%d -DSTRIDE=%d -DPAD=%d -DACT=%d -DRES=0 -DSG=16 "
                      "-cl-mad-enable -cl-fast-relaxed-math", obw, obh, conv.STRIDE, conv.PAD, conv.ACT);
        cl_kernel k = rt.buildKernel("conv_ov", "conv3x3_ov", oo);
        const int fmg = (Cout + 31) / 32;
        std::vector<uint16_t> hWo(static_cast<size_t>(fmg) * Cin * 9 * 32, 0);
        for (int g = 0; g < fmg; ++g)
          for (int ci = 0; ci < Cin; ++ci)
            for (int kk = 0; kk < 9; ++kk) {
              uint16_t * dst = &hWo[((static_cast<size_t>(g) * Cin + ci) * 9 + kk) * 32];
              for (int p = 0; p < 32; ++p) {
                const int oc = g * 32 + p;
                dst[p] = (oc < Cout) ? hW[(static_cast<size_t>(oc) * Cin + ci) * 9 + kk] : (uint16_t)0;
              }
            }
        cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * H * W * 2, CL_MEM_READ_ONLY);
        cl_mem dW = rt.alloc(hWo.size() * 2, CL_MEM_READ_ONLY);
        cl_mem dB = nullptr, dRes = nullptr;
        cl_mem dY = rt.alloc(static_cast<size_t>(Cout) * Hout * Wout * 2, CL_MEM_WRITE_ONLY);
        rt.write(dX, static_cast<size_t>(Cin) * H * W * 2, hX.data());
        rt.write(dW, hWo.size() * 2, hWo.data());
        if (!hB.empty()) { dB = rt.alloc(static_cast<size_t>(Cout) * 2, CL_MEM_READ_ONLY); rt.write(dB, static_cast<size_t>(Cout) * 2, hB.data()); }
        clSetKernelArg(k, 0, sizeof(dX), &dX);
        clSetKernelArg(k, 1, sizeof(dW), &dW);
        clSetKernelArg(k, 2, sizeof(dB), &dB);
        clSetKernelArg(k, 3, sizeof(dRes), &dRes);
        clSetKernelArg(k, 4, sizeof(dY), &dY);
        clSetKernelArg(k, 5, sizeof(Cin), &Cin);
        clSetKernelArg(k, 6, sizeof(H), &H);
        clSetKernelArg(k, 7, sizeof(W), &W);
        clSetKernelArg(k, 8, sizeof(Cout), &Cout);
        clSetKernelArg(k, 9, sizeof(Hout), &Hout);
        clSetKernelArg(k, 10, sizeof(Wout), &Wout);
        const size_t lws[3] = {1, 1, 16};
        const size_t gws[3] = {
          static_cast<size_t>((Wout + obw - 1) / obw),
          static_cast<size_t>((Hout + obh - 1) / obh),
          static_cast<size_t>((((Cout + 1) / 2) + 15) / 16) * 16};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws);
        rt.finish();
        if (!dump.empty()) {
          std::vector<uint16_t> hY(static_cast<size_t>(Cout) * Hout * Wout);
          rt.read(dY, static_cast<size_t>(Cout) * Hout * Wout * 2, hY.data());
          writeBin(dump, hY);
          std::fprintf(stderr, "[kernel_numtest] wrote %s (%dx%dx%d ov fp16)\n", dump.c_str(), Cout, Hout, Wout);
        }
        clReleaseMemObject(dX); clReleaseMemObject(dW); clReleaseMemObject(dY);
        if (dB) clReleaseMemObject(dB);
        clReleaseKernel(k);
        return 0;
      }

      cl_kernel k = rt.buildKernel("conv", "conv3x3_f16", conv.options());
    std::fprintf(stderr, "[kernel_numtest] conv options: %s\n", conv.options().c_str());
      cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * H * W * 2, CL_MEM_READ_ONLY);
      cl_mem dW = rt.alloc(static_cast<size_t>(Cout) * Cin * 9 * 2, CL_MEM_READ_ONLY);
      cl_mem dB = nullptr;
      cl_mem dY = rt.alloc(static_cast<size_t>(Cout) * Hout * Wout * 2, CL_MEM_WRITE_ONLY);
      rt.write(dX, static_cast<size_t>(Cin) * H * W * 2, hX.data());
      rt.write(dW, static_cast<size_t>(Cout) * Cin * 9 * 2, hW.data());
      if (!hB.empty()) {
        dB = rt.alloc(static_cast<size_t>(Cout) * 2, CL_MEM_READ_ONLY);
        rt.write(dB, static_cast<size_t>(Cout) * 2, hB.data());
      }
      clSetKernelArg(k, 0, sizeof(dX), &dX);
      clSetKernelArg(k, 1, sizeof(dW), &dW);
      clSetKernelArg(k, 2, sizeof(dB), &dB);
      clSetKernelArg(k, 3, sizeof(dY), &dY);
      clSetKernelArg(k, 4, sizeof(Cin), &Cin);
      clSetKernelArg(k, 5, sizeof(H), &H);
      clSetKernelArg(k, 6, sizeof(W), &W);
      clSetKernelArg(k, 7, sizeof(Cout), &Cout);
      clSetKernelArg(k, 8, sizeof(Hout), &Hout);
      clSetKernelArg(k, 9, sizeof(Wout), &Wout);
      const size_t lws[3] = {static_cast<size_t>(conv.TX / conv.TM),
                             static_cast<size_t>(conv.TY), 1};
      const size_t gws[3] = {
        static_cast<size_t>((Wout + conv.TX - 1) / conv.TX) * lws[0],
        static_cast<size_t>((Hout + conv.TY - 1) / conv.TY) * lws[1],
        static_cast<size_t>((Cout + conv.CB - 1) / conv.CB)};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(static_cast<size_t>(Cout) * Hout * Wout);
        rt.read(dY, static_cast<size_t>(Cout) * Hout * Wout * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (%dx%dx%d fp16)\n",
          dump.c_str(), Cout, Hout, Wout);
      }
      clReleaseMemObject(dX); clReleaseMemObject(dW); clReleaseMemObject(dY);
      if (dB) clReleaseMemObject(dB);
      clReleaseKernel(k);
    } else if (op == "bmm") {
      if (M <= 0 || N <= 0 || K <= 0) throw std::runtime_error("need --m --n --k");
      const std::string kern = knl.empty() ? "bmm" : knl;
      auto hA = readBin(in_a, static_cast<size_t>(B0) * B1 * M * K);
      auto hB = readBin(in_b, static_cast<size_t>(B0) * B1 * K * N);
      cl_kernel k = rt.buildKernel("ops", kern, opts);
      cl_mem dA = rt.alloc(static_cast<size_t>(B0) * B1 * M * K * 2, CL_MEM_READ_ONLY);
      cl_mem dB = rt.alloc(static_cast<size_t>(B0) * B1 * K * N * 2, CL_MEM_READ_ONLY);
      cl_mem dY = rt.alloc(static_cast<size_t>(B0) * B1 * M * N * 2, CL_MEM_WRITE_ONLY);
      rt.write(dA, static_cast<size_t>(B0) * B1 * M * K * 2, hA.data());
      rt.write(dB, static_cast<size_t>(B0) * B1 * K * N * 2, hB.data());
      clSetKernelArg(k, 0, sizeof(dA), &dA);
      clSetKernelArg(k, 1, sizeof(dB), &dB);
      clSetKernelArg(k, 2, sizeof(dY), &dY);
      clSetKernelArg(k, 3, sizeof(B0), &B0);
      clSetKernelArg(k, 4, sizeof(B1), &B1);
      clSetKernelArg(k, 5, sizeof(M), &M);
      clSetKernelArg(k, 6, sizeof(K), &K);
      clSetKernelArg(k, 7, sizeof(N), &N);
      auto optInt = [&](const char * key, int def) {
        const auto p = opts.find(key);
        return p == std::string::npos ? def : std::atoi(opts.c_str() + p + std::strlen(key));
      };
      if (kern == "bmm2") {
        const size_t g[3] = {(size_t)N, (size_t)M, (size_t)B0 * B1};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 3, g, nullptr);
      } else if (kern == "bmm_t") {
        const int TM = optInt("-DBMM_TM=", 4), TN = optInt("-DBMM_TN=", 8);
        const size_t g[3] = {(size_t)((N + TN - 1) / TN), (size_t)((M + TM - 1) / TM),
                             (size_t)B0 * B1};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 3, g, nullptr);
      } else {
        const size_t g[1] = {(size_t)B0 * B1 * M * N};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 1, g, nullptr);
      }
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(static_cast<size_t>(B0) * B1 * M * N);
        rt.read(dY, static_cast<size_t>(B0) * B1 * M * N * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (bmm %dx%dx%d B%d.%d fp16)\n",
                     dump.c_str(), M, N, K, B0, B1);
      }
      clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dY); clReleaseKernel(k);
    } else if (op == "softmax") {
      if (outer <= 0 || axdim <= 0 || inner <= 0)
        throw std::runtime_error("need --outer --axdim --inner");
      const std::string kern = knl.empty() ? "softmax_axis" : knl;
      auto hX = readBin(in_x, static_cast<size_t>(outer) * axdim * inner);
      cl_kernel k = rt.buildKernel("ops", kern, opts);
      cl_mem dX = rt.alloc(hX.size() * 2, CL_MEM_READ_ONLY);
      cl_mem dY = rt.alloc(hX.size() * 2, CL_MEM_WRITE_ONLY);
      rt.write(dX, hX.size() * 2, hX.data());
      clSetKernelArg(k, 0, sizeof(dX), &dX);
      clSetKernelArg(k, 1, sizeof(dY), &dY);
      clSetKernelArg(k, 2, sizeof(outer), &outer);
      clSetKernelArg(k, 3, sizeof(axdim), &axdim);
      clSetKernelArg(k, 4, sizeof(inner), &inner);
      if (kern == "softmax_axis_r") {
        const auto p = opts.find("-DSM_WGS=");
        const int wgs = p == std::string::npos ? 128
                                               : std::atoi(opts.c_str() + p + std::strlen("-DSM_WGS="));
        const size_t lws[2] = {(size_t)wgs, 1};
        const size_t gws[2] = {(size_t)outer * wgs, (size_t)inner};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws);
      } else {
        const size_t g[1] = {(size_t)outer * inner};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 1, g, nullptr);
      }
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(hX.size());
        rt.read(dY, hX.size() * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (softmax %dx%dx%d fp16)\n",
                     dump.c_str(), outer, axdim, inner);
      }
      clReleaseMemObject(dX); clReleaseMemObject(dY); clReleaseKernel(k);
    } else if (op == "gap") {
      // R51 D4+ unit test: NCHW [C,H,W] -> reorder to b_fs_yx_fsv16 -> gap_r -DGAP_IN_FSV16=1,
      // dump [C]. Python reference = mean over H,W (NCHW).
      if (Cin <= 0 || H <= 0 || W <= 0) throw std::runtime_error("need --cin --h --w");
      const int HW = H * W;
      const int Cpad = (Cin + 15) / 16 * 16;
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);

      cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * H * W * 2, CL_MEM_READ_ONLY);
      cl_mem dF = rt.alloc(static_cast<size_t>(Cpad) * H * W * 2, CL_MEM_READ_WRITE);
      rt.write(dX, static_cast<size_t>(Cin) * H * W * 2, hX.data());
      std::vector<uint16_t> zeros(static_cast<size_t>(Cpad) * H * W, 0);
      rt.write(dF, zeros.size() * 2, zeros.data());
      cl_kernel kr = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
      clSetKernelArg(kr, 0, sizeof(dX), &dX);
      clSetKernelArg(kr, 1, sizeof(dF), &dF);
      clSetKernelArg(kr, 2, sizeof(Cin), &Cin);
      clSetKernelArg(kr, 3, sizeof(H), &H);
      clSetKernelArg(kr, 4, sizeof(W), &W);
      const size_t rg[3] = {(size_t)W, (size_t)H, (size_t)Cin};
      infvino::ClRuntime::enqueueND(rt.queue(), kr, 3, rg, nullptr);

      const std::string kern = knl.empty() ? "gap_r" : knl;
      const std::string o = opts.empty() ? "-DGAP_WGS=128 -DGAP_IN_FSV16=1" : opts;
      cl_kernel k = rt.buildKernel("ops", kern, o);
      cl_mem dY = rt.alloc(static_cast<size_t>(Cin) * 2, CL_MEM_WRITE_ONLY);
      clSetKernelArg(k, 0, sizeof(dF), &dF);
      clSetKernelArg(k, 1, sizeof(dY), &dY);
      clSetKernelArg(k, 2, sizeof(Cin), &Cin);
      clSetKernelArg(k, 3, sizeof(HW), &HW);
      const size_t lws[1] = {128};
      const size_t gws[1] = {static_cast<size_t>(Cin) * 128};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 1, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(static_cast<size_t>(Cin));
        rt.read(dY, static_cast<size_t>(Cin) * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (gap_fsv16 C=%d fp16)\n", dump.c_str(), Cin);
      }
      clReleaseMemObject(dX); clReleaseMemObject(dF); clReleaseMemObject(dY);
      clReleaseKernel(kr); clReleaseKernel(k);
    } else if (op == "conv1x1blk") {
      // R51 D5 unit test: conv1x1_blk with -DMUL_SCALE=1, fsv16 input.
      //   y[o,h,w] = sum_c W[o,c] * ( X[c,h,w] * scale[c] ) + bias[o]
      if (Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::runtime_error("need --cin --cout --h --w");
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin);
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);
      auto hS = readBin(in_scale, static_cast<size_t>(Cin));
      std::vector<uint16_t> hB;
      if (!in_bias.empty()) hB = readBin(in_bias, static_cast<size_t>(Cout));
      const int icb = (Cin + 15) / 16, ocb = (Cout + 15) / 16;
      // os_is_yx_isv16_osv16 repack.
      std::vector<uint16_t> hWo(static_cast<size_t>(ocb) * icb * 16 * 16, 0);
      for (int o = 0; o < Cout; ++o)
        for (int i = 0; i < Cin; ++i)
          hWo[((((size_t)(o / 16) * icb + (i / 16)) * 16) + (i % 16)) * 16 + (o % 16)] =
              hW[static_cast<size_t>(o) * Cin + i];
      // reorder X -> fsv16.
      const int Cpad = (Cin + 15) / 16 * 16;
      cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * H * W * 2, CL_MEM_READ_ONLY);
      cl_mem dF = rt.alloc(static_cast<size_t>(Cpad) * H * W * 2, CL_MEM_READ_WRITE);
      rt.write(dX, static_cast<size_t>(Cin) * H * W * 2, hX.data());
      std::vector<uint16_t> zeros(static_cast<size_t>(Cpad) * H * W, 0);
      rt.write(dF, zeros.size() * 2, zeros.data());
      cl_kernel kr = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
      clSetKernelArg(kr, 0, sizeof(dX), &dX);
      clSetKernelArg(kr, 1, sizeof(dF), &dF);
      clSetKernelArg(kr, 2, sizeof(Cin), &Cin);
      clSetKernelArg(kr, 3, sizeof(H), &H);
      clSetKernelArg(kr, 4, sizeof(W), &W);
      const size_t rg[3] = {(size_t)W, (size_t)H, (size_t)Cin};
      infvino::ClRuntime::enqueueND(rt.queue(), kr, 3, rg, nullptr);
      clReleaseKernel(kr);
      // weights / bias / scale.
      cl_mem dWo = rt.alloc(hWo.size() * 2, CL_MEM_READ_ONLY);
      rt.write(dWo, hWo.size() * 2, hWo.data());
      cl_mem dS = rt.alloc(static_cast<size_t>(Cin) * 2, CL_MEM_READ_ONLY);
      rt.write(dS, static_cast<size_t>(Cin) * 2, hS.data());
      cl_mem dB = nullptr;
      if (!hB.empty()) { dB = rt.alloc(static_cast<size_t>(Cout) * 2, CL_MEM_READ_ONLY);
                         rt.write(dB, static_cast<size_t>(Cout) * 2, hB.data()); }
      cl_mem dRes = nullptr;
      cl_mem dY = rt.alloc(static_cast<size_t>(Cout) * H * W * 2, CL_MEM_WRITE_ONLY);
      const int XB = 4, YB = 1, SLM = 4;
      const std::string o = "-DX_BLOCK=4 -DY_BLOCK=1 -DSLM_DIV=4 -DACT=0 -DMUL_SCALE=1 -DSG=16 "
                            "-cl-mad-enable -cl-fast-relaxed-math";
      cl_kernel k = rt.buildKernel("conv1x1_blk", "conv1x1_blk", o);
      clSetKernelArg(k, 0, sizeof(dF), &dF);
      clSetKernelArg(k, 1, sizeof(dWo), &dWo);
      clSetKernelArg(k, 2, sizeof(dB), &dB);
      clSetKernelArg(k, 3, sizeof(dY), &dY);
      clSetKernelArg(k, 4, sizeof(dRes), &dRes);
      clSetKernelArg(k, 5, sizeof(Cin), &Cin);
      clSetKernelArg(k, 6, sizeof(H), &H);
      clSetKernelArg(k, 7, sizeof(W), &W);
      clSetKernelArg(k, 8, sizeof(Cout), &Cout);
      clSetKernelArg(k, 9, sizeof(dS), &dS);
      const size_t lws[3] = {1, static_cast<size_t>(16 * SLM), 1};
      const size_t ybCount = static_cast<size_t>((H + YB - 1) / YB);
      const size_t gws[3] = {static_cast<size_t>(((W + XB - 1) / XB)) * ybCount,
                             static_cast<size_t>(((Cout + 15) / 16) * lws[1]), 1};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(static_cast<size_t>(Cout) * H * W);
        rt.read(dY, hY.size() * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (conv1x1blk %dx%dx%d fp16)\n",
                     dump.c_str(), Cout, H, W);
      }
      clReleaseMemObject(dX); clReleaseMemObject(dF); clReleaseMemObject(dWo);
      clReleaseMemObject(dS); clReleaseMemObject(dY);
      if (dB) clReleaseMemObject(dB);
      clReleaseKernel(k);
    } else if (op == "depthwiseblk") {
      // R51 D4+ unit test: depthwise_blk with OUT_FSV16=1 (fsv16 in/out).
      if (Cin <= 0 || H <= 0 || W <= 0) throw std::runtime_error("need --cin --h --w");
      const int S = stride >= 0 ? stride : 1, P = pad >= 0 ? pad : 1;
      const int Ho = (H + 2 * P - dwK) / S + 1, Wo = (W + 2 * P - dwK) / S + 1;
      const int Cpad = (Cin + 15) / 16 * 16;
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);
      auto hW = readBin(in_w, static_cast<size_t>(Cin) * dwK * dwK);
      std::vector<uint16_t> hB;
      if (!in_bias.empty()) hB = readBin(in_bias, static_cast<size_t>(Cin));
      // [C/16][K][K][16] weight repack.
      std::vector<uint16_t> hWo(static_cast<size_t>((Cin + 15) / 16) * dwK * dwK * 16, 0);
      for (int c = 0; c < Cin; ++c)
        for (int kk = 0; kk < dwK * dwK; ++kk)
          hWo[((static_cast<size_t>(c / 16) * dwK * dwK) + kk) * 16 + (c % 16)] =
              hW[static_cast<size_t>(c) * dwK * dwK + kk];
      // reorder X -> fsv16.
      cl_mem dX = rt.alloc(static_cast<size_t>(Cin) * H * W * 2, CL_MEM_READ_ONLY);
      cl_mem dF = rt.alloc(static_cast<size_t>(Cpad) * H * W * 2, CL_MEM_READ_WRITE);
      rt.write(dX, static_cast<size_t>(Cin) * H * W * 2, hX.data());
      std::vector<uint16_t> zeros(static_cast<size_t>(Cpad) * H * W, 0);
      rt.write(dF, zeros.size() * 2, zeros.data());
      cl_kernel kr = rt.buildKernel("conv_blk", "reorder_bfyx_to_fsv16", "");
      clSetKernelArg(kr, 0, sizeof(dX), &dX);
      clSetKernelArg(kr, 1, sizeof(dF), &dF);
      clSetKernelArg(kr, 2, sizeof(Cin), &Cin);
      clSetKernelArg(kr, 3, sizeof(H), &H);
      clSetKernelArg(kr, 4, sizeof(W), &W);
      const size_t rg[3] = {(size_t)W, (size_t)H, (size_t)Cin};
      infvino::ClRuntime::enqueueND(rt.queue(), kr, 3, rg, nullptr);
      clReleaseKernel(kr);
      cl_mem dWo = rt.alloc(hWo.size() * 2, CL_MEM_READ_ONLY);
      rt.write(dWo, hWo.size() * 2, hWo.data());
      cl_mem dB = nullptr;
      if (!hB.empty()) { dB = rt.alloc(static_cast<size_t>(Cin) * 2, CL_MEM_READ_ONLY);
                         rt.write(dB, static_cast<size_t>(Cin) * 2, hB.data()); }
      cl_mem dY = rt.alloc(static_cast<size_t>(Cpad) * Ho * Wo * 2, CL_MEM_WRITE_ONLY);
      char ob[320];
      if (!opts.empty()) {
        std::snprintf(ob, sizeof(ob), "%s", opts.c_str());
      } else {
        std::snprintf(ob, sizeof(ob),
                      "-DX_BLOCK=4 -DY_BLOCK=1 -DDWK=%d -DSTRIDE=%d -DPAD=%d -DACT=%d -DSG=16 "
                      "-DOUT_FSV16=1 -cl-mad-enable -cl-fast-relaxed-math", dwK, S, P, act);
      }
      cl_kernel k = rt.buildKernel("depthwise_blk", "depthwise_blk", ob);
      clSetKernelArg(k, 0, sizeof(dF), &dF);
      clSetKernelArg(k, 1, sizeof(dWo), &dWo);
      clSetKernelArg(k, 2, sizeof(dB), &dB);
      clSetKernelArg(k, 3, sizeof(dY), &dY);
      clSetKernelArg(k, 4, sizeof(Cin), &Cin);
      clSetKernelArg(k, 5, sizeof(H), &H);
      clSetKernelArg(k, 6, sizeof(W), &W);
      clSetKernelArg(k, 7, sizeof(Ho), &Ho);
      clSetKernelArg(k, 8, sizeof(Wo), &Wo);
      const size_t lws[3] = {1, 16, 1};
      const int XB = 4;
      const size_t gws[3] = {static_cast<size_t>(((Wo + XB - 1) / XB) * ((Ho + 1 - 1) / 1)),
                             static_cast<size_t>(((Cin + 15) / 16) * 16), 1};
      for (int i = 0; i < iters; ++i)
        infvino::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hYf(static_cast<size_t>(Cpad) * Ho * Wo, 0);
        rt.read(dY, hYf.size() * 2, hYf.data());
        // de-fsv16 -> NCHW.
        std::vector<uint16_t> hY(static_cast<size_t>(Cin) * Ho * Wo);
        for (int p = 0; p < Cpad * Ho * Wo; ++p) {
          const int l = p % 16, q = p / 16, x = q % Wo, q2 = q / Wo, y = q2 % Ho, cb = q2 / Ho;
          const int c = cb * 16 + l;
          if (c < Cin) hY[(static_cast<size_t>(c) * Ho + y) * Wo + x] = hYf[p];
        }
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (depthwiseblk %dx%dx%d fp16)\n",
                     dump.c_str(), Cin, Ho, Wo);
      }
      clReleaseMemObject(dX); clReleaseMemObject(dF); clReleaseMemObject(dWo);
      clReleaseMemObject(dY);
      if (dB) clReleaseMemObject(dB);
      clReleaseKernel(k);
    } else if (op == "depthwise") {
      if (Cin <= 0 || H <= 0 || W <= 0) throw std::runtime_error("need --cin --h --w");
      int S = stride >= 0 ? stride : 1, P = pad >= 0 ? pad : 1;
      int Ho = (H + 2 * P - dwK) / S + 1, Wo = (W + 2 * P - dwK) / S + 1;
      const std::string kern = knl.empty() ? "depthwise_f16" : knl;
      char base[192];
      std::snprintf(base, sizeof(base),
                    "-DDW_K=%d -DDW_S=%d -DDW_P=%d -DDW_ACT=%d", dwK, S, P, act);
      std::string o = base;
      if (!opts.empty()) o += " " + opts;
      o += " -cl-mad-enable -cl-fast-relaxed-math";
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);
      auto hW = readBin(in_w, static_cast<size_t>(Cin) * dwK * dwK);
      std::vector<uint16_t> hB;
      if (!in_bias.empty()) hB = readBin(in_bias, static_cast<size_t>(Cin));
      const bool isVp = (kern == "depthwise_vp");
      cl_kernel k = rt.buildKernel("conv_general", kern, o);
      cl_mem dX = rt.alloc(hX.size() * 2, CL_MEM_READ_ONLY);
      cl_mem dW = rt.alloc(hW.size() * 2, CL_MEM_READ_ONLY);
      cl_mem dB = nullptr;
      cl_mem dY = rt.alloc(static_cast<size_t>(Cin) * Ho * Wo * 2, CL_MEM_WRITE_ONLY);
      rt.write(dX, hX.size() * 2, hX.data());
      rt.write(dW, hW.size() * 2, hW.data());
      if (!hB.empty()) {
        dB = rt.alloc(hB.size() * 2, CL_MEM_READ_ONLY);
        rt.write(dB, hB.size() * 2, hB.data());
      }
      // R31: depthwise_vp reads a zero-padded input produced by depthwise_pad.
      cl_mem dXin = dX;
      int    Hs = H, Ws = W;
      if (isVp) {
        const auto p = o.find("-DDW_TW=");
        const int tw = p == std::string::npos
                         ? 8
                         : std::atoi(o.c_str() + p + std::strlen("-DDW_TW="));
        const int groups = (Wo + tw - 1) / tw;
        const int strlen = (tw - 1) * S + dwK;  // == DW_STRLEN
        int wpad = std::max((Wo - 1) * S + dwK, (groups - 1) * tw * S + strlen);
        if (wpad < P + W) wpad = P + W;
        Hs = H + 2 * P;
        Ws = wpad;
        const size_t nbytes = static_cast<size_t>(Cin) * Hs * wpad * 2;
        dXin = rt.alloc(nbytes, CL_MEM_READ_WRITE);
        std::vector<char> zeros(nbytes, 0);
        rt.write(dXin, nbytes, zeros.data());
        cl_kernel kpad = rt.buildKernel("conv_general", "depthwise_pad", "");
        clSetKernelArg(kpad, 0, sizeof(dX), &dX);
        clSetKernelArg(kpad, 1, sizeof(dXin), &dXin);
        clSetKernelArg(kpad, 2, sizeof(Cin), &Cin);
        clSetKernelArg(kpad, 3, sizeof(H), &H);
        clSetKernelArg(kpad, 4, sizeof(W), &W);
        clSetKernelArg(kpad, 5, sizeof(Hs), &Hs);
        clSetKernelArg(kpad, 6, sizeof(wpad), &wpad);
        clSetKernelArg(kpad, 7, sizeof(P), &P);
        const size_t gp[3] = {(size_t)W, (size_t)H, (size_t)Cin};
        infvino::ClRuntime::enqueueND(rt.queue(), kpad, 3, gp, nullptr);
        clReleaseKernel(kpad);
      }
      clSetKernelArg(k, 0, sizeof(dXin), &dXin);
      clSetKernelArg(k, 1, sizeof(dW), &dW);
      clSetKernelArg(k, 2, sizeof(dB), &dB);
      clSetKernelArg(k, 3, sizeof(dY), &dY);
      clSetKernelArg(k, 4, sizeof(Cin), &Cin);
      clSetKernelArg(k, 5, sizeof(Hs), &Hs);
      clSetKernelArg(k, 6, sizeof(Ws), &Ws);
      clSetKernelArg(k, 7, sizeof(Ho), &Ho);
      clSetKernelArg(k, 8, sizeof(Wo), &Wo);
      if (kern == "depthwise_v" || isVp) {
        const auto p = o.find("-DDW_TW=");
        const int tw = p == std::string::npos
                         ? 4
                         : std::atoi(o.c_str() + p + std::strlen("-DDW_TW="));
        const size_t g[3] = {static_cast<size_t>((Wo + tw - 1) / tw),
                             static_cast<size_t>(Ho), static_cast<size_t>(Cin)};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 3, g, nullptr);
      } else {
        const size_t g[1] = {static_cast<size_t>(Cin) * Ho * Wo};
        for (int i = 0; i < iters; ++i) infvino::ClRuntime::enqueueND(rt.queue(), k, 1, g, nullptr);
      }
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hY(static_cast<size_t>(Cin) * Ho * Wo);
        rt.read(dY, hY.size() * 2, hY.data());
        writeBin(dump, hY);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (depthwise %dx%dx%d fp16)\n",
                     dump.c_str(), Cin, Ho, Wo);
      }
      if (isVp) clReleaseMemObject(dXin);
      clReleaseMemObject(dX); clReleaseMemObject(dW); clReleaseMemObject(dY);
      if (dB) clReleaseMemObject(dB);
      clReleaseKernel(k);
    } else {
      throw std::runtime_error("unsupported op: " + op);
    }
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[kernel_numtest] error: %s\n", e.what());
    return 1;
  }
  return 0;
}
