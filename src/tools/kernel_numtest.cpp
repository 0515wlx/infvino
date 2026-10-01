// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_numtest —— 自研 kernel 数值检验：对确定性输入跑自定义算子并 dump 原始输出，
// 供 scripts/numerical_check_kernel.py 与 onnxruntime/numpy 参考逐元素比对。
//
//   kernel_numtest --op gemm --m 64 --n 64 --k 64
//       --input-a /work/a.bin --input-b /work/b.bin --dump /work/c.bin
//
// 约定：输入/输出均为 fp16 little-endian 裸数据（*.bin），row-major。
#include <cstdint>
#include <cstdio>
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
  int M = 0, N = 0, K = 0, iters = 1;
  int Cin = 0, Cout = 0, H = 0, W = 0, stride = -1, pad = -1;
  gk::Tiles tiles;
  gk::Conv3x3Cfg conv;

  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--op") op = next();
    else if (a == "--kernel-dir") kernel_dir = next();
    else if (a == "--m") M = std::atoi(next().c_str());
    else if (a == "--n") N = std::atoi(next().c_str());
    else if (a == "--k") K = std::atoi(next().c_str());
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
    else if (a == "--dump") dump = next();
    else if (a == "--tiles") tiles = gk::parseTiles(next());
    else if (a == "--conv") conv = gk::parseConv(next());
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  if (stride >= 0) conv.STRIDE = stride;
  if (pad >= 0) conv.PAD = pad;

  try {
    gk::ClRuntime rt(kernel_dir);

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
        gk::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hC(static_cast<size_t>(M) * N);
        rt.read(dC, static_cast<size_t>(M) * N * 2, hC.data());
        writeBin(dump, hC);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (%dx%d fp16)\n", dump.c_str(), M, N);
      }
      clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC); clReleaseKernel(k);
    } else if (op == "conv1x1") {
      // 1x1 conv == GEMM: C[Cout, HW] = W[Cout,Cin] * X[Cin, HW].
      if (Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::runtime_error("need --cin --cout --h --w");
      const int M = Cout, N = H * W, K = Cin;
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin);
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);

      cl_kernel k = rt.buildKernel("gemm", "gemm_f16", tiles.options());
      cl_mem dA = rt.alloc(static_cast<size_t>(M) * K * 2, CL_MEM_READ_ONLY);
      cl_mem dB = rt.alloc(static_cast<size_t>(K) * N * 2, CL_MEM_READ_ONLY);
      cl_mem dC = rt.alloc(static_cast<size_t>(M) * N * 2, CL_MEM_WRITE_ONLY);
      rt.write(dA, static_cast<size_t>(M) * K * 2, hW.data());
      rt.write(dB, static_cast<size_t>(K) * N * 2, hX.data());
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
        gk::ClRuntime::enqueueND(rt.queue(), k, 2, gws, lws);
      rt.finish();
      if (!dump.empty()) {
        std::vector<uint16_t> hC(static_cast<size_t>(M) * N);
        rt.read(dC, static_cast<size_t>(M) * N * 2, hC.data());
        writeBin(dump, hC);
        std::fprintf(stderr, "[kernel_numtest] wrote %s (conv1x1 %dx%d fp16)\n",
          dump.c_str(), Cout, N);
      }
      clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dC); clReleaseKernel(k);
    } else if (op == "conv3x3") {
      if (Cin <= 0 || Cout <= 0 || H <= 0 || W <= 0)
        throw std::runtime_error("need --cin --cout --h --w");
      const int Hout = (H + 2 * conv.PAD - 3) / conv.STRIDE + 1;
      const int Wout = (W + 2 * conv.PAD - 3) / conv.STRIDE + 1;
      auto hX = readBin(in_x, static_cast<size_t>(Cin) * H * W);
      auto hW = readBin(in_w, static_cast<size_t>(Cout) * Cin * 9);
      std::vector<uint16_t> hB;
      if (!in_bias.empty()) hB = readBin(in_bias, static_cast<size_t>(Cout));

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
        gk::ClRuntime::enqueueND(rt.queue(), k, 3, gws, lws);
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
    } else {
      throw std::runtime_error("unsupported op: " + op);
    }
  } catch (const std::exception & e) {
    std::fprintf(stderr, "[kernel_numtest] error: %s\n", e.what());
    return 1;
  }
  return 0;
}
