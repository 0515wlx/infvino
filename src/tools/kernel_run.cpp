// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_run —— 执行一份“算子计划”（由 scripts/onnx2plan.py 生成）的开发/调试 CLI。
// 库内生产路径由 gk::PlanModel 承担；本工具额外支持 --report / --iters / 文件 dump。
//
//   kernel_run --plan plan.txt [--input in.bin] [--output out.bin]
//              [--out-dir DIR] [--iters N] [--report] [--kernel-dir DIR]
//
// 计划格式见 include/infvino/PlanModel.hpp。
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "infvino/PlanModel.hpp"

namespace
{
bool readFile(const std::string & path, std::vector<uint16_t> & buf)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::streamsize bytes = f.tellg();
  f.seekg(0);
  buf.resize(static_cast<size_t>(bytes) / 2);
  f.read(reinterpret_cast<char *>(buf.data()), bytes);
  return true;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string plan_path, kernel_dir, input_path, output_path, out_dir, dump_tensor, input2_path;
  int         iters  = 1;
  bool        report = false;

  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() { return std::string(i + 1 < argc ? argv[++i] : ""); };
    if (a == "--plan") plan_path = next();
    else if (a == "--kernel-dir") kernel_dir = next();
    else if (a == "--input") input_path = next();
    else if (a == "--input2") input2_path = next();
    else if (a == "--output") output_path = next();
    else if (a == "--out-dir") out_dir = next();
    else if (a == "--iters") iters = std::atoi(next().c_str());
    else if (a == "--report") report = true;
    else if (a == "--dump-tensor") dump_tensor = next();
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  if (plan_path.empty()) { std::fprintf(stderr, "usage: kernel_run --plan <plan.txt>\n"); return 2; }
  if (iters < 1) iters = 1;

  try
  {
    const std::string kdir = kernel_dir.empty() ? INFVINO_KERNEL_DIR : kernel_dir;
    gk::PlanModel     model(plan_path, kdir, -1, 0, report);

    std::vector<uint16_t> in(static_cast<size_t>(model.inputNumel()), 0);
    if (!input_path.empty() && !readFile(input_path, in))
    {
      std::fprintf(stderr, "[kernel_run] cannot open input %s\n", input_path.c_str());
      return 1;
    }
    else if (!input_path.empty() &&
             in.size() != static_cast<size_t>(model.inputNumel()))
    {
      std::fprintf(
        stderr, "[kernel_run] input size mismatch: got %zu elems, expected %zu\n",
        in.size(), static_cast<size_t>(model.inputNumel()));
      return 1;
    }
    model.setInput(in.data());

    model.run();  // warmup（构建/预热 cache，不计入统计）
    model.clearProfile();
    for (int it = 0; it < iters; ++it) model.run();

    // --input2：再喂一份**不同**的输入并前向一次，输出/*dump 都取这一帧。
    // 这样任何「把每帧激活错误缓存成只算一次」的 bug 都会在输出里暴露
    // （单输入重复跑永远发现不了；见 scripts/reuse_check.py）。
    if (!input2_path.empty())
    {
      std::vector<uint16_t> in2(static_cast<size_t>(model.inputNumel()));
      if (!readFile(input2_path, in2) ||
          in2.size() != static_cast<size_t>(model.inputNumel()))
      {
        std::fprintf(stderr, "[kernel_run] cannot read/!size --input2 %s\n", input2_path.c_str());
        return 1;
      }
      model.setInput(in2.data());
      model.run();   // frame 2（不计入统计；输出取该帧）
      std::fprintf(stderr, "[kernel_run] frame2 input=%s\n", input2_path.c_str());
    }

    for (size_t i = 0; i < model.outputCount(); ++i)
    {
      std::vector<uint16_t> buf(static_cast<size_t>(model.outputNumel(i)));
      model.readOutput(i, buf.data());
      std::string path;
      if (!out_dir.empty())
        path = out_dir + "/out" + std::to_string(i) + ".bin";
      else if (i == 0 && !output_path.empty())
        path = output_path;
      if (path.empty()) continue;
      std::ofstream f(path, std::ios::binary);
      f.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
      std::fprintf(stderr, "[kernel_run] wrote %s\n", path.c_str());
    }

    if (!dump_tensor.empty())
    {
      // P0 诊断：把任意中间张量写到 <dump_tensor>.bin（fp16，行主序）。
      const size_t n = model.tensorNumel(dump_tensor);
      if (n == 0) std::fprintf(stderr, "[kernel_run] no tensor: %s\n", dump_tensor.c_str());
      else
      {
        std::vector<uint16_t> buf(n);
        model.readTensor(dump_tensor, buf.data());
        std::string safe = dump_tensor;
        for (char & c : safe) if (c == '/') c = '_';
        const std::string path = safe + ".bin";
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char *>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
        std::fprintf(stderr, "[kernel_run] wrote %s (%zu elems)\n", path.c_str(), n);
      }
    }

    if (report)
    {
      double total = 0;
      for (auto & kv : model.opProfile()) total += kv.second.first;
      std::printf("device: %s\n", model.device().describe().c_str());
      const size_t req = model.poolRequestedBytes(), alc = model.poolAllocatedBytes();
      std::printf(
        "activation pool: requested %.1f MB -> allocated %.1f MB (%zu buffers, reuse %.0f%%)\n",
        req / 1e6, alc / 1e6, model.poolBufferCount(),
        req ? 100.0 * (1.0 - static_cast<double>(alc) / static_cast<double>(req)) : 0.0);
      std::printf("total kernel time: %.3f ms (iters=%d)\n", total / iters, iters);
      std::vector<std::pair<std::string, std::pair<double, int>>> v(
        model.opProfile().begin(), model.opProfile().end());
      std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
      for (auto & kv : v)
        std::printf(
          "  %-12s %8.3f ms  x%d\n", kv.first.c_str(), kv.second.first / iters,
          kv.second.second / iters);
    }
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "[kernel_run] error: %s\n", e.what());
    return 1;
  }
  return 0;
}
