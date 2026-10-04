// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// kernel_run —— 执行一份“算子计划”（由 scripts/onnx2plan.py 生成）的开发/调试 CLI。
// 库内生产路径由 infvino::PlanModel 承担；本工具额外支持 --report / --iters / 文件 dump。
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
#include <map>
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

std::string jsonEscape(const std::string & s)
{
  std::string o;
  o.reserve(s.size() + 2);
  for (char c : s)
  {
    switch (c)
    {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\t': o += "\\t"; break;
      default: o += c;
    }
  }
  return o;
}

// Phase 2: 把结构化剖面写成机器可读 JSON（busy/net/e2e 分析框架的来源）。
void writeProfileJson(
  const std::string & path, const std::string & label, const std::string & plan,
  const std::string & device, int iters, const infvino::PlanModel::PlanProfile & p,
  const std::map<std::string, std::pair<double, int>> & ops)
{
  const double inv = iters > 0 ? 1.0 / iters : 1.0;
  std::ofstream f(path);
  if (!f) { std::fprintf(stderr, "[kernel_run] cannot write %s\n", path.c_str()); return; }
  f << "{\n";
  f << "  \"model\": \"" << jsonEscape(label) << "\",\n";
  f << "  \"plan\": \"" << jsonEscape(plan) << "\",\n";
  f << "  \"device\": \"" << jsonEscape(device) << "\",\n";
  f << "  \"iters\": " << iters << ",\n";
  f << "  \"timing_ms\": {\"wall\": " << p.wall_ms << ", \"busy\": " << p.busy_ms * inv
    << ", \"enqueue\": " << p.enqueue_ms * inv << ", \"sync\": " << p.sync_ms * inv << "},\n";
  f << "  \"structural\": {\"plan_nodes\": " << p.plan_nodes
    << ", \"dispatched_nodes\": " << p.dispatched_nodes << ", \"skipped_nodes\": " << p.skipped_nodes
    << ", \"dispatches_per_frame\": " << p.dispatches * inv << ", \"reorder_calls_per_frame\": "
    << p.reorder_calls * inv << ", \"reorder_ms_per_frame\": " << p.reorder_ms * inv
    << ", \"fusions_res\": " << p.fusions_res << ", \"fusions_concat\": " << p.fusions_concat
    << ", \"fsv16_tensors\": " << p.fsv16_tensors << "},\n";
  f << "  \"memory\": {\"requested_bytes\": " << p.pool_requested
    << ", \"allocated_bytes\": " << p.pool_allocated << ", \"buffers\": " << p.pool_buffers << "},\n";
  f << "  \"ops\": [\n";
  bool first = true;
  for (const auto & kv : ops)
  {
    f << (first ? "" : ",\n") << "    {\"tag\": \"" << jsonEscape(kv.first)
      << "\", \"ms_per_frame\": " << kv.second.first * inv << ", \"calls_per_frame\": "
      << kv.second.second * inv << "}";
    first = false;
  }
  f << "\n  ],\n  \"nodes\": [\n";
  first = true;
  for (const auto & r : p.nodes)
  {
    f << (first ? "" : ",\n") << "    {\"index\": " << r.index << ", \"op\": \""
      << jsonEscape(r.op) << "\", \"tag\": \"" << jsonEscape(r.tag) << "\", \"signature\": \""
      << jsonEscape(r.signature) << "\", \"ms_per_frame\": " << r.ms * inv
      << ", \"calls_per_frame\": " << r.calls * inv << "}";
    first = false;
  }
  f << "\n  ]\n}\n";
  std::fprintf(stderr, "[kernel_run] wrote %s\n", path.c_str());
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string plan_path, kernel_dir, input_path, output_path, out_dir, dump_tensor, input2_path;
  std::string profile_json, profile_label;
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
    else if (a == "--profile-json") profile_json = next();
    else if (a == "--profile-label") profile_label = next();
    else { std::fprintf(stderr, "unknown arg: %s\n", a.c_str()); return 2; }
  }
  if (plan_path.empty()) { std::fprintf(stderr, "usage: kernel_run --plan <plan.txt>\n"); return 2; }
  if (iters < 1) iters = 1;

  try
  {
    const std::string kdir = kernel_dir.empty() ? INFVINO_KERNEL_DIR : kernel_dir;
    infvino::PlanModel     model(plan_path, kdir, -1, 0, report);

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

    // Phase 2：在 iters 循环之后、input2 帧之前抓取结构化剖面（input2 不计入统计）。
    const infvino::PlanModel::PlanProfile prof     = model.profile();
    const auto                              prof_ops = model.opProfile();

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
      if (model.runtime().programCacheEnabled())
        std::printf("program cache: hits=%zu misses=%zu (on-disk kernel binaries)\n",
                    model.runtime().programCacheHits(), model.runtime().programCacheMisses());
      const double busy = total / iters;
      std::printf("total kernel time: %.3f ms (iters=%d)\n", busy, iters);
      // P2: wall - busy 的构成（仅 profiling 下有 host 分段）：入队提交 / 同步等待 /
      // 其余 host（setArg、循环/视图簿记）。wait 含 GPU 执行，故 host_total = wall - wait。
      {
        const double wall = model.lastRunMs();
        const double enq = model.hostEnqueueMs() / iters;
        const double wait = model.hostWaitMs() / iters;
        std::printf(
          "host segmentation: wall=%.3f busy=%.3f enqueue=%.3f sync=%.3f "
          "host_total~=%.3f setarg_est~=%.3f ms\n",
          wall, busy, enq, wait, wall - wait, wall - wait - enq);
      }
      // Phase 2: 结构计数（nodes / launch / 融合 / alias / 布局），每帧值。
      std::printf(
        "structural: nodes=%d launched=%d alias/skip=%d dispatches=%.1f reorder=%.1f(%.3fms) "
        "fusions(res=%d,concat=%d) fsv16=%d\n",
        prof.plan_nodes, prof.dispatched_nodes, prof.skipped_nodes,
        static_cast<double>(prof.dispatches) / iters,
        static_cast<double>(prof.reorder_calls) / iters, prof.reorder_ms / iters,
        prof.fusions_res, prof.fusions_concat, prof.fsv16_tensors);
      std::vector<std::pair<std::string, std::pair<double, int>>> v(
        model.opProfile().begin(), model.opProfile().end());
      std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second.first > b.second.first; });
      for (auto & kv : v)
        std::printf(
          "  %-12s %8.3f ms  x%d\n", kv.first.c_str(), kv.second.first / iters,
          kv.second.second / iters);
    }

    if (!profile_json.empty())
    {
      const std::string label = profile_label.empty() ? plan_path : profile_label;
      writeProfileJson(
        profile_json, label, plan_path, model.device().describe(), iters, prof, prof_ops);
    }
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "[kernel_run] error: %s\n", e.what());
    return 1;
  }
  return 0;
}
