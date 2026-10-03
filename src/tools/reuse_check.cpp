// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// reuse_check —— 跨推理一致性（P0/P1 回归护栏）。
//
// 背景：PlanModel 里有些按张量名缓存的东西（块布局重排、池化缓冲）。一旦把「每帧都会变的
// 激活」错误地缓存成「只算一次」，第 2 帧及以后就会用到上一帧的数据——而「单输入重复跑」
// 的测试永远发现不了。
//
// 本工具在同一进程里先喂 A、再喂 B，并把**第二帧(B)**的输出 dump 成 f32。
// 与「全新进程只喂 B」的结果（kernel_run / infvino_numtest dump）比较：
// 一致 ⇒ 无跨推理陈旧；不一致 ⇒ 缓存 bug。调用方脚本负责两次运行与 diff。
//
// 用法:
//   reuse_check --config config/models.yaml --key yolov8n-pose --a A.bin --b B.bin --dump-b out.bin
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "infvino/ClBackend.hpp"
#include "infvino/Half.hpp"
#include "infvino/ModelInfo.hpp"

static std::vector<float> loadF16(const std::string & path, size_t n)
{
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  std::vector<uint16_t> h(n);
  f.read(reinterpret_cast<char *>(h.data()), static_cast<std::streamsize>(n * 2));
  std::vector<float> o(n);
  for (size_t i = 0; i < n; ++i) o[i] = gk::f16_to_f32(h[i]);
  return o;
}

int main(int argc, char ** argv)
{
  std::string config, key, a_path, b_path, dump_b;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--config") config = next();
    else if (a == "--key") key = next();
    else if (a == "--a") a_path = next();
    else if (a == "--b") b_path = next();
    else if (a == "--dump-b") dump_b = next();
    else { std::fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
  }
  if (config.empty() || key.empty() || a_path.empty() || b_path.empty() || dump_b.empty())
  {
    std::fprintf(stderr,
                 "usage: reuse_check --config <yaml> --key <m> --a A.bin --b B.bin --dump-b f32\n");
    return 2;
  }
  try
  {
    auto info = infvino::ModelInfo::fromYaml(config, key);
    infvino::ClBackend be(info);
    const size_t in_n = 3 * static_cast<size_t>(info.input_size.height * info.input_size.width);
    auto A = loadF16(a_path, in_n);
    auto B = loadF16(b_path, in_n);
    be.infer(A.data(), in_n);                 // 帧 1
    auto rB = be.infer(B.data(), in_n);       // 帧 2：不同输入，同进程
    const auto & b = rB[0].data;
    std::ofstream f(dump_b, std::ios::binary);
    f.write(reinterpret_cast<const char *>(b.data()), static_cast<std::streamsize>(b.size() * 4));
    std::printf("reuse_check: wrote frame2(B) -> %s (%zu elems)\n", dump_b.c_str(), b.size());
    return 0;
  }
  catch (const std::exception & e)
  {
    std::fprintf(stderr, "reuse_check error: %s\n", e.what());
    return 1;
  }
}
