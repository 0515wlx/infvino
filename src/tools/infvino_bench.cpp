// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 独立基准/调试工具（不依赖 ROS）：
//   infvino_bench --config models.yaml --key yolov8n-pose
//   infvino_bench --plan models/yolov8n-pose/model.plan --task pose --kpt 17,3 --image x.jpg

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "infvino/InferenceEngine.hpp"

namespace
{
std::vector<int> parsePair(const std::string & s)
{
  std::vector<int> out;
  const auto comma = s.find(',');
  if (comma == std::string::npos)
  {
    out.push_back(std::stoi(s));
  }
  else
  {
    out.push_back(std::stoi(s.substr(0, comma)));
    out.push_back(std::stoi(s.substr(comma + 1)));
  }
  return out;
}
} // namespace

int main(int argc, char ** argv)
{
  std::string yaml, key, model, plan, task, device, image;
  int         nc = -1, iters = 200, imgsz = 0;
  double      gflops = 0.0, eu = 80.0, clock_ghz = 1.3;
  std::vector<int> kpt;

  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--config") yaml = next();
    else if (a == "--key") key = next();
    else if (a == "--model") model = next();
    else if (a == "--plan") plan = next();
    else if (a == "--task") task = next();
    else if (a == "--nc") nc = std::stoi(next());
    else if (a == "--device") device = next();
    else if (a == "--imgsz") imgsz = std::stoi(next());
    else if (a == "--iters") iters = std::stoi(next());
    else if (a == "--image") image = next();
    else if (a == "--kpt") kpt = parsePair(next());
    else if (a == "--gflops") gflops = std::stod(next());
    else if (a == "--eu") eu = std::stod(next());
    else if (a == "--clock") clock_ghz = std::stod(next());
    else { std::cerr << "unknown arg: " << a << "\n"; return 2; }
  }

  infvino::ModelInfo info;
  if (!yaml.empty())
  {
    info = infvino::ModelInfo::fromYaml(yaml, key);
  }
  if (!model.empty()) info.path = model;
  if (!plan.empty()) info.plan = plan;
  if (!task.empty()) info.task = infvino::taskFromString(task);
  if (nc > 0) info.num_classes = nc;
  if (!kpt.empty()) info.kpt_shape = kpt;
  if (imgsz > 0) info.input_size = {imgsz, imgsz};
  if (!device.empty()) info.device = device;

  if (info.plan.empty())
  {
    std::cerr
      << "usage: infvino_bench --config <yaml> [--key name] | --plan <plan> [--model <onnx> ...]\n"
      << "  (需要 .plan：先跑 python3 scripts/onnx2plan.py --onnx <onnx> --out-dir models)\n";
    return 2;
  }

  infvino::InferenceEngine engine(info);
  const auto & dev = engine.device();
  std::cout << "model      : " << info.path << "  task=" << infvino::toString(info.task)
            << "  nc=" << info.num_classes << "  imgsz=" << info.input_size.width << "x"
            << info.input_size.height << "\n";
  std::cout << "devices    :";
  for (const auto & d : dev.available) std::cout << " " << d;
  std::cout << "\n";
  std::cout << "requested  : " << dev.requested << " -> resolved: " << dev.resolved
            << (dev.fell_back_to_cpu ? "  [CPU FALLBACK]" : "") << "  (" << dev.full_name << ")\n";

  cv::Mat img = image.empty() ? cv::Mat(1080, 1920, CV_8UC3, cv::Scalar(0)) : cv::imread(image);
  if (img.empty()) { std::cerr << "cannot read image / empty\n"; return 1; }

  auto session = engine.createSession();
  for (int i = 0; i < 20; ++i) session->infer(img);

  std::vector<double> total_ts, infer_ts;
  total_ts.reserve(iters);
  infer_ts.reserve(iters);
  infvino::InferResult last;
  for (int i = 0; i < iters; ++i)
  {
    last = session->infer(img);
    total_ts.push_back(last.total_ms);
    infer_ts.push_back(last.infer_ms);
  }
  const auto stats = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
    return std::array<double, 4>{mean, v[v.size() / 2], v[static_cast<size_t>(0.9 * v.size())], v.front()};
  };
  const auto tot = stats(total_ts);
  const auto inf = stats(infer_ts);
  const double total_fps = 1000.0 / tot[0];
  const double infer_fps = 1000.0 / inf[0];

  if (!last.topk.empty())
  {
    std::cout << "\ntopk       :";
    for (const auto & kv : last.topk) std::cout << "  #" << kv.first << "=" << kv.second;
    std::cout << "\n";
  }
  else
  {
    std::cout << "\ndetections : " << last.detections.size() << "\n";
    for (size_t i = 0; i < last.detections.size() && i < 10; ++i)
    {
      const auto & d = last.detections[i];
      std::cout << "   #" << i << " cls=" << d.class_id << " score=" << d.score
                << " box=[" << d.box.x << "," << d.box.y << "," << d.box.width << "," << d.box.height
                << "] kpts=" << d.keypoints.size() << "\n";
    }
  }

  std::cout << "\nlatency(ms) total: mean=" << tot[0] << " p50=" << tot[1] << " p90=" << tot[2]
            << " min=" << tot[3] << "  -> " << total_fps << " fps (pipeline)\n";
  std::cout << "latency(ms) infer: mean=" << inf[0] << " p50=" << inf[1] << " p90=" << inf[2]
            << " min=" << inf[3] << "  -> " << infer_fps << " fps (net only)\n";
  std::cout << "breakdown  : pre=" << last.preprocess_ms << " infer=" << last.infer_ms
            << " post=" << last.postprocess_ms << "\n";
  if (gflops > 0.0)
  {
    const double ops_eu_cyc = gflops * 1e9 * infer_fps / (eu * clock_ghz * 1e9);
    std::cout << "efficiency : " << gflops << " GFLOPs -> " << ops_eu_cyc
              << " ops/EU/cyc = " << (ops_eu_cyc / 32.0 * 100.0) << "% of 32"
              << "  (assume EU=" << eu << ", clock=" << clock_ghz << "GHz)\n";
  }
  return 0;
}
