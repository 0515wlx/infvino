// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// 数值检验工具：用自研 OpenCL 后端(ClBackend)跑一次，把每个输出张量按 f32 dump 到文件，
// 供与 onnxruntime/pytorch 参考值逐一比对。
//
//   # 直接喂预处理后的 f32 NCHW 输入
//   infvino_numtest --config config/models.yaml --key yolov8n-pose --input in.bin --dump /tmp/inf_m
//
//   # 喂图片：内部做与引擎一致的预处理（letterbox/mean/std），验证预处理 + 后端
//   infvino_numtest --config config/models.yaml --key yolov8n-pose --image x.jpg --dump /tmp/inf_m
//
// 生成: /tmp/inf_m_out0.bin, /tmp/inf_m_out1.bin ...
// 配合 scripts/numerical_check.py / engine_check.py 使用。

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "infvino/ClBackend.hpp"
#include "infvino/Preprocess.hpp"

int main(int argc, char ** argv)
{
  std::string config, key, plan, device, input_path, image_path, dump_prefix;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    const auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
    if (a == "--config") config = next();
    else if (a == "--key") key = next();
    else if (a == "--plan") plan = next();
    else if (a == "--device") device = next();
    else if (a == "--input") input_path = next();
    else if (a == "--image") image_path = next();
    else if (a == "--dump") dump_prefix = next();
    else { std::cerr << "unknown arg " << a << "\n"; return 2; }
  }

  infvino::ModelInfo info;
  if (!config.empty())
  {
    info = infvino::ModelInfo::fromYaml(config, key);
  }
  if (!plan.empty()) info.plan = plan;
  if (!device.empty()) info.device = device;

  if (info.plan.empty())
  {
    std::cerr << "usage: infvino_numtest --config <yaml> [--key name] | --plan <plan>\n";
    return 2;
  }

  // 构造输入：优先图片（走 Preprocessor），否则读 f32 NCHW 二进制。
  std::vector<float> buf;
  if (!image_path.empty())
  {
    const cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) { std::cerr << "cannot read image " << image_path << "\n"; return 1; }
    infvino::Preprocessor pre(infvino::PreprocessConfig{
      info.input_size, info.letterbox, info.to_rgb, info.normalize, info.mean, info.std});
    const cv::Mat blob = pre(bgr);
    buf.assign(reinterpret_cast<const float *>(blob.data),
               reinterpret_cast<const float *>(blob.data) + blob.total() * blob.channels());
  }
  else if (!input_path.empty())
  {
    std::ifstream fin(input_path, std::ios::binary | std::ios::ate);
    if (!fin) { std::cerr << "cannot open input " << input_path << "\n"; return 1; }
    const std::streamsize bytes = fin.tellg();
    fin.seekg(0);
    buf.resize(static_cast<size_t>(bytes) / sizeof(float));
    fin.read(reinterpret_cast<char *>(buf.data()), bytes);
  }

  infvino::ClBackend backend(info);
  const auto & dev = backend.device();
  std::cout << "device: " << dev.resolved << " (" << dev.full_name << ")"
            << (dev.fell_back_to_cpu ? " [NON-GPU]" : "") << "\n";

  std::vector<infvino::Tensor> outputs = backend.infer(buf.data(), buf.size());
  for (size_t i = 0; i < outputs.size(); ++i)
  {
    const auto & t = outputs[i];
    std::cout << "output[" << i << "] shape=";
    for (auto d : t.shape) std::cout << d << " ";
    std::cout << "\n";

    if (!dump_prefix.empty())
    {
      const std::string path = dump_prefix + "_out" + std::to_string(i) + ".bin";
      std::ofstream fout(path, std::ios::binary);
      fout.write(
        reinterpret_cast<const char *>(t.data.data()),
        static_cast<std::streamsize>(t.data.size() * sizeof(float)));
    }
  }
  return 0;
}
