// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_kernel_families —— 算子族注册表的结构不变量（R48 §3.1 / R48 D4 / R31）。
//
// 注册表是「候选 / 布局契约 / 上限」的单一真相源；一处声明写错会静默污染选择与布局
// （R48 D4 的 inIndex 槽位 bug 就是这样冒出来的）。覆盖：
//   * 名字唯一、kernel 源文件真实存在（族声明与 .cl 不漂移）；
//   * familyByName 往返；候选 source 对应的 .cl 存在；
//   * 候选枚举确定性；代表签名非空；actMask 过滤生效；
//   * inIndex 布局契约（conv1x1/gemm 槽 0 是权重，激活在槽 1）；
//   * canOutFsv16 由族契约传播到候选；
//   * R31 负结果：depthwise_vp 默认不进候选。
//
// 纯 CPU（只读 kernels/ 目录），不需要 GPU。
#include <filesystem>
#include <set>
#include <string>
#include <vector>

#include "infvino/KernelFamily.hpp"
#include "test_util.hpp"

using namespace infvino;

namespace
{
const std::filesystem::path kKernelDir = INFVINO_KERNEL_DIR;

bool sourceExists(const std::string & src)
{
  return !src.empty() && std::filesystem::exists(kKernelDir / (src + ".cl"));
}

// 枚举一个签名并检查每个候选的源文件存在；返回候选数。
size_t checkCandidates(const OpSignature & sig, const char * label)
{
  const auto cands = candidatesFromRegistry(sig);
  const std::string msg = std::string(label) + ": candidates non-empty";
  CHECK(!cands.empty(), msg.c_str());
  for (const auto & c : cands)
  {
    const std::string m = std::string(label) + ": candidate source '" + c.source + "' exists";
    CHECK(sourceExists(c.source), m.c_str());
    const std::string m2 = std::string(label) + ": candidate kernel name non-empty";
    CHECK(!c.kernel.empty(), m2.c_str());
  }
  return cands.size();
}

bool hasKernel(const std::vector<Candidate> & cs, const std::string & k)
{
  for (const auto & c : cs)
    if (c.kernel == k) return true;
  return false;
}
}  // namespace

static void run_tests()
{
  // --- 族表结构自洽 ---
  {
    const auto & fams = kernelFamilies();
    CHECK(!fams.empty(), "registry non-empty");
    std::set<std::string> names;
    for (const auto & f : fams)
    {
      CHECK(!f.name.empty(), "family name non-empty");
      CHECK(names.insert(f.name).second, "family names unique");
      const std::string m = "family '" + f.name + "' source file exists";
      CHECK(sourceExists(f.source), m.c_str());
      // familyByName 往返：应返回表内同一元素。
      CHECK(familyByName(f.name) == &f, "familyByName round-trips to same entry");
    }
    CHECK(familyByName("does_not_exist_family") == nullptr, "unknown family -> nullptr");
  }

  // --- 代表签名：候选非空 + 候选源存在 ---
  {
    checkCandidates(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 0), "conv3x3 s1");
    checkCandidates(OpSignature::conv3x3(160, 160, 2, 1, 3, 16, 1), "conv3x3 stem s2");
    checkCandidates(OpSignature::gemm(128, 1024, 256, 0), "gemm");
    checkCandidates(OpSignature::conv1x1(256, 196, 256, 0, 0), "conv1x1 N>1");
    checkCandidates(OpSignature::conv1x1(1000, 1, 1024, 0, 0), "conv1x1 N==1");
    checkCandidates(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 1), "depthwise");
    checkCandidates(OpSignature::gap(64, 784), "gap");
    checkCandidates(OpSignature::custom("copy_c", {25600, 16}), "copy_c");
    checkCandidates(OpSignature::custom("bmm", {1, 2, 64, 400, 400}), "bmm");
  }

  // --- 代表族确实被列入 ---
  {
    const auto c1 = candidatesFromRegistry(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 0));
    CHECK(hasKernel(c1, "conv3x3_ov"), "conv3x3 s1 includes conv3x3_ov");
    CHECK(hasKernel(c1, "conv3x3_blk"), "conv3x3 s1 includes conv3x3_blk");
    CHECK(hasKernel(c1, "conv3x3_f16"), "conv3x3 s1 includes conv3x3_f16");

    const auto stem = candidatesFromRegistry(OpSignature::conv3x3(160, 160, 2, 1, 3, 16, 1));
    CHECK(hasKernel(stem, "conv3x3_cin3"), "Cin=3 stem includes conv3x3_cin3");

    const auto c11 = candidatesFromRegistry(OpSignature::conv1x1(256, 196, 256, 0, 0));
    CHECK(hasKernel(c11, "conv1x1_blk"), "conv1x1 N>1 includes conv1x1_blk");

    const auto c11n1 = candidatesFromRegistry(OpSignature::conv1x1(1000, 1, 1024, 0, 0));
    CHECK(hasKernel(c11n1, "conv1x1_gemv_f16"), "conv1x1 N==1 includes gemv");

    const auto dw = candidatesFromRegistry(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 1));
    CHECK(hasKernel(dw, "depthwise_f16"), "depthwise includes depthwise_f16");
    CHECK(hasKernel(dw, "depthwise_v"), "depthwise includes depthwise_v");
    CHECK(hasKernel(dw, "depthwise_blk"), "depthwise includes depthwise_blk");
    CHECK(!hasKernel(dw, "depthwise_vp"), "R31: depthwise_vp absent by default");
  }

  // --- 确定性：同签名两次枚举结果逐项一致 ---
  {
    const auto sig = OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1);
    const auto a = candidatesFromRegistry(sig);
    const auto b = candidatesFromRegistry(sig);
    CHECK_EQ(a.size(), b.size(), "candidate enumeration deterministic (size)");
    bool same = a.size() == b.size();
    for (size_t i = 0; same && i < a.size(); ++i)
      same = a[i].kernel == b[i].kernel && a[i].options == b[i].options &&
             a[i].config == b[i].config && a[i].source == b[i].source;
    CHECK(same, "candidate enumeration deterministic (elements)");
  }

  // --- actMask 过滤：注册表拒绝语义不支持的激活码 ---
  {
    // conv3x3 只实现 {0,1,3}；act=2 (ReLU) 不在此列 -> 无候选。
    const auto bad = candidatesFromRegistry(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 2));
    CHECK(bad.empty(), "conv3x3 act=2 rejected by actMask");
    // depthwise 实现 {0..4}；act=5 (Sigmoid) 被拒。
    const auto bad_dw = candidatesFromRegistry(OpSignature::depthwise(80, 80, 1, 1, 64, 3, 5));
    CHECK(bad_dw.empty(), "depthwise act=5 rejected by actMask");
    // act=1 (SiLU) 仍支持。
    CHECK(!candidatesFromRegistry(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 1)).empty(),
          "conv3x3 act=1 accepted");
  }

  // --- R48 D4: 布局契约 inIndex（槽 0 权重 vs 激活）---
  {
    const KernelFamily * gemm = familyByName("gemm_f16");
    const KernelFamily * gemm_cat4 = familyByName("gemm_cat4_f16");
    const KernelFamily * gemv = familyByName("conv1x1_gemv_f16");
    const KernelFamily * c1x1blk = familyByName("conv1x1_blk");
    const KernelFamily * dwblk = familyByName("depthwise_blk");
    const KernelFamily * convblk = familyByName("conv3x3_blk");
    const KernelFamily * gapfsv = familyByName("gap_fsv16");
    CHECK(gemm && gemm->layout.inIndex == 1, "gemm_f16 activation in slot 1 (R48 D4)");
    CHECK(gemm_cat4 && gemm_cat4->layout.inIndex == 1, "gemm_cat4_f16 activation in slot 1");
    CHECK(gemv && gemv->layout.inIndex == 1, "conv1x1_gemv activation in slot 1");
    CHECK(c1x1blk && c1x1blk->layout.inIndex == 1, "conv1x1_blk activation in slot 1 (R48 D4)");
    CHECK(dwblk && dwblk->layout.inIndex == 0, "depthwise_blk activation in slot 0");
    CHECK(convblk && convblk->layout.inIndex == 0, "conv3x3_blk activation in slot 0");
    CHECK(gapfsv && gapfsv->layout.inIndex == 0, "gap_fsv16 activation in slot 0");
    CHECK(gapfsv && gapfsv->layout.in == Layout::FSV16, "gap_fsv16 input layout is FSV16");

    // 通用契约不变量（防止新增族漏声明 inIndex）：凡服务「槽 0 = 权重」的 op（gemm /
    // conv1x1 / conv1x1_cat4），激活都在槽 1；其余 op 激活在槽 0。曾经 gemm_sk_f16
    // 漏声明 → default 0，与同族其它成员不一致（潜在布局/分配漂移）。
    const KernelFamily * sk = familyByName("gemm_sk_f16");
    CHECK(sk && sk->layout.inIndex == 1, "gemm_sk_f16 activation in slot 1 (contract fix)");
    for (const auto & f : kernelFamilies())
    {
      const bool weightFirst = (f.op == "gemm" || f.op == "conv1x1" || f.op == "conv1x1_cat4");
      const int want = weightFirst ? 1 : 0;
      const std::string m =
          "family '" + f.name + "' (" + f.op + ") declares activation slot " + std::to_string(want);
      CHECK_EQ(f.layout.inIndex, want, m.c_str());
    }
  }

  // --- canOutFsv16 由族契约传播到候选 ---
  {
    const auto conv = candidatesFromRegistry(OpSignature::conv3x3(80, 80, 1, 1, 64, 64, 0));
    for (const auto & c : conv)
    {
      if (c.kernel == "conv3x3_blk")
        CHECK(c.canOutFsv16, "conv3x3_blk candidate canOutFsv16 propagated");
      if (c.kernel == "conv3x3_ov")
        CHECK(!c.canOutFsv16, "conv3x3_ov candidate cannot out fsv16");
    }
    const auto c11 = candidatesFromRegistry(OpSignature::conv1x1(256, 196, 256, 0, 0));
    bool saw_blk = false;
    for (const auto & c : c11)
    {
      if (c.kernel == "conv1x1_blk") { saw_blk = true; CHECK(c.canOutFsv16, "conv1x1_blk canOutFsv16"); }
    }
    CHECK(saw_blk, "conv1x1_blk present for N>1/Cin>=16");
  }
}

ITEST_MAIN("test_kernel_families")
