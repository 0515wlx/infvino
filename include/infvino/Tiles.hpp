// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// GEMM/conv kernel 的编译期 tile 配置（bench 与 numtest 共用）。
#ifndef INFVINO_GK__TILES_HPP_
#define INFVINO_GK__TILES_HPP_

#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

namespace gk
{

struct Tiles
{
  // Round 8/9 defaults: BM=128,BN=64,TM=8,TN=4,VEC=4 with the double-buffered
  // pipeline (DBUF=1) at BK=16. Round 12 added SG=16 (force SIMD16) which lifts
  // this default to ~13.2 ops/EU/cyc @4096x512x512; PlanModel switches to the
  // BK=32 single-buffer variant (~13.7) for large grids.
  int BM = 128, BN = 64, BK = 16, TM = 8, TN = 4, VEC2 = 0, PAD = 0, DBUF = 1;
  // Round 8: staging vector width (halfs) and async staging toggle.
  int VEC = 4, ASYNC = 0;
  // Round 9: bottleneck probes (diagnostic only).
  int SKIP_STAGE = 0, SKIP_COMPUTE = 0;
  // Round 11: store the A tile transposed so a thread's TM values are contiguous
  // and load as half8 (fewer SLM transactions). 0 = row-major A.
  int AT = 0;
  // Round 12: single-SLM-buffer software pipeline that prefetches the next tile
  // into registers (lets BK double within the same SLM budget). 1 = on.
  int PF = 0;
  // Round 12: no-SLM probe — read A/B straight from global memory (L1/L2),
  // zero __local and zero barriers. Diagnostic only.
  int GN = 0;
  // Round 12: drop the per-k-tile barrier (diagnostic only; wrong results, used
  // with SKIP_STAGE to isolate the barrier's true cost).
  int SB = 0;
  // Round 12: force intel_reqd_sub_group_size(SG). SG is the requested SIMD
  // width (0 = let IGC decide; 8/16/32 are the useful values). Without it IGC
  // silently drops the *full* kernel to SIMD8 under staging register pressure.
  // 16 is the measured sweet spot on Xe-LP (32 spills, 8 starves the FPU).
  int SG = 16;
  // Round 14: diagnostic — keep the mad structure but drop the per-kk SLM
  // operand loads (register constants instead), to isolate the feed cost.
  int NOLOAD = 0;
  // Round 14: kk-level software pipeline (prefetch next kk's A/B during mads).
  int PIPE = 0;
  // Round 22: fused bias+activation epilogue (see gemm.cl). EPI=1 adds a Bias
  // kernel argument; ACT is the activation code (0 none/1 silu/2 hardswish/3 relu/4 hardsigmoid).
  int EPI = 0, ACT = 0;

  std::string options() const
  {
    std::ostringstream o;
    o << "-DBM=" << BM << " -DBN=" << BN << " -DBK=" << BK
      << " -DTM=" << TM << " -DTN=" << TN
      << " -DVEC2=" << VEC2 << " -DPAD=" << PAD << " -DDBUF=" << DBUF
      << " -DVEC=" << VEC << " -DASYNC=" << ASYNC
      << " -DSKIP_STAGE=" << SKIP_STAGE << " -DSKIP_COMPUTE=" << SKIP_COMPUTE
      << " -DAT=" << AT << " -DPF=" << PF << " -DGN=" << GN
      << " -DSKIP_BARRIER=" << SB << " -DSG=" << SG << " -DNOLOAD=" << NOLOAD
      << " -DPIPE=" << PIPE
      << " -DEPI=" << EPI << " -DACT=" << ACT
      << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  }

  std::string label() const
  {
    std::ostringstream o;
    o << BM << "," << BN << "," << BK << "," << TM << "," << TN
      << " vec2=" << VEC2 << " pad=" << PAD << " dbuf=" << DBUF
      << " vec=" << VEC << " async=" << ASYNC
      << (SKIP_STAGE ? " skipstage" : "") << (SKIP_COMPUTE ? " skipcompute" : "")
      << " at=" << AT << " pf=" << PF << (GN ? " gn" : "") << (SB ? " skipbar" : "");
    return o.str();
  }

  size_t localX() const { return static_cast<size_t>(BN / TN); }
  size_t localY() const { return static_cast<size_t>(BM / TM); }
};

/** @brief 解析 "BM,BN,BK,TM,TN[,VEC2,PAD,DBUF,VEC,ASYNC[,SKIP_STAGE,SKIP_COMPUTE]]"；字段不足时沿用默认。 */
inline Tiles parseTiles(const std::string & s)
{
  Tiles t;
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
  if (v.size() > 0) t.BM = v[0];
  if (v.size() > 1) t.BN = v[1];
  if (v.size() > 2) t.BK = v[2];
  if (v.size() > 3) t.TM = v[3];
  if (v.size() > 4) t.TN = v[4];
  if (v.size() > 5) t.VEC2 = v[5];
  if (v.size() > 6) t.PAD = v[6];
  if (v.size() > 7) t.DBUF = v[7];
  if (v.size() > 8) t.VEC = v[8];
  if (v.size() > 9) t.ASYNC = v[9];
  if (v.size() > 10) t.SKIP_STAGE = v[10];
  if (v.size() > 11) t.SKIP_COMPUTE = v[11];
  if (v.size() > 12) t.AT = v[12];
  if (v.size() > 13) t.PF = v[13];
  if (v.size() > 14) t.GN = v[14];
  if (v.size() > 15) t.SB = v[15];
  if (v.size() > 16) t.SG = v[16];
  if (v.size() > 17) t.NOLOAD = v[17];
  if (v.size() > 18) t.PIPE = v[18];
  if (v.size() > 19) t.EPI = v[19];
  if (v.size() > 20) t.ACT = v[20];
  return t;
}

/** @brief 3x3 直接卷积配置（见 kernels/conv.cl）。 */
struct Conv3x3Cfg
{
  int TX = 64, TY = 8, TM = 1, CB = 32, CINC = 16, STRIDE = 1, PAD = 1, ACT = 0,
      UNROLL_CI = 3, VECC = 1;
  // Round 16: register-tiled kernel (conv3x3_rt): TN channels per thread.
  int TN = 8;
  int RT = 0;
  // Round 17: output-channel-vectorized kernel (conv3x3_osv): sub-group lanes = output
  // channels (SG), each lane owns VECO consecutive channels x TM output columns.
  int OSV = 0;
  int VECO = 2;
  // Round 17b: SLM-free sub-group-broadcast kernel (conv3x3_sg). TX=OBW, TY=OBH,
  // CB=SG*VECO; SGK=1 selects it.
  int SGK = 0;
  // Round 17: SLM weight-load vector width for conv3x3_f16 (2/4/8 halfs).
  int WVEC = 2;
  // Round 17: diagnostic probe (0/1/2/3) — see conv.cl.
  int PROBE = 0;
  // Round 15: SIMD width override (0 = IGC decides; 16 avoids the SIMD8 cliff).
  int SG = 0;
  // Round 18: double-buffered CINC software pipeline (kernels/conv.cl:conv3x3_db).
  int DB = 0;
  // Round 18: coalesced weight staging loop order (kernels/conv.cl: -DWCOAL).
  // On by default — pure loop reorder, numerically identical, +5–12% on s1.
  int WC = 1;
  // Round 18: skip input SLM staging, read strips directly from global (-DXGN).
  int XG = 0;
  // Round 19: skip weight SLM tile, read repacked weights from GPU L3 (-DWGL).
  int WGL = 0;
  // Round 22: OpenVINO os_iyx_osv32 port (kernels/conv_ov.cl). TX=OBW, TY=OBH.
  int OV = 0;
  // Round 25: OpenVINO blocked conv port (kernels/conv_blk.cl), OBW = TX.
  int BLK = 0;

  std::string options() const
  {
    std::ostringstream o;
    o << "-DTX=" << TX << " -DTY=" << TY << " -DTM=" << TM << " -DTN=" << TN
      << " -DCB=" << CB
      << " -DCINC=" << CINC << " -DSTRIDE=" << STRIDE
      << " -DPAD=" << PAD << " -DACT=" << ACT << " -DUNROLL_CI=" << UNROLL_CI
      << " -DVECC=" << VECC << " -DSG=" << SG << " -DVECO=" << VECO
      << " -DWVEC=" << WVEC << " -DPROBE=" << PROBE
      << " -DWCOAL=" << WC
      << " -DXGN=" << XG
      << " -DWGL=" << WGL
      << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  }
  std::string label() const
  {
    std::ostringstream o;
    o << "TX" << TX << " TY" << TY << " TM" << TM << " TN" << TN << " CB" << CB
      << " CINC" << CINC << " s" << STRIDE << " act" << ACT << " u" << UNROLL_CI
      << " v" << VECC << " sg" << SG << " veco" << VECO << (RT ? " rt" : "") << (OSV ? " osv" : "") << (SGK ? " sgk" : "") << (DB ? " db" : "");
    return o.str();
  }
};

/**
 * @brief 1x1 卷积（pointwise）专用 kernel 配置（见 kernels/conv1x1.cl）。
 *
 * 每个 work-item 计算 TM 个输出通道 x TN 个空间位置的输出块，沿 Cin 归约；
 * 融合 bias、激活与（可选）残差，取代「gemm + bias_add + ew_unary」三连击。
 */
struct Conv1x1Cfg
{
  int TM = 4;    // 每 work-item 输出通道数
  int TN = 4;    // 每 work-item 空间位置数
  int ACT = 0;   // 0=none 1=SiLU 2=Relu 3=Hardswish 4=Hardsigmoid 5=Sigmoid
  int RES = 0;   // 1 = epilogue 加残差
  int SG = 16;   // 强制子组宽度（0=IGC 决定）
  int UNROLL = 4;  // Cin 循环展开因子（提升 load/FMA 重叠）

  std::string options() const
  {
    std::ostringstream o;
    o << "-DTM=" << TM << " -DTN=" << TN << " -DACT=" << ACT << " -DRES=" << RES
      << " -DSG=" << SG << " -DUNROLL=" << UNROLL
      << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  }
  std::string label() const
  {
    std::ostringstream o;
    o << "TM" << TM << " TN" << TN << " act" << ACT << (RES ? " res" : "") << " sg" << SG
      << " u" << UNROLL;
    return o.str();
  }
};

/** @brief 解析 "TM,TN[,ACT,RES,SG]"。 */
inline Conv1x1Cfg parseConv1x1(const std::string & s)
{
  Conv1x1Cfg c;
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
  if (v.size() > 0) c.TM = v[0];
  if (v.size() > 1) c.TN = v[1];
  if (v.size() > 2) c.ACT = v[2];
  if (v.size() > 3) c.RES = v[3];
  if (v.size() > 4) c.SG = v[4];
  if (v.size() > 5) c.UNROLL = v[5];
  return c;
}

/** @brief 解析 "TX,TY,TM,CB,CINC[,STRIDE,PAD,ACT,UNROLL_CI,VECC]"。 */
inline Conv3x3Cfg parseConv(const std::string & s)
{
  Conv3x3Cfg c;
  std::vector<int> v;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) v.push_back(std::atoi(tok.c_str()));
  if (v.size() > 0) c.TX = v[0];
  if (v.size() > 1) c.TY = v[1];
  if (v.size() > 2) c.TM = v[2];
  if (v.size() > 3) c.CB = v[3];
  if (v.size() > 4) c.CINC = v[4];
  if (v.size() > 5) c.STRIDE = v[5];
  if (v.size() > 6) c.PAD = v[6];
  if (v.size() > 7) c.ACT = v[7];
  if (v.size() > 8) c.UNROLL_CI = v[8];
  if (v.size() > 9) c.VECC = v[9];
  if (v.size() > 10) c.SG = v[10];
  if (v.size() > 11) c.TN = v[11];
  if (v.size() > 12) c.RT = v[12];
  if (v.size() > 13) c.OSV = v[13];
  if (v.size() > 14) c.VECO = v[14];
  if (v.size() > 15) c.SGK = v[15];
  if (v.size() > 16) c.WVEC = v[16];
  if (v.size() > 17) c.PROBE = v[17];
  if (v.size() > 18) c.DB = v[18];
  if (v.size() > 19) c.WC = v[19];
  if (v.size() > 20) c.XG = v[20];
  if (v.size() > 21) c.WGL = v[21];
  return c;
}

}  // namespace gk

#endif  // INFVINO_GK__TILES_HPP_
