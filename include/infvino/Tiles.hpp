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
  // pipeline (DBUF=1) at BK=16. Double-buffering needs 2x SLM per workgroup, so
  // BK had to drop 32->16 to stay inside the ~16 KB SLM/WG budget; that trade is
  // a net win (~11.8 vs ~6.7 ops/EU/cyc @ 4096x512x512, see docs/kernel.md R9).
  int BM = 128, BN = 64, BK = 16, TM = 8, TN = 4, VEC2 = 0, PAD = 0, DBUF = 1;
  // Round 8: staging vector width (halfs) and async staging toggle.
  int VEC = 4, ASYNC = 0;
  // Round 9: bottleneck probes (diagnostic only).
  int SKIP_STAGE = 0, SKIP_COMPUTE = 0;
  // Round 11: store the A tile transposed so a thread's TM values are contiguous
  // and load as half8 (fewer SLM transactions). 0 = row-major A.
  int AT = 0;

  std::string options() const
  {
    std::ostringstream o;
    o << "-DBM=" << BM << " -DBN=" << BN << " -DBK=" << BK
      << " -DTM=" << TM << " -DTN=" << TN
      << " -DVEC2=" << VEC2 << " -DPAD=" << PAD << " -DDBUF=" << DBUF
      << " -DVEC=" << VEC << " -DASYNC=" << ASYNC
      << " -DSKIP_STAGE=" << SKIP_STAGE << " -DSKIP_COMPUTE=" << SKIP_COMPUTE
      << " -DAT=" << AT
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
      << " at=" << AT;
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
  return t;
}

/** @brief 3x3 直接卷积配置（见 kernels/conv.cl）。 */
struct Conv3x3Cfg
{
  int TX = 64, TY = 8, TM = 1, CB = 32, CINC = 16, STRIDE = 1, PAD = 1, ACT = 0,
      UNROLL_CI = 3, VECC = 1;

  std::string options() const
  {
    std::ostringstream o;
    o << "-DTX=" << TX << " -DTY=" << TY << " -DTM=" << TM << " -DCB=" << CB
      << " -DCINC=" << CINC << " -DSTRIDE=" << STRIDE
      << " -DPAD=" << PAD << " -DACT=" << ACT << " -DUNROLL_CI=" << UNROLL_CI
      << " -DVECC=" << VECC
      << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  }
  std::string label() const
  {
    std::ostringstream o;
    o << "TX" << TX << " TY" << TY << " TM" << TM << " CB" << CB
      << " CINC" << CINC << " s" << STRIDE << " act" << ACT << " u" << UNROLL_CI
      << " v" << VECC;
    return o.str();
  }
};

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
  return c;
}

}  // namespace gk

#endif  // INFVINO_GK__TILES_HPP_
