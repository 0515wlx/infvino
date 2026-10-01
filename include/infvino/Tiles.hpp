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
  // 经 round-1 sweep 选出的默认（见 docs/kernel.md）：大 K/N 与 YOLO 小 shape 折中最佳。
  int BM = 128, BN = 64, BK = 8, TM = 8, TN = 4, VEC2 = 0, PAD = 0;

  std::string options() const
  {
    std::ostringstream o;
    o << "-DBM=" << BM << " -DBN=" << BN << " -DBK=" << BK
      << " -DTM=" << TM << " -DTN=" << TN
      << " -DVEC2=" << VEC2 << " -DPAD=" << PAD
      << " -cl-mad-enable -cl-fast-relaxed-math";
    return o.str();
  }

  std::string label() const
  {
    std::ostringstream o;
    o << BM << "," << BN << "," << BK << "," << TM << "," << TN
      << " vec2=" << VEC2 << " pad=" << PAD;
    return o.str();
  }

  size_t localX() const { return static_cast<size_t>(BN / TN); }
  size_t localY() const { return static_cast<size_t>(BM / TM); }
};

/** @brief 解析 "BM,BN,BK,TM,TN[,VEC2,PAD]"；字段不足时沿用默认。 */
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
  return t;
}

/** @brief 3x3 直接卷积配置（见 kernels/conv.cl）。 */
struct Conv3x3Cfg
{
  int TX = 128, TY = 8, TM = 2, CB = 16, CINC = 16, STRIDE = 1, PAD = 1, ACT = 0,
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
