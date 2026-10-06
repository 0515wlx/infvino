// Copyright (c) 2026 HEU-Wings-of-Dream. All Rights Reserved.
//
// test_util —— 极简离线单测骨架（**零第三方依赖**，与仓库 tuning_test 的风格一致）。
//
// 约定：每个 test_*.cpp 定义一个 `run_tests()`，末尾用 `ITEST_MAIN("名字")` 生成 main。
// 断言只累计失败（不 abort），保证一次跑完看到全部问题；退出码非 0 表示有失败。
//
// 这些测试**不需要 GPU / 容器**（只要 libinfvino 能链接），是每次改动的快速门。
// 需要 iGPU 的数值/整网回归仍由 scripts/*_check.py 承担。
#ifndef INFVINO__TESTS__TEST_UTIL_HPP_
#define INFVINO__TESTS__TEST_UTIL_HPP_

#include <cmath>
#include <cstdio>

namespace itest
{
inline int g_pass = 0;
inline int g_fail = 0;

inline void report(const char * suite)
{
  std::printf("\n%s: %d passed, %d failed\n", suite, g_pass, g_fail);
}
}  // namespace itest

// 单条断言：失败打印文件:行 + 说明，不终止。
#define CHECK(cond, msg)                                                        \
  do {                                                                          \
    if (cond) {                                                                 \
      ++itest::g_pass;                                                          \
    } else {                                                                    \
      ++itest::g_fail;                                                          \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, (msg));              \
    }                                                                           \
  } while (0)

#define CHECK_EQ(a, b, msg) CHECK((a) == (b), msg)
#define CHECK_NEAR(a, b, eps, msg) CHECK(std::fabs((a) - (b)) <= (eps), msg)

// 生成 main（要求前面定义了 static void run_tests()）。
#define ITEST_MAIN(suite)                                                       \
  int main()                                                                    \
  {                                                                             \
    run_tests();                                                                \
    itest::report(suite);                                                       \
    return itest::g_fail ? 1 : 0;                                               \
  }

#endif  // INFVINO__TESTS__TEST_UTIL_HPP_
