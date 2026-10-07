cmake_minimum_required(VERSION 3.16)
project(infvino LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# OpenCL 头文件：显式声明目标版本，并允许使用 1.2 的队列 API（clCreateCommandQueue），
# 避免 cl_version.h 的 pragma message 与 deprecated 警告。
add_compile_definitions(CL_TARGET_OPENCL_VERSION=300)
add_compile_definitions(CL_USE_DEPRECATED_OPENCL_1_2_APIS)

# 依赖：OpenCV（预处理/图像）、OpenCL（自研 kernel）、yaml-cpp（模型配置）。
# 不依赖 ROS / OpenVINO。
find_package(OpenCV REQUIRED)
find_package(OpenCL REQUIRED)
find_package(yaml-cpp REQUIRED)

# .cl 源码目录：默认用源码树里的 kernels/（运行时加载，便于迭代）。
# 安装/部署时可 -DINFVINO_KERNEL_DIR_OVERRIDE=/path/to/kernels 覆盖。
set(INFVINO_KERNEL_DIR_DEFAULT "${CMAKE_CURRENT_SOURCE_DIR}/kernels")
if(INFVINO_KERNEL_DIR_OVERRIDE)
  set(INFVINO_KERNEL_DIR_DEFAULT "${INFVINO_KERNEL_DIR_OVERRIDE}")
endif()

add_library(infvino SHARED
  src/Types.cpp
  src/ModelInfo.cpp
  src/Device.cpp
  src/Preprocess.cpp
  src/Nms.cpp
  src/Decoders.cpp
  src/Half.cpp
  src/ClRuntime.cpp
  src/Tuning.cpp
  src/Autotuner.cpp
  src/KernelFamilies.cpp
  src/LayoutSolver.cpp
  src/L3Model.cpp
  src/L3LineModel.cpp
  src/PlanModel.cpp
  src/ClBackend.cpp
  src/InferenceEngine.cpp
)
target_include_directories(infvino PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_compile_definitions(infvino PUBLIC
  INFVINO_KERNEL_DIR="${INFVINO_KERNEL_DIR_DEFAULT}"
)
target_link_libraries(infvino PUBLIC
  opencv_core
  opencv_imgproc
  opencv_dnn
  OpenCL::OpenCL
  yaml-cpp
)

add_executable(infvino_bench src/tools/infvino_bench.cpp)
target_link_libraries(infvino_bench PRIVATE infvino opencv_imgcodecs)

add_executable(infvino_numtest src/tools/infvino_numtest.cpp)
target_link_libraries(infvino_numtest PRIVATE infvino opencv_imgcodecs)

add_executable(reuse_check src/tools/reuse_check.cpp)
target_link_libraries(reuse_check PRIVATE infvino)

add_executable(kernel_run src/tools/kernel_run.cpp)
target_link_libraries(kernel_run PRIVATE infvino)

add_executable(kernel_bench src/tools/kernel_bench.cpp)
target_link_libraries(kernel_bench PRIVATE infvino)

add_executable(kernel_numtest src/tools/kernel_numtest.cpp)
target_link_libraries(kernel_numtest PRIVATE infvino)

add_executable(kernel_autotune src/tools/kernel_autotune.cpp)
target_link_libraries(kernel_autotune PRIVATE infvino)

# R61: host-only line-granular L3 replacement simulator (probe replication + complexity).
add_executable(l3linesim src/tools/l3linesim.cpp)
target_link_libraries(l3linesim PRIVATE infvino)

add_executable(tuning_test src/tools/tuning_test.cpp)
target_link_libraries(tuning_test PRIVATE infvino)
enable_testing()
add_test(NAME tuning_test COMMAND tuning_test)

# ---------------------------------------------------------------------------
# 离线单元测试（**不需要 GPU / 容器**）：只测 host 侧纯逻辑，是改动的快速门。
# 覆盖布局最小割 / 调优缓存 / 物理标尺 / 算子族注册表——即 R42–R52 里反复出
# bug（静默选型、记账、布局契约）的那一层。整网数值回归仍由 scripts/*_check.py 承担。
#
#   cmake --build build --target check      # 构建并跑全部离线测试
#   ctest --test-dir build --output-on-failure
# ---------------------------------------------------------------------------
set(INFVINO_UNIT_TESTS
  test_layout_solver
  test_tuning_cache
  test_rulers
  test_kernel_families
  test_l3_model
)
foreach(t ${INFVINO_UNIT_TESTS})
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE infvino)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
add_custom_target(check
  COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
  DEPENDS ${INFVINO_UNIT_TESTS} tuning_test
  USES_TERMINAL)

# R43: GPU hardware counters via the Intel Metrics Discovery API (optional).
# Provides independent evidence for "which wall" (EU time, L3/DRAM bytes, stalls),
# cross-checking kernel_bench's -DPROBE attribution. Skipped if libigdmd is absent.
find_library(IGDMD_LIB igdmd)
find_path(IGDMD_INC metrics_discovery_api.h)
if(IGDMD_LIB AND IGDMD_INC)
  add_executable(gpu_metrics src/tools/gpu_metrics.cpp)
  target_include_directories(gpu_metrics PRIVATE ${IGDMD_INC})
  target_link_libraries(gpu_metrics PRIVATE ${IGDMD_LIB})
else()
  message(STATUS "gpu_metrics skipped: libigdmd / metrics_discovery_api.h not found")
endif()

install(TARGETS infvino infvino_bench infvino_numtest reuse_check kernel_run kernel_bench kernel_numtest kernel_autotune l3linesim tuning_test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
if(TARGET gpu_metrics)
  install(TARGETS gpu_metrics RUNTIME DESTINATION bin)
endif()
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY kernels/ DESTINATION share/infvino/kernels)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY config DESTINATION share/infvino)
endif()
