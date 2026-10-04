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

add_executable(tuning_test src/tools/tuning_test.cpp)
target_link_libraries(tuning_test PRIVATE infvino)

enable_testing()
add_test(NAME tuning_test COMMAND tuning_test)

install(TARGETS infvino infvino_bench infvino_numtest reuse_check kernel_run kernel_bench kernel_numtest kernel_autotune tuning_test
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
install(DIRECTORY kernels/ DESTINATION share/infvino/kernels)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config)
  install(DIRECTORY config DESTINATION share/infvino)
endif()
