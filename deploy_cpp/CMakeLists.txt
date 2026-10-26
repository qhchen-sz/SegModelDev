cmake_minimum_required(VERSION 3.20)
project(deploy_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- 依赖根目录（本机实测路径，可 -D 覆盖） ----
set(TRT_ROOT "E:/Tensorrt/TensorRT-8.6.1.6" CACHE PATH "TensorRT root")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1" CACHE PATH "CUDA root")
# 名字刻意不叫 *_ROOT，避免和 find_package 的 <Pkg>_ROOT 变量撞车（CMake policy CMP0144）
set(OPENCV_BUILD_DIR "E:/opencv_exe/opencv/build" CACHE PATH "OpenCV prebuilt build dir")

# Windows 头文件里的 min/max 宏会和 std::min/std::max 冲突
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN)

if(MSVC)
  # 源码含中文注释，必须显式声明 UTF-8，否则 MSVC 按 GBK 解析会出 C4819 甚至乱码
  add_compile_options(/utf-8 /W3 /EHsc)
  add_compile_options($<$<CONFIG:Release>:/O2>)
endif()

set(OpenCV_DIR "${OPENCV_BUILD_DIR}" CACHE PATH "OpenCV config dir")
find_package(OpenCV REQUIRED)

if(NOT EXISTS "${TRT_ROOT}/include/NvInfer.h")
  message(FATAL_ERROR "NvInfer.h not found under TRT_ROOT=${TRT_ROOT}")
endif()
if(NOT EXISTS "${CUDA_ROOT}/include/cuda_runtime.h")
  message(FATAL_ERROR "cuda_runtime.h not found under CUDA_ROOT=${CUDA_ROOT}")
endif()

# ---- 两个目标共用的依赖设置 ----
set(SEG_INCLUDES
  "${TRT_ROOT}/include"
  "${CUDA_ROOT}/include"
  ${OpenCV_INCLUDE_DIRS}
)
set(SEG_LINKDIRS
  "${TRT_ROOT}/lib"
  "${CUDA_ROOT}/lib/x64"
)
set(SEG_LIBS nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

# 推理内核：exe 用 src/ 下的源码。
# DLL 用 segformer_dll/ 下的自包含副本（与 VS 工程编译的是同一批文件）——
# 包装层 segformer_dll.cpp 包含同目录头文件，两份头文件必须保持同步。
set(SEG_CORE_SOURCES
  src/trt_engine.cpp
  src/seg_pipeline.cpp
)
set(SEG_DLL_CORE_SOURCES
  segformer_dll/trt_engine.cpp
  segformer_dll/seg_pipeline.cpp
)

# ---- 1) 命令行推理 / 评测程序 ----
add_executable(seg_infer
  src/main.cpp
  ${SEG_CORE_SOURCES}
)
target_include_directories(seg_infer PRIVATE ${SEG_INCLUDES})
target_link_directories(seg_infer PRIVATE ${SEG_LINKDIRS})
target_link_libraries(seg_infer PRIVATE ${SEG_LIBS})

# ---- 2) C ABI 导出 DLL，供 C#（EVDll_Segformer）经 P/Invoke 调用 ----
# 输出名对齐 D:\CaoJiaoJie\gpu_Dll_func 的约定：<name>_Dll_func.dll
add_library(segformer_dll SHARED
  segformer_dll/segformer_dll.cpp
  ${SEG_DLL_CORE_SOURCES}
)
target_include_directories(segformer_dll PRIVATE ${SEG_INCLUDES})
target_link_directories(segformer_dll PRIVATE ${SEG_LINKDIRS})
target_link_libraries(segformer_dll PRIVATE ${SEG_LIBS})
target_compile_definitions(segformer_dll PRIVATE SEGFORMER_DLL_EXPORTS)
set_target_properties(segformer_dll PROPERTIES OUTPUT_NAME "segformer_Dll_func")
