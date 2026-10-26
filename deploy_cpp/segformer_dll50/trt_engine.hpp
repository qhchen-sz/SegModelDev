#pragma once
// TensorRT 引擎的薄封装：加载 .engine、按 batch 设定输入形状、执行推理。
// 只做与引擎相关的事，滑窗/预处理/后处理都在 seg_pipeline 里。
//
// ★ 本文件属于 segformer_dll50（RTX50 / TensorRT 10 版本），内容与
//   ..\segformer_dll\trt_engine.hpp 相同；唯一差异在 trt_engine.cpp 的
//   teardown()：TRT 10 用 delete，TRT 8 用 destroy()（已用 NV_TENSORRT_MAJOR 分支兼容）。
//   非 TRT10 相关的逻辑改动请同时同步回 ..\segformer_dll 与 ..\src。

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity sev, const char* msg) noexcept override {
        if (sev <= Severity::kWARNING) {
            std::fprintf(stderr, "[TRT] %s\n", msg);
        }
    }
};

class TrtEngine {
public:
    TrtEngine() = default;
    ~TrtEngine();

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    bool load(const std::string& enginePath, std::string& err);

    // 打印 I/O 张量名、形状、dtype —— 验证第 1 关就看这个
    void describe() const;

    const std::string& inputName() const { return inName_; }
    const std::string& outputName() const { return outName_; }
    int inputChannels() const { return inC_; }
    int inputHeight() const { return inH_; }
    int inputWidth() const { return inW_; }
    int maxBatch() const { return maxBatch_; }

    // 最近一次 setBatch() 推断出的输出张量形状（如 {16,3,160,160}）
    const std::vector<int>& outputDims() const { return outDims_; }

    // 设定本次推理的 batch，并推算出输出张量的总元素数（用于分配显存）
    bool setBatch(int batch, std::size_t& outElems, std::string& err);

    // 设备内存管理
    void* allocDevice(std::size_t bytes, std::string& err);
    void freeDevice(void* p);

    // 同步推理：dIn/dOut 均为设备指针
    bool infer(void* dIn, void* dOut, cudaStream_t stream, std::string& err);

    static bool cudaCheck(cudaError_t e, const char* what, std::string& err);

private:
    // 幂等释放 runtime/engine/context，并复位 I/O 元信息。析构与重复 load() 共用。
    void teardown();

    TrtLogger logger_;
    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* ctx_ = nullptr;

    std::string inName_;
    std::string outName_;
    int inC_ = 0;
    int inH_ = 0;
    int inW_ = 0;
    int maxBatch_ = 1;
    nvinfer1::Dims outStaticDims_{};   // 引擎侧输出形状（batch 维为 -1，其余已固定）
    std::vector<int> outDims_;         // 最近一次 setBatch() 推断出的完整形状
};
