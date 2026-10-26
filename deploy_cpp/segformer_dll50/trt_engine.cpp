// ⚠️ deploy_cpp\src\trt_engine.cpp 的副本，两处请同步改（见 trt_engine.hpp 头部说明）。
#include "trt_engine.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace {

const char* dtypeName(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT: return "float32";
        case nvinfer1::DataType::kHALF:  return "float16";
        case nvinfer1::DataType::kINT32: return "int32";
        case nvinfer1::DataType::kINT8:  return "int8";
        case nvinfer1::DataType::kBOOL:  return "bool";
        default:                         return "?";
    }
}

std::string dimsStr(const nvinfer1::Dims& d) {
    std::string s;
    for (int i = 0; i < d.nbDims; ++i) {
        if (i) s += "x";
        if (d.d[i] < 0) s += "-1";
        else s += std::to_string(d.d[i]);
    }
    return s;
}

}  // namespace

TrtEngine::~TrtEngine() {
    teardown();
}

// 幂等释放。TrtEngine 必须按 context -> engine -> runtime 的顺序销毁。
// 重复 load()（失败后重试 / 运行时热切换模型）也会走到这里，避免旧对象被直接覆盖而泄漏。
//
// ★ RTX50 / TensorRT 10 适配：TRT 10 已删除各接口的 destroy()（8.6 起标记 deprecated，
//   10.0 移除），改用 delete —— 这些接口都带虚析构。保留 #if 分支是为了本文件
//   在 TRT 8/10 两套 SDK 下都能直接编译（本工程实际只会用 TRT 10）。
void TrtEngine::teardown() {
#if NV_TENSORRT_MAJOR >= 10
    delete ctx_;
    ctx_ = nullptr;
    delete engine_;
    engine_ = nullptr;
    delete runtime_;
    runtime_ = nullptr;
#else
    if (ctx_) {
        ctx_->destroy();
        ctx_ = nullptr;
    }
    if (engine_) {
        engine_->destroy();
        engine_ = nullptr;
    }
    if (runtime_) {
        runtime_->destroy();
        runtime_ = nullptr;
    }
#endif
    inName_.clear();
    outName_.clear();
    inC_ = inH_ = inW_ = 0;
    maxBatch_ = 1;
    outStaticDims_ = nvinfer1::Dims{};
    outDims_.clear();
}

bool TrtEngine::cudaCheck(cudaError_t e, const char* what, std::string& err) {
    if (e != cudaSuccess) {
        err = std::string(what) + ": " + cudaGetErrorString(e);
        return false;
    }
    return true;
}

bool TrtEngine::load(const std::string& enginePath, std::string& err) {
    std::ifstream f(enginePath, std::ios::binary);
    if (!f) {
        err = "cannot open engine file: " + enginePath +
              "  (run tools\\build_engine.bat first)";
        return false;
    }
    f.seekg(0, std::ios::end);
    std::streamsize n = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(static_cast<std::size_t>(n));
    if (!f.read(blob.data(), n)) {
        err = "cannot read engine file: " + enginePath;
        return false;
    }
    f.close();

    // 重复 load()（失败后重试 / 运行时热切换模型）先释放上一份 runtime/engine/context。
    // 放在文件成功读入之后：路径写错时仍保留正在使用的旧引擎。
    teardown();

    runtime_ = nvinfer1::createInferRuntime(logger_);
    if (!runtime_) {
        err = "createInferRuntime failed";
        return false;
    }
    engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
    if (!engine_) {
        err = "deserializeCudaEngine failed -- the .engine was built for a different "
              "TensorRT version or GPU. Rebuild with tools\\build_engine.bat";
        return false;
    }
    ctx_ = engine_->createExecutionContext();
    if (!ctx_) {
        err = "createExecutionContext failed";
        return false;
    }

    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* nm = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT) inName_ = nm;
        else outName_ = nm;
    }
    if (inName_.empty() || outName_.empty()) {
        err = "engine has no input/output tensor";
        return false;
    }

    nvinfer1::Dims d = engine_->getTensorShape(inName_.c_str());
    if (d.nbDims != 4) {
        err = "expected a 4D input (NCHW), got " + dimsStr(d);
        return false;
    }
    inC_ = d.d[1];
    inH_ = d.d[2];
    inW_ = d.d[3];
    if (inH_ <= 0 || inW_ <= 0) {
        err = "engine input spatial dims are dynamic (" + dimsStr(d) +
              "); rebuild with fixed 640x640 -- see tools\\build_engine.bat";
        return false;
    }

    nvinfer1::Dims pmax = engine_->getProfileShape(
        inName_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    maxBatch_ = (pmax.nbDims >= 1 && pmax.d[0] > 0) ? pmax.d[0] : 1;

    // 输出的空间维在建引擎时已固定（本项目 160x160），只有 batch 维是 -1。
    // 记下来，setBatch() 就能直接算出输出元素数，不必调 inferShapes()。
    outStaticDims_ = engine_->getTensorShape(outName_.c_str());
    if (outStaticDims_.nbDims < 1) {
        err = "engine output has no dims";
        return false;
    }

    return true;
}

void TrtEngine::describe() const {
    std::printf("  engine I/O:\n");
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
        const char* nm = engine_->getIOTensorName(i);
        bool isIn = engine_->getTensorIOMode(nm) == nvinfer1::TensorIOMode::kINPUT;
        std::printf("    %-5s %-14s dims=%-16s dtype=%s\n", isIn ? "in" : "out", nm,
                    dimsStr(engine_->getTensorShape(nm)).c_str(),
                    dtypeName(engine_->getTensorDataType(nm)));
    }
    if (engine_->getNbIOTensors() > 1) {
        std::printf("    profile(batch): min=%d opt=%d max=%d\n",
                    engine_->getProfileShape(inName_.c_str(), 0,
                                             nvinfer1::OptProfileSelector::kMIN).d[0],
                    engine_->getProfileShape(inName_.c_str(), 0,
                                             nvinfer1::OptProfileSelector::kOPT).d[0],
                    maxBatch_);
    }
}

bool TrtEngine::setBatch(int batch, std::size_t& outElems, std::string& err) {
    if (batch < 1 || batch > maxBatch_) {
        err = "batch " + std::to_string(batch) + " out of engine profile range [1," +
              std::to_string(maxBatch_) + "]";
        return false;
    }
    nvinfer1::Dims4 d{batch, inC_, inH_, inW_};
    if (!ctx_->setInputShape(inName_.c_str(), d)) {
        err = "setInputShape failed for " + dimsStr(d);
        return false;
    }

    // 输出形状 = batch + 引擎侧已固定的其余维。只有 batch 动态，所以直接乘出来，
    // 不需要 inferShapes()（那样每批都要多做一次形状推导）。
    std::size_t n = 1;
    outDims_.clear();
    outDims_.push_back(batch);
    for (int i = 1; i < outStaticDims_.nbDims; ++i) {
        if (outStaticDims_.d[i] <= 0) {
            err = "engine output has dynamic non-batch dims: " + dimsStr(outStaticDims_) +
                  "; this deployment expects fixed spatial output dims";
            return false;
        }
        outDims_.push_back(outStaticDims_.d[i]);
        n *= static_cast<std::size_t>(outStaticDims_.d[i]);
    }
    outElems = n * static_cast<std::size_t>(batch);
    return true;
}

void* TrtEngine::allocDevice(std::size_t bytes, std::string& err) {
    void* p = nullptr;
    if (!cudaCheck(cudaMalloc(&p, bytes), "cudaMalloc", err)) return nullptr;
    return p;
}

void TrtEngine::freeDevice(void* p) {
    if (p) cudaFree(p);
}

bool TrtEngine::infer(void* dIn, void* dOut, cudaStream_t stream, std::string& err) {
    if (!ctx_->setTensorAddress(inName_.c_str(), dIn)) {
        err = "setTensorAddress(input) failed";
        return false;
    }
    if (!ctx_->setTensorAddress(outName_.c_str(), dOut)) {
        err = "setTensorAddress(output) failed";
        return false;
    }
    if (!ctx_->enqueueV3(stream)) {
        err = "enqueueV3 failed";
        return false;
    }
    if (!cudaCheck(cudaStreamSynchronize(stream), "cudaStreamSynchronize", err)) return false;
    return true;
}
