#pragma once
// 整图滑窗推理管线。
//
// ★ 本文件属于 segformer_dll50（RTX50 / TensorRT 10 版本），已与 ..\segformer_dll
//   及 ..\src 版本分叉：RTX50 上该模型 **batch>1 数值错误**（logits 系统性偏移、
//   整图被判成背景），只有 batch=1 正确，因此这里固定 batch=1，改用 numWorkers 路
//   execution context 并发来换吞吐。修改时不要再与本工程外的副本互相同步。
//
// 单块预处理/上采样/累加的语义仍与 infer_fast_strat.py 逐条对齐
// （mIoU 0.7531 / silie 0.5611 / fanzhe 0.7081）：
//
//   collect_tiles       :28-42   -> collectTiles()
//   预处理              :45-47   -> fillNCHW()
//   推理                :48-52   -> forwardBatch()
//   上采样 bilinear     :51       -> cv::resize(INTER_LINEAR)
//   logits 累加/argmax  :64-69   -> inferImage()
//   叠加图配色          :71-76   -> 见 main.cpp 的 overlay()

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "trt_engine.hpp"

struct SegConfig {
    int stride = 448;      // 滑窗步长；tile 由引擎输入尺寸决定（本项目为 640）
    // RTX50 / TensorRT 10 实测：该模型 batch>1 时 logits 系统性偏移（缺陷类被压低），
    // 只有 batch=1 数值正确。因此本工程固定 batch=1，用多路并发换吞吐。
    int batch = 1;         // 每路前向的 batch（RTX50 必须为 1）
    int numWorkers = 4;    // 并发路数：每路独立 execution context + stream
    int numClasses = 3;    // 0 背景 / 1 silie / 2 fanzhe
};

class SegPipeline {
public:
    ~SegPipeline();

    SegPipeline(const SegPipeline&) = delete;
    SegPipeline& operator=(const SegPipeline&) = delete;
    SegPipeline() = default;

    bool init(TrtEngine& eng, const SegConfig& cfg, std::string& err);

    // 整图滑窗推理，labels 输出为 CV_8UC1、原图尺寸，取值 0/1/2
    bool inferImage(const cv::Mat& bgr, cv::Mat& labels, std::string& err);

    // 调试用：只对左上角单个 tile 前向一次(batch=1)，返回原始 logits
    // (numClasses x outH x outW, NCHW 展平)。用于和 PyTorch 做数值对齐，
    // 把"引擎对不对"与"后处理对不对"解耦。
    bool debugForwardTile(const cv::Mat& bgr, std::vector<float>& logits, std::string& err);

    double lastMs() const { return lastMs_; }
    int tileSize() const { return tile_; }
    int outHeight() const { return outH_; }
    int outWidth() const { return outW_; }

private:
    struct Tile {
        int y0, x0;   // 在原图中的左上角
        int ph, pw;   // 有效区域（padding 之前）的尺寸
    };

    // 每个并发路一套独立资源：context 由 TrtEngine 持有（本类不销毁），
    // stream/device/pinned 缓冲由本类分配与释放。
    struct Worker {
        nvinfer1::IExecutionContext* ctx = nullptr;
        cudaStream_t stream = nullptr;
        void* dIn = nullptr;     // [1,3,tile,tile] float32
        void* dOut = nullptr;    // [1,3,outH,outW] float32
        std::size_t dInBytes = 0;
        std::size_t dOutBytes = 0;
        std::size_t outElems = 0;
        float* hIn = nullptr;    // pinned (cudaHostAlloc)
        float* hOut = nullptr;   // pinned
    };

    // 幂等释放所有 worker 的 stream/device/pinned 缓冲与 upAll_。析构与重复 init() 共用。
    // 注意：不销毁 ctx —— ctx 的生命周期由 TrtEngine（engine.load/teardown）管理。
    void release();
    void releaseWorkers();

    void collectTiles(int H, int W, std::vector<Tile>& out) const;

    TrtEngine* eng_ = nullptr;
    SegConfig cfg_;
    int tile_ = 0;
    int outH_ = 0;
    int outW_ = 0;

    std::vector<Worker> workers_;
    std::vector<float> upAll_;   // [tiles * numClasses * tile * tile] 上采样结果

    double lastMs_ = 0.0;
};

// infer_fast_strat.py:71-76 的叠加图配色（BGR）：
//   背景 (60,60,60) / silie (0,0,255) / fanzhe (255,0,0)，alpha = 0.5
// 注意 Python 对 c=0 也做混合，所以整幅图都会被压暗——必须照搬才能与参考叠加图对齐。
//
// 放在这里而不是 main.cpp，是为了让 seg_infer.exe 在评测时输出的叠加图与
// segformer_Dll_func.dll 实际返回给软件的图**是同一份实现**，
// 否则「用 --eval 验证 DLL 的可视化是否正确」这条验证链就断了。
cv::Mat segOverlay(const cv::Mat& bgr, const cv::Mat& labels, float alpha = 0.5f);
