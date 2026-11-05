#pragma once
// 整图滑窗推理管线。
//
// 这里复刻 infer_fast_strat.py 的滑窗与后处理，预处理及置信度阈值必须保持一致：
//
//   collect_tiles              -> collectTiles()
//   预处理                     -> fillNCHW()
//   推理与上采样 bilinear      -> forwardBatch() / cv::resize(INTER_LINEAR)
//   平均 logits / Softmax 过滤 -> inferImage()
//   叠加图配色                 -> 见 main.cpp 的 overlay()

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "trt_engine.hpp"

struct SegConfig {
    int stride = 448;      // 滑窗步长；tile 由引擎输入尺寸决定（本项目为 640）
    int batch = 16;        // 单次前向包含多少块
    int numClasses = 3;    // 0 背景 / 1 silie / 2 fanzhe
    float confidenceThreshold = 0.5f;  // Softmax 最大置信度低于此值时判为背景
};

class SegPipeline {
public:
    ~SegPipeline();

    SegPipeline(const SegPipeline&) = delete;
    SegPipeline& operator=(const SegPipeline&) = delete;
    SegPipeline() = default;

    bool init(TrtEngine& eng, const SegConfig& cfg, std::string& err);
    bool setConfidenceThreshold(float threshold, std::string& err);

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

    // 幂等释放 dIn_/dOut_/hIn_/hOut_/stream_/upBuf_/acc_/cov_。析构与重复 init() 共用。
    void release();

    void collectTiles(int H, int W, std::vector<Tile>& out) const;

    TrtEngine* eng_ = nullptr;
    SegConfig cfg_;
    int tile_ = 0;
    int outH_ = 0;
    int outW_ = 0;

    void* dIn_ = nullptr;    // [batch,3,tile,tile] float32
    void* dOut_ = nullptr;   // [batch,3,outH,outW] float32
    std::size_t dInBytes_ = 0;
    std::size_t dOutBytes_ = 0;
    cudaStream_t stream_ = nullptr;

    float* hIn_ = nullptr;       // pinned 主机内存 (cudaHostAlloc)
    float* hOut_ = nullptr;      // pinned 主机内存；pageable 内存会让
                                 // cudaMemcpyAsync 退化成同步拷贝，实测这是大头
    std::vector<float> upBuf_;   // 上采样中转缓冲，供并行 resize 写入
    // [numClasses] CV_32F，原图尺寸。跨帧复用：尺寸不变时 create 是空操作，
    // 只做 setTo(0)，避免每帧重新分配约 3*H*W*4 字节（3000x2600 约 94MB）。
    std::vector<cv::Mat> acc_;
    cv::Mat cov_;  // CV_32F，每个像素的有效滑窗覆盖次数

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

// 同上，但把结果写进调用方提供的 dst 并复用其内存（DLL 每帧取叠加图时用，
// 避免每帧 clone 一份原图）。dst 尺寸/类型不符时自动 create。
void segOverlayInto(const cv::Mat& bgr, const cv::Mat& labels, cv::Mat& dst,
                    float alpha = 0.5f);
