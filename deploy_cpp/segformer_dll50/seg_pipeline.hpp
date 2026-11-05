#pragma once
// 整图滑窗推理管线。
//
// ★ 本文件属于 segformer_dll50（RTX50 / TensorRT 10 版本），已与 ..\segformer_dll
//   及 ..\src 版本分叉：RTX50 上该模型 **batch>1 数值错误**（logits 系统性偏移、
//   整图被判成背景），只有 batch=1 正确，因此这里固定 batch=1，改用 numWorkers 路
//   execution context 并发来换吞吐。修改时不要再与本工程外的副本互相同步。
//
// 单块预处理、滑窗融合及置信度过滤与 infer_fast_strat.py 对齐：
//
//   collect_tiles              -> collectTiles()
//   预处理                     -> fillNCHW()
//   推理与上采样 bilinear      -> forwardBatch() / cv::resize(INTER_LINEAR)
//   平均 logits / Softmax 过滤 -> inferImage()
//   叠加图配色                 -> 见 main.cpp 的 overlay()

#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "seg_profile.h"
#include "trt_engine.hpp"

struct SegConfig {
    int stride = 448;      // 滑窗步长；tile 由引擎输入尺寸决定（本项目为 640）
    // RTX50 / TensorRT 10 实测：该模型 batch>1 时 logits 系统性偏移（缺陷类被压低），
    // 只有 batch=1 数值正确。因此本工程固定 batch=1，用多路并发换吞吐。
    int batch = 1;         // 每路前向的 batch（RTX50 必须为 1）
    int numWorkers = 4;    // 并发路数：每路独立 execution context + stream
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
    // 诊断用只读接口：并发路数、最近一帧的滑窗块数。
    int numWorkers() const { return static_cast<int>(workers_.size()); }
    int lastTileCount() const { return static_cast<int>(lastTiles_); }

    // 最近一帧的结构化耗时（仅在 segtrace 或结构化计时开启时有效）。
    const SegFrameTiming& lastTiming() const { return lastTiming_; }
    void setTotalNs(long long ns) { lastTiming_.totalNs = ns; }
    void setMaskCopyNs(long long ns) { lastTiming_.maskCopyNs = ns; }

    // 用常驻 worker 线程并行执行 [0, rows) 的行任务；无线程时退化为串行。
    // fn 必须在本函数返回前保持有效；只能在持有该 pipeline 的线程调用。
    bool runRows(int rows, const std::function<void(int, int)>& fn, std::string& err);

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

        // 结构化计时：事件按 stream 顺序记录，sync 之后 elapsed 得到纯 GPU 时间。
        cudaEvent_t ev[4] = {nullptr, nullptr, nullptr, nullptr};
        // 以下由 worker 线程写、主线程在 doneCv_ 之后读（有 mutex 建立可见性）。
        long long pWakeNs = 0;
        long long pPrepNs = 0;
        long long pGpuWallNs = 0;
        long long pInferEnqueueNs = 0;
        long long pSyncNs = 0;
        long long pH2dNs = 0;
        long long pInferNs = 0;
        long long pD2hNs = 0;
        long long pUpNs = 0;
        long long pDoneNs = 0;
        long long pLastEndNs = 0;
        int pTiles = 0;
    };

    // 幂等释放所有 worker 的 stream/device/pinned 缓冲与 upAll_/cov_。析构与重复 init() 共用。
    // 注意：不销毁 ctx —— ctx 的生命周期由 TrtEngine（engine.load/teardown）管理。
    void release();
    void releaseWorkers();

    // 持久线程池：init 成功后常驻，inferImage 只做分发/等待，不再每帧创建线程。
    void startPool();
    void stopPool();
    void runWorker(Worker& w, int idx);
    void runRowsWorker();

    void collectTiles(int H, int W, std::vector<Tile>& out) const;

    TrtEngine* eng_ = nullptr;
    SegConfig cfg_;
    int tile_ = 0;
    int outH_ = 0;
    int outW_ = 0;

    std::vector<Worker> workers_;
    std::size_t lastTiles_ = 0;  // 最近一帧的滑窗块数（仅诊断用）
    std::vector<float> upAll_;   // [tiles * numClasses * tile * tile] 上采样结果
    // [numClasses] CV_32F，原图尺寸。每帧复用：尺寸不变时 create 是空操作，
    // 只做 setTo(0)，避免每帧重新分配约 3*H*W*4 字节（3000x2600 约 94MB）。
    std::vector<cv::Mat> acc_;
    cv::Mat cov_;  // CV_32F，每个像素的有效滑窗覆盖次数

    // ---- 持久线程池 ----
    // 关键点：worker 线程在 init() 时创建、常驻到 release()。每帧只递增 poolSeq_
    // 并 notify_all，避免「每帧新建 4 个线程 + 动态队列」导致某些帧只有 1~2 个
    // 线程真正并行（表现为 300ms~1100ms 的随机波动）。
    std::vector<std::thread> threads_;
    std::mutex poolMtx_;
    std::condition_variable poolCv_;    // 主线程 -> worker：新帧
    std::condition_variable doneCv_;    // worker -> 主线程：本帧结束
    bool poolStop_ = false;
    std::size_t poolSeq_ = 0;           // 帧序号，worker 用它判断是否被唤醒
    int done_ = 0;                      // 本帧已完成的 worker 数
    // 每帧共享状态：主线程在 notify 前写，worker 在 poolMtx_ 之后读。
    const cv::Mat* frameBgr_ = nullptr;
    const std::vector<Tile>* frameTiles_ = nullptr;
    std::atomic<std::size_t> frameNext_{0};
    std::atomic<bool> frameFailed_{false};
    std::mutex errMtx_;
    std::string frameErr_;

    // 池任务的第二种形态：按行动态分块（Softmax 后处理 / overlay 复用同一批常驻线程）。
    // 不再每帧 new std::thread —— 在 EasyVision 这类加载大量 DLL 的进程里，
    // CreateThread 的 DLL_THREAD_ATTACH 会带来数百毫秒的抖动。
    enum class PoolTask { Tiles, Rows };
    PoolTask poolTask_ = PoolTask::Tiles;
    const std::function<void(int, int)>* rowFn_ = nullptr;  // 指向调用方栈上的 functor
    int rowCount_ = 0;
    int rowChunk_ = 1;
    std::atomic<int> rowNext_{0};

    // OpenCV 全局线程数：init 时压到 1（worker 内 resize 单线程），release 时恢复。
    // 不再每帧 setNumThreads，避免线程池反复销毁/重建。
    int cvThreadsOld_ = 0;
    bool cvThreadsSaved_ = false;

    // ---- 诊断（SEGFORMER_PROFILE=1 或 SegProfileConfigure，默认关闭）----
    bool prof_ = false;
    std::atomic<long long> profPrepNs_{0};
    std::atomic<long long> profGpuNs_{0};
    std::atomic<long long> profUpNs_{0};
    std::atomic<long long> profGpuMinNs_{0};
    std::atomic<long long> profGpuMaxNs_{0};
    std::atomic<int> profBlocks_[16]{};       // 每个 worker 处理的块数（最多 16 路）
    std::vector<long long> profGpuPerBlock_;  // 每块 GPU 时间，用于算中位数

    // ---- 结构化快照（C# 通过 SegGetLastTiming 读取）----
    SegFrameTiming lastTiming_{};
    long long timingSeq_ = 0;
    std::chrono::steady_clock::time_point frameStartTp_{};
    std::chrono::steady_clock::time_point frameNotifyTp_{};

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
                    float alpha = 0.5f, SegPipeline* pool = nullptr);

// ============================================================================
// 诊断日志：只在 SEGFORMER_PROFILE=1 时启用，默认关闭且不改变推理路径。
//
// 目的：把每帧拆成可对齐的段（清零 / 并行推理 / 累加 / Softmax 后处理 / 统计 / 取图），
// 用于定位耗时波动来自哪一段。
//
// 输出 CSV，每行一个事件：
//   frame,wall,t_ms,event,ms,note
// 文件：SEGFORMER_PROFILE_FILE 指定的路径；未指定时写到
//   %TEMP%\seg_trace_<时间戳>_<pid>.csv，并把路径打印到 stderr。
//
// 关闭时所有接口只做一次原子判断，不取时间戳、不打开文件。
// ============================================================================
namespace segtrace {

using Clock = std::chrono::steady_clock;

// 是否启用（首次调用时读环境变量并打开文件，线程安全）。
bool enabled();

// 结构化快照开关：C# 调用 SegProfileConfigure 打开。
// 与 CSV 日志（enabled）相互独立；任一开启都会填充 SegPipeline::lastTiming()。
bool timingEnabled();
void setTimingEnabled(bool on);

// 一帧的开始/结束：分配帧号、记录帧起始时刻；结束时 flush 一次。
// SegForward / SegForwardMask 入口会自动调用，不需要外部配合。
void frameBegin();
void frameEnd();

// 立即把缓冲区落盘。取 mask / 统计这类帧外事件结束后调用，避免数据滞留。
void flush();

// 写一行元信息（引擎、GPU、workers 等），frame 列写 0。
void meta(const char* note);

// 记录一个已完成的耗时事件。
void event(const char* name, double ms, const char* note);

// RAII 计时器：构造开始计时，析构时写事件；note 可在析构前用 setNote 补全。
class Scope {
public:
    explicit Scope(const char* name, const char* note = nullptr);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

    void setNote(const std::string& note);

private:
    const char* name_;
    std::string note_;
    Clock::time_point t0_;
    bool on_;
};

// 一帧的作用域：构造时 frameBegin()，析构时 frameEnd()。
// 放在导出函数入口即可保证所有 return 路径都会把本帧日志落盘。
class Frame {
public:
    Frame();
    ~Frame();
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
};

// 作用域结束时 flush，用于帧外事件（SegGetMask / SegGetStat / SegGetOverlay）。
class FlushOnExit {
public:
    ~FlushOnExit() { flush(); }
};

}  // namespace segtrace
