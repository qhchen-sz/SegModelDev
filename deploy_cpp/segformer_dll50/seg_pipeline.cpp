// ★ segformer_dll50（RTX50 / TensorRT 10）专用：固定 batch=1 + 多路 context 并发。
//   与 ..\segformer_dll 及 ..\src 的版本已分叉，不要再互相同步（见 seg_pipeline.hpp 说明）。

// 必须先于任何可能包含 windows.h 的头文件定义，否则 min/max 宏会展开 std::min/std::max。
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include "seg_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cwchar>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

// 与 infer_fast_strat.py:14-15 完全一致
const float kMean[3] = {0.485f, 0.456f, 0.406f};
const float kStd[3] = {0.229f, 0.224f, 0.225f};

// BGR CV_8UC3 (tile x tile) -> NCHW float32 + ImageNet 归一化
// 对应 Python: p[:, :, ::-1].astype(float32)/255 -> (a-mean)/std -> permute(0,3,1,2)
void fillNCHW(const cv::Mat& bgr, float* dst, int tile) {
    const int HW = tile * tile;
    for (int y = 0; y < tile; ++y) {
        const uchar* row = bgr.ptr<uchar>(y);
        float* d0 = dst + 0 * HW + y * tile;
        float* d1 = dst + 1 * HW + y * tile;
        float* d2 = dst + 2 * HW + y * tile;
        for (int x = 0; x < tile; ++x) {
            const float r = row[x * 3 + 2] / 255.0f;   // BGR -> RGB
            const float g = row[x * 3 + 1] / 255.0f;
            const float b = row[x * 3 + 0] / 255.0f;
            d0[x] = (r - kMean[0]) / kStd[0];
            d1[x] = (g - kMean[1]) / kStd[1];
            d2[x] = (b - kMean[2]) / kStd[2];
        }
    }
}

// 结构化计时的公共时钟；与 segtrace::Clock 同一 clock，但这里只用于快照。
using ProfClock = std::chrono::steady_clock;

inline long long nsBetween(ProfClock::time_point a, ProfClock::time_point b) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(b - a).count();
}

}  // namespace

// ============================================================================
// segtrace 实现（声明见 seg_pipeline.hpp）
// ============================================================================
namespace segtrace {
namespace {

std::mutex g_mtx;
std::FILE* g_file = nullptr;
std::atomic<int> g_state{0};   // 0=未初始化 1=关闭 2=开启
std::atomic<bool> g_timing{false};  // 结构化快照开关（C# 通过 SegProfileConfigure 打开）
long long g_frame = 0;
Clock::time_point g_frameT0{};
std::string g_wall;

std::string wallNow() {
#ifdef _WIN32
    SYSTEMTIME st;
    GetLocalTime(&st);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d.%03d", st.wHour, st.wMinute,
                  st.wSecond, st.wMilliseconds);
    return std::string(buf);
#else
    return std::string();
#endif
}

// 进程正常退出时关闭 trace 文件；强烈建议关闭诊断时不要设置 SEGFORMER_PROFILE。
void closeTraceFile() {
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_file) {
        std::fflush(g_file);
        std::fclose(g_file);
        g_file = nullptr;
    }
}

// 打开日志文件。调用方必须已持有 g_mtx。
bool openFile() {
    std::FILE* f = nullptr;
#ifdef _WIN32
    std::wstring path;
    const char* envFile = std::getenv("SEGFORMER_PROFILE_FILE");
    if (envFile && *envFile) {
        // cmd 的 set 写入的是系统 ANSI 代码页，按 CP_ACP 转换
        const int n = MultiByteToWideChar(CP_ACP, 0, envFile, -1, nullptr, 0);
        if (n > 0) {
            std::wstring w(static_cast<std::size_t>(n), L'\0');
            MultiByteToWideChar(CP_ACP, 0, envFile, -1, &w[0], n);
            if (!w.empty() && w.back() == L'\0') w.pop_back();
            path.swap(w);
        }
    }
    if (path.empty()) {
        wchar_t tmp[MAX_PATH] = {0};
        const DWORD n = GetTempPathW(MAX_PATH, tmp);
        SYSTEMTIME st;
        GetLocalTime(&st);
        wchar_t name[128];
        std::swprintf(name, 128, L"seg_trace_%04d%02d%02d_%02d%02d%02d_%lu.csv",
                      st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond,
                      static_cast<unsigned long>(GetCurrentProcessId()));
        if (n > 0 && n < MAX_PATH) path.assign(tmp, n);
        path += name;
    }
    f = _wfopen(path.c_str(), L"a");
    if (f) {
        char ansi[1024] = {0};
        if (WideCharToMultiByte(CP_ACP, 0, path.c_str(), -1, ansi,
                                static_cast<int>(sizeof(ansi)), nullptr, nullptr) > 0) {
            std::fprintf(stderr, "[segtrace] %s\n", ansi);
        }
    }
#else
    const char* envFile = std::getenv("SEGFORMER_PROFILE_FILE");
    const char* p = (envFile && *envFile) ? envFile : "/tmp/seg_trace.csv";
    f = std::fopen(p, "a");
    if (f) std::fprintf(stderr, "[segtrace] %s\n", p);
#endif
    if (!f) {
        std::fprintf(stderr, "[segtrace] 无法打开日志文件，日志已关闭\n");
        return false;
    }
    std::setvbuf(f, nullptr, _IOFBF, 1 << 16);
    std::fseek(f, 0, SEEK_END);
    if (std::ftell(f) <= 0) {
        std::fprintf(f, "frame,wall,t_ms,event,ms,note\n");
    }
    g_file = f;
    g_wall = wallNow();
    std::atexit(closeTraceFile);
    return true;
}

}  // namespace

bool enabled() {
    int s = g_state.load(std::memory_order_acquire);
    if (s != 0) return s == 2;
    std::lock_guard<std::mutex> lk(g_mtx);
    s = g_state.load(std::memory_order_relaxed);
    if (s == 0) {
        const char* env = std::getenv("SEGFORMER_PROFILE");
        const bool want = env && *env && std::strcmp(env, "0") != 0;
        const bool ok = want && openFile();
        s = ok ? 2 : 1;
        g_state.store(s, std::memory_order_release);
    }
    return s == 2;
}

bool timingEnabled() {
    return g_timing.load(std::memory_order_relaxed);
}

void setTimingEnabled(bool on) {
    g_timing.store(on, std::memory_order_relaxed);
}

void frameBegin() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    ++g_frame;
    g_frameT0 = Clock::now();
    g_wall = wallNow();
}

void frameEnd() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    // 每帧 fflush 会放大 trace 自身的开销；取图/取统计后的 FlushOnExit 仍会及时落盘。
    if (g_file && (g_frame % 32) == 0) std::fflush(g_file);
}

void flush() {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (g_file) std::fflush(g_file);
}

void meta(const char* note) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_file) return;
    if (g_wall.empty()) g_wall = wallNow();
    std::fprintf(g_file, "%lld,%s,0.000,meta,0.0000,\"%s\"\n", g_frame, g_wall.c_str(),
                 note ? note : "");
    std::fflush(g_file);
}

void event(const char* name, double ms, const char* note) {
    if (!enabled()) return;
    std::lock_guard<std::mutex> lk(g_mtx);
    if (!g_file) return;
    // 帧外事件（GetMask / GetStat）之前若没有 frameBegin，补一个起始时刻，
    // 避免 t_ms 变成自 epoch 起的巨大值。
    if (g_frameT0 == Clock::time_point{}) {
        g_frameT0 = Clock::now();
        if (g_wall.empty()) g_wall = wallNow();
    }
    const double t =
        std::chrono::duration<double, std::milli>(Clock::now() - g_frameT0).count();
    std::fprintf(g_file, "%lld,%s,%.3f,%s,%.4f,\"%s\"\n", g_frame, g_wall.c_str(), t,
                 name ? name : "?", ms, note ? note : "");
}

Scope::Scope(const char* name, const char* note)
    : name_(name), note_(note ? note : ""), t0_(), on_(enabled()) {
    if (on_) t0_ = Clock::now();
}

Scope::~Scope() {
    if (!on_) return;
    const double ms =
        std::chrono::duration<double, std::milli>(Clock::now() - t0_).count();
    event(name_, ms, note_.c_str());
}

void Scope::setNote(const std::string& note) {
    if (on_) note_ = note;
}

Frame::Frame() { frameBegin(); }
Frame::~Frame() { frameEnd(); }

}  // namespace segtrace

void segOverlayInto(const cv::Mat& bgr, const cv::Mat& labels, cv::Mat& dst, float alpha,
                    SegPipeline* pool) {
    const cv::Vec3f cols[3] = {cv::Vec3f(60, 60, 60), cv::Vec3f(0, 0, 255),
                               cv::Vec3f(255, 0, 0)};
    // copyTo 在 dst 尺寸/类型已匹配时直接复用其内存，不会每帧重新分配。
    bgr.copyTo(dst);
    // 逐行独立，行并行不改变任何输出像素（叠加图在 DLL 的每帧路径上，值得并行）
    auto body = [&](int y0, int y1) {
        for (int y = y0; y < y1; ++y) {
            const uchar* lab = labels.ptr<uchar>(y);
            cv::Vec3b* row = dst.ptr<cv::Vec3b>(y);
            for (int x = 0; x < dst.cols; ++x) {
                const cv::Vec3f& c = cols[std::min<int>(lab[x], 2)];
                for (int k = 0; k < 3; ++k) {
                    row[x][k] = static_cast<uchar>(row[x][k] * (1.0f - alpha) + c[k] * alpha);
                }
            }
        }
    };

    if (pool && pool->numWorkers() > 0) {
        std::string e;
        if (!pool->runRows(dst.rows, body, e)) {
            throw std::runtime_error(e);
        }
    } else {
        body(0, dst.rows);
    }
}

cv::Mat segOverlay(const cv::Mat& bgr, const cv::Mat& labels, float alpha) {
    cv::Mat ov;
    segOverlayInto(bgr, labels, ov, alpha);
    return ov;
}

SegPipeline::~SegPipeline() {
    release();
}

// 幂等释放。重复 init()（SegLoad 重试 / 热切换模型）时若直接覆盖这些成员，
// 每轮会漏掉显存、pinned 主机内存与 stream。
//
// 注意：execution context 不在这里销毁 —— 它由 TrtEngine 统一持有，并在
// engine.load()/teardown()/析构时销毁。这里只释放 stream/device/pinned 缓冲。
void SegPipeline::releaseWorkers() {
    // 必须先让常驻线程退出，再释放它们正在使用的 stream/device/pinned 缓冲。
    stopPool();
    for (auto& w : workers_) {
        for (auto& e : w.ev) {
            if (e) {
                cudaEventDestroy(e);
                e = nullptr;
            }
        }
        // 显存是 cudaMalloc 在本进程 CUDA 上下文里分配的，cudaFree 不依赖 TrtEngine 对象。
        if (w.dIn) cudaFree(w.dIn);
        if (w.dOut) cudaFree(w.dOut);
        if (w.hIn) cudaFreeHost(w.hIn);
        if (w.hOut) cudaFreeHost(w.hOut);
        if (w.stream) cudaStreamDestroy(w.stream);
        w = Worker{};
    }
    workers_.clear();
}

void SegPipeline::release() {
    releaseWorkers();
    std::vector<float>().swap(upAll_);   // 真正归还上采样中转缓冲
    std::vector<cv::Mat>().swap(acc_);   // 归还累加缓冲（cv::Mat 析构释放像素内存）
    // 恢复 OpenCV 全局线程数（init 时压到 1，见 startPool 后那段）
    if (cvThreadsSaved_) {
        cv::setNumThreads(cvThreadsOld_);
        cvThreadsSaved_ = false;
    }
}

// 启动常驻 worker 线程。必须在 workers_ 与 ctx 都创建成功后调用。
// 线程创建失败时清理已创建的线程并把异常抛给 init()，由 init() 降级重试。
void SegPipeline::startPool() {
    stopPool();   // 幂等：先清掉上一轮

    const std::size_t nWorkers = workers_.size();
    if (nWorkers == 0) return;

    {
        std::lock_guard<std::mutex> lk(poolMtx_);
        poolStop_ = false;
        poolSeq_ = 0;
        done_ = 0;
    }

    // 可选：把 worker 绑到 P 核。i9-12900K 这类混合架构在 Win10 上可能把新线程
    // 调度到 E 核，单块耗时会明显变长；默认关闭，避免在其它 CPU 上绑错核。
    static const bool kAffinity = (std::getenv("SEGFORMER_AFFINITY") != nullptr);

    threads_.reserve(nWorkers);
    try {
        for (std::size_t i = 0; i < nWorkers; ++i) {
            threads_.emplace_back([this, i, nWorkers] {
#ifdef _WIN32
                if (kAffinity) {
                    const int nCpu = static_cast<int>(std::thread::hardware_concurrency());
                    const int nBind = std::min(16, nCpu);
                    if (nBind > 0) {
                        const DWORD_PTR mask =
                            (nBind >= static_cast<int>(sizeof(DWORD_PTR) * 8))
                                ? ~static_cast<DWORD_PTR>(0)
                                : ((static_cast<DWORD_PTR>(1) << nBind) - 1);
                        SetThreadAffinityMask(GetCurrentThread(), mask);
                    }
                }
#endif
                std::size_t localSeq = 0;
                for (;;) {
                    std::unique_lock<std::mutex> lk(poolMtx_);
                    poolCv_.wait(lk, [&] { return poolStop_ || poolSeq_ != localSeq; });
                    if (poolStop_) return;
                    localSeq = poolSeq_;
                    lk.unlock();

                    if (poolTask_ == PoolTask::Rows) {
                        runRowsWorker();
                    } else {
                        runWorker(workers_[i], static_cast<int>(i));
                    }

                    lk.lock();
                    if (++done_ == static_cast<int>(nWorkers)) doneCv_.notify_one();
                }
            });
        }
    } catch (...) {
        stopPool();
        throw;
    }
}

void SegPipeline::stopPool() {
    {
        std::lock_guard<std::mutex> lk(poolMtx_);
        if (threads_.empty()) {
            poolStop_ = false;
            return;
        }
        poolStop_ = true;
    }
    poolCv_.notify_all();
    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }
    threads_.clear();
    {
        std::lock_guard<std::mutex> lk(poolMtx_);
        poolStop_ = false;
        done_ = 0;
    }
}

bool SegPipeline::init(TrtEngine& eng, const SegConfig& cfg, std::string& err) {
    // 旧资源必须经旧的 eng_ 释放，因此在重新赋值之前调用
    release();
    eng_ = &eng;
    cfg_ = cfg;

    tile_ = eng.inputHeight();
    if (eng.inputWidth() != tile_) {
        err = "engine input is not square";
        return false;
    }
    if (eng.inputChannels() != 3) {
        err = "engine input does not have 3 channels";
        return false;
    }
    // RTX50 / TensorRT 10：该模型 batch>1 数值错误，强制每路 1 块。
    // （并发通过 numWorkers 提升，而不是 batch。）
    cfg_.batch = 1;
    // profBlocks_ 是固定 16 槽的数组，这里必须夹住上限；SEGFORMER_WORKERS 也已限制 1~16。
    const int nWant = std::max(1, std::min(16, cfg_.numWorkers));

    // 上一轮由本管线创建的 context 如果还在（正常情况下 engine.load 已清空），先销毁，
    // 避免重复 init 时 context 累积占用显存。
    eng.destroyContexts();

    // 从 nWant 递减尝试：显存/context 不足时自动降级到更少路数，而不是启动失败。
    for (int n = nWant; n >= 1; --n) {
        std::size_t outElems = 0;
        int oH = 0, oW = 0;
        bool ok = true;

        workers_.clear();
        workers_.resize(static_cast<std::size_t>(n));

        for (int i = 0; i < n && ok; ++i) {
            Worker& w = workers_[i];
            w.ctx = eng.createContext();
            if (!w.ctx) {
                err = "createExecutionContext failed";
                ok = false;
                break;
            }
            if (!eng.setBatchOn(w.ctx, cfg_.batch, w.outElems, err)) {
                ok = false;
                break;
            }
            if (i == 0) {
                const std::vector<int>& od = eng.outputDims();
                if (od.size() != 4 || od[1] != cfg_.numClasses) {
                    err = "unexpected engine output dims";
                    ok = false;
                    break;
                }
                oH = od[2];
                oW = od[3];
                outElems = w.outElems;
            }

            w.dInBytes = static_cast<std::size_t>(cfg_.batch) * 3 * tile_ * tile_ * sizeof(float);
            w.dOutBytes = w.outElems * sizeof(float);

            w.dIn = eng.allocDevice(w.dInBytes, err);
            if (!w.dIn) {
                ok = false;
                break;
            }
            w.dOut = eng.allocDevice(w.dOutBytes, err);
            if (!w.dOut) {
                ok = false;
                break;
            }
            if (!TrtEngine::cudaCheck(cudaStreamCreate(&w.stream), "cudaStreamCreate", err)) {
                ok = false;
                break;
            }
            // pinned 主机内存：否则 cudaMemcpyAsync 对 pageable 内存会退化成同步拷贝
            if (!TrtEngine::cudaCheck(
                    cudaHostAlloc(reinterpret_cast<void**>(&w.hIn), w.dInBytes,
                                  cudaHostAllocDefault),
                    "cudaHostAlloc(input)", err)) {
                ok = false;
                break;
            }
            if (!TrtEngine::cudaCheck(
                    cudaHostAlloc(reinterpret_cast<void**>(&w.hOut), w.dOutBytes,
                                  cudaHostAllocDefault),
                    "cudaHostAlloc(output)", err)) {
                ok = false;
                break;
            }
        }

        if (ok) {
            try {
                startPool();
            } catch (const std::exception& e) {
                err = std::string("thread pool: ") + e.what();
                ok = false;
            }
        }
        if (ok) {
            outH_ = oH;
            outW_ = oW;
            (void)outElems;
            // 默认关闭；只有显式设置 SEGFORMER_PROFILE 为非空且非 "0" 时才开启。
            const char* profEnv = std::getenv("SEGFORMER_PROFILE");
            prof_ = profEnv && *profEnv && std::strcmp(profEnv, "0") != 0;

            // worker 内的 cv::resize/copyMakeBorder 不需要也不应该再开 OpenCV 线程池：
            // 全局压到 1，overlay/argmax 复用常驻 worker 线程（runRows）。只在这里设置一次，
            // release() 时恢复，避免每帧销毁/重建 OpenCV 线程池。
            if (!cvThreadsSaved_) {
                cvThreadsOld_ = cv::getNumThreads();
                cv::setNumThreads(1);
                cvThreadsSaved_ = true;
            }

            if (n < nWant) {
                std::fprintf(stderr,
                             "[SegPipeline] 资源不足，并发路数由 %d 降级为 %d\n", nWant, n);
            }
            return true;
        }

        // 本轮失败：释放已分配的 stream/缓冲，并销毁本轮创建的 context，然后降级重试。
        releaseWorkers();
        eng.destroyContexts();
    }

    if (err.empty()) err = "pipeline init failed";
    return false;
}

// 复刻 infer_fast_strat.py:28-42 的 collect_tiles
void SegPipeline::collectTiles(int H, int W, std::vector<Tile>& out) const {
    std::vector<int> ys, xs;

    const int stopY = std::max(1, H - tile_ + 1);
    for (int y = 0; y < stopY; y += cfg_.stride) ys.push_back(y);
    if (ys.back() + tile_ < H) ys.push_back(std::max(0, H - tile_));

    const int stopX = std::max(1, W - tile_ + 1);
    for (int x = 0; x < stopX; x += cfg_.stride) xs.push_back(x);
    if (xs.back() + tile_ < W) xs.push_back(std::max(0, W - tile_));

    out.clear();
    out.reserve(ys.size() * xs.size());
    for (int y0 : ys) {
        for (int x0 : xs) {
            Tile t;
            t.y0 = y0;
            t.x0 = x0;
            t.ph = std::min(tile_, H - y0);
            t.pw = std::min(tile_, W - x0);
            out.push_back(t);
        }
    }
}

// 常驻 worker 的执行体：从本帧共享队列取块，做「预处理 -> H2D -> 推理 -> D2H -> 上采样」。
// 由 startPool() 启动的线程反复调用；异常在此消化并置 frameFailed_，不向线程函数外抛。
void SegPipeline::runWorker(Worker& w, int idx) {
    try {
        const std::vector<Tile>& tiles = *frameTiles_;
        const cv::Mat& bgr = *frameBgr_;
        const int tileHW = tile_ * tile_;
        const int planeHW = outH_ * outW_;
        const std::size_t upStride =
            static_cast<std::size_t>(cfg_.numClasses) * static_cast<std::size_t>(tileHW);

        const bool sProf = segtrace::enabled() || segtrace::timingEnabled();
        const bool needT = prof_ || sProf;
        const ProfClock::time_point tFrameStart = frameStartTp_;
        if (sProf) {
            w.pWakeNs = 0;
            w.pPrepNs = 0;
            w.pGpuWallNs = 0;
            w.pInferEnqueueNs = 0;
            w.pSyncNs = 0;
            w.pH2dNs = 0;
            w.pInferNs = 0;
            w.pD2hNs = 0;
            w.pUpNs = 0;
            w.pDoneNs = 0;
            w.pLastEndNs = 0;
            w.pTiles = 0;
            if (!w.ev[0]) {
                for (int k = 0; k < 4; ++k) cudaEventCreate(&w.ev[k]);
            }
        }

        for (;;) {
            const std::size_t j = frameNext_.fetch_add(1);
            if (j >= tiles.size() || frameFailed_.load()) break;

            if (sProf && w.pTiles == 0) {
                w.pWakeNs = nsBetween(frameNotifyTp_, ProfClock::now());
            }
            if (sProf) ++w.pTiles;

            auto tStart = std::chrono::steady_clock::time_point{};
            if (needT) tStart = ProfClock::now();

            const Tile& t = tiles[j];
            cv::Mat patch = bgr(cv::Rect(t.x0, t.y0, t.pw, t.ph));
            if (t.ph != tile_ || t.pw != tile_) {
                cv::Mat padded;
                // cv2.BORDER_REFLECT (不是默认的 BORDER_REFLECT_101)
                cv::copyMakeBorder(patch, padded, 0, tile_ - t.ph, 0, tile_ - t.pw,
                                   cv::BORDER_REFLECT);
                patch = padded;
            }
            fillNCHW(patch, w.hIn, tile_);

            auto tPrep = tStart;
            if (needT) tPrep = ProfClock::now();

            std::string e;
            if (sProf && w.ev[0]) cudaEventRecord(w.ev[0], w.stream);
            bool ok = TrtEngine::cudaCheck(
                cudaMemcpyAsync(w.dIn, w.hIn, w.dInBytes, cudaMemcpyHostToDevice, w.stream),
                "cudaMemcpyAsync(H2D)", e);
            if (sProf && w.ev[1]) cudaEventRecord(w.ev[1], w.stream);

            const ProfClock::time_point tEnq0 = sProf ? ProfClock::now() : ProfClock::time_point{};
            if (ok) ok = eng_->inferOn(w.ctx, w.dIn, w.dOut, w.stream, e);
            const ProfClock::time_point tEnq1 = sProf ? ProfClock::now() : ProfClock::time_point{};
            if (sProf && w.ev[2]) cudaEventRecord(w.ev[2], w.stream);

            if (ok) {
                ok = TrtEngine::cudaCheck(
                    cudaMemcpyAsync(w.hOut, w.dOut, w.dOutBytes, cudaMemcpyDeviceToHost,
                                    w.stream),
                    "cudaMemcpyAsync(D2H)", e);
            }
            if (sProf && w.ev[3]) cudaEventRecord(w.ev[3], w.stream);

            const ProfClock::time_point tSync0 = sProf ? ProfClock::now() : ProfClock::time_point{};
            if (ok) ok = TrtEngine::cudaCheck(cudaStreamSynchronize(w.stream),
                                              "cudaStreamSynchronize", e);
            const ProfClock::time_point tSync1 = sProf ? ProfClock::now() : ProfClock::time_point{};

            if (!ok) {
                std::lock_guard<std::mutex> lk(errMtx_);
                if (frameErr_.empty()) frameErr_ = e;
                frameFailed_ = true;
                return;
            }

            auto tGpu = tPrep;
            if (needT) tGpu = tSync1;

            if (sProf) {
                float e01 = 0.0f, e12 = 0.0f, e23 = 0.0f;
                if (w.ev[0] && w.ev[1]) cudaEventElapsedTime(&e01, w.ev[0], w.ev[1]);
                if (w.ev[1] && w.ev[2]) cudaEventElapsedTime(&e12, w.ev[1], w.ev[2]);
                if (w.ev[2] && w.ev[3]) cudaEventElapsedTime(&e23, w.ev[2], w.ev[3]);
                w.pH2dNs += static_cast<long long>(e01 * 1e6);
                w.pInferNs += static_cast<long long>(e12 * 1e6);
                w.pD2hNs += static_cast<long long>(e23 * 1e6);
                w.pInferEnqueueNs += nsBetween(tEnq0, tEnq1);
                w.pSyncNs += nsBetween(tSync0, tSync1);
                w.pGpuWallNs += nsBetween(tPrep, tSync1);
            }

            float* dst = upAll_.data() + j * upStride;
            for (int c = 0; c < cfg_.numClasses; ++c) {
                cv::Mat src(outH_, outW_, CV_32F,
                            w.hOut + static_cast<std::size_t>(c) * planeHW);
                cv::Mat d(tile_, tile_, CV_32F, dst + static_cast<std::size_t>(c) * tileHW);
                // 与 F.interpolate(..., bilinear, align_corners=False) 同为半像素中心约定
                cv::resize(src, d, cv::Size(tile_, tile_), 0, 0, cv::INTER_LINEAR);
            }

            const auto tUp = needT ? ProfClock::now() : ProfClock::time_point{};
            if (sProf) {
                w.pUpNs += nsBetween(tSync1, tUp);
                w.pPrepNs += nsBetween(tStart, tPrep);
                w.pLastEndNs = nsBetween(tFrameStart, tUp);
            }

            if (prof_) {
                const long long prepNs = nsBetween(tStart, tPrep);
                const long long gpuNs = nsBetween(tPrep, tGpu);
                const long long upNs = nsBetween(tGpu, tUp);
                profPrepNs_.fetch_add(prepNs, std::memory_order_relaxed);
                profGpuNs_.fetch_add(gpuNs, std::memory_order_relaxed);
                profUpNs_.fetch_add(upNs, std::memory_order_relaxed);
                profGpuPerBlock_[j] = gpuNs;
                long long curMin = profGpuMinNs_.load(std::memory_order_relaxed);
                while (gpuNs < curMin &&
                       !profGpuMinNs_.compare_exchange_weak(curMin, gpuNs,
                                                            std::memory_order_relaxed)) {
                }
                long long curMax = profGpuMaxNs_.load(std::memory_order_relaxed);
                while (gpuNs > curMax &&
                       !profGpuMaxNs_.compare_exchange_weak(curMax, gpuNs,
                                                            std::memory_order_relaxed)) {
                }
                profBlocks_[idx].fetch_add(1, std::memory_order_relaxed);
            }
        }

        if (sProf) w.pDoneNs = nsBetween(tFrameStart, ProfClock::now());
    } catch (const std::exception& e) {
        std::lock_guard<std::mutex> lk(errMtx_);
        if (frameErr_.empty()) frameErr_ = std::string("worker: ") + e.what();
        frameFailed_ = true;
    } catch (...) {
        std::lock_guard<std::mutex> lk(errMtx_);
        if (frameErr_.empty()) frameErr_ = "worker: unknown exception";
        frameFailed_ = true;
    }
}

// 常驻 worker 的行任务：动态领取 chunk，执行 (*rowFn_)(y0, y1)。
// 与 tile 任务共用同一批线程和完成计数，主线程在 runRows() 里等待。
void SegPipeline::runRowsWorker() {
    for (;;) {
        const int idx = rowNext_.fetch_add(1, std::memory_order_relaxed);
        const int y0 = idx * rowChunk_;
        if (y0 >= rowCount_ || frameFailed_.load(std::memory_order_relaxed)) return;
        const int y1 = std::min(rowCount_, y0 + rowChunk_);
        try {
            if (rowFn_) (*rowFn_)(y0, y1);
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(errMtx_);
            if (frameErr_.empty()) frameErr_ = std::string("row task: ") + e.what();
            frameFailed_ = true;
            return;
        } catch (...) {
            std::lock_guard<std::mutex> lk(errMtx_);
            if (frameErr_.empty()) frameErr_ = "row task: unknown exception";
            frameFailed_ = true;
            return;
        }
    }
}

bool SegPipeline::runRows(int rows, const std::function<void(int, int)>& fn,
                          std::string& err) {
    if (rows <= 0) return true;
    if (threads_.empty()) {   // init 失败/无线程：串行兜底
        fn(0, rows);
        return true;
    }

    const int n = static_cast<int>(threads_.size());
    int chunk = (rows + n * 4 - 1) / (n * 4);   // 每个 worker 约 4 块
    if (chunk < 1) chunk = 1;
    if (chunk > 64) chunk = 64;

    {
        std::lock_guard<std::mutex> lk(poolMtx_);
        poolTask_ = PoolTask::Rows;
        rowFn_ = &fn;
        rowCount_ = rows;
        rowChunk_ = chunk;
        rowNext_.store(0, std::memory_order_relaxed);
        frameFailed_.store(false, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> elk(errMtx_);
            frameErr_.clear();
        }
        done_ = 0;
        ++poolSeq_;
    }
    poolCv_.notify_all();
    {
        std::unique_lock<std::mutex> lk(poolMtx_);
        doneCv_.wait(lk, [&] { return done_ == static_cast<int>(threads_.size()); });
    }
    rowFn_ = nullptr;

    if (frameFailed_.load()) {
        std::lock_guard<std::mutex> lk(errMtx_);
        err = frameErr_.empty() ? "row task failed" : frameErr_;
        return false;
    }
    return true;
}

bool SegPipeline::debugForwardTile(const cv::Mat& bgrIn, std::vector<float>& logits,
                                   std::string& err) {
    cv::Mat bgr;
    if (bgrIn.channels() == 3) {
        bgr = bgrIn;
    } else if (bgrIn.channels() == 4) {
        cv::cvtColor(bgrIn, bgr, cv::COLOR_BGRA2BGR);
    } else if (bgrIn.channels() == 1) {
        cv::cvtColor(bgrIn, bgr, cv::COLOR_GRAY2BGR);
    } else {
        err = "unsupported channel count";
        return false;
    }
    if (bgr.depth() != CV_8U) bgr.convertTo(bgr, CV_8U);

    if (workers_.empty()) {
        err = "pipeline not initialized";
        return false;
    }
    if (cfg_.batch != 1) {
        err = "debugForwardTile requires batch=1";
        return false;
    }

    cv::Mat patch = bgr(cv::Rect(0, 0, std::min(tile_, bgr.cols), std::min(tile_, bgr.rows)));
    if (patch.rows != tile_ || patch.cols != tile_) {
        cv::Mat padded;
        cv::copyMakeBorder(patch, padded, 0, tile_ - patch.rows, 0, tile_ - patch.cols,
                           cv::BORDER_REFLECT);
        patch = padded;
    }

    // 单 tile 调试只需要一路资源；复用 worker[0]，不与 inferImage 并发调用。
    Worker& w = workers_[0];
    fillNCHW(patch, w.hIn, tile_);
    const std::size_t outElems = w.outElems;

    if (!TrtEngine::cudaCheck(
            cudaMemcpyAsync(w.dIn, w.hIn, w.dInBytes, cudaMemcpyHostToDevice, w.stream),
            "cudaMemcpyAsync(H2D)", err)) {
        return false;
    }
    if (!eng_->inferOn(w.ctx, w.dIn, w.dOut, w.stream, err)) return false;
    if (!TrtEngine::cudaCheck(
            cudaMemcpyAsync(w.hOut, w.dOut, w.dOutBytes, cudaMemcpyDeviceToHost, w.stream),
            "cudaMemcpyAsync(D2H)", err)) {
        return false;
    }
    if (!TrtEngine::cudaCheck(cudaStreamSynchronize(w.stream), "cudaStreamSynchronize", err)) {
        return false;
    }
    logits.assign(w.hOut, w.hOut + outElems);
    return true;
}

bool SegPipeline::inferImage(const cv::Mat& bgrIn, cv::Mat& labels, std::string& err) {
    segtrace::Scope scTotal("dll.inferImage");

    const bool prof = segtrace::enabled() || segtrace::timingEnabled();
    const bool needT = prof || prof_;   // CSV 打开失败时 prof_ 仍可能为真
    const ProfClock::time_point tFrameStart = prof ? ProfClock::now() : ProfClock::time_point{};
    if (prof) {
        lastTiming_ = SegFrameTiming{};
        lastTiming_.version = SEG_TIMING_VERSION;
        lastTiming_.size = static_cast<std::int32_t>(sizeof(SegFrameTiming));
        lastTiming_.frameSeq = ++timingSeq_;
        lastTiming_.nWorkers = static_cast<std::int32_t>(threads_.size());
        frameStartTp_ = tFrameStart;
    }

    cv::Mat bgr;
    if (bgrIn.channels() == 3) {
        bgr = bgrIn;
    } else if (bgrIn.channels() == 4) {
        cv::cvtColor(bgrIn, bgr, cv::COLOR_BGRA2BGR);
    } else if (bgrIn.channels() == 1) {
        cv::cvtColor(bgrIn, bgr, cv::COLOR_GRAY2BGR);
    } else {
        err = "unsupported channel count";
        return false;
    }
    if (bgr.depth() != CV_8U) bgr.convertTo(bgr, CV_8U);

    const int H = bgr.rows;
    const int W = bgr.cols;
    if (H < 1 || W < 1) {
        err = "empty image";
        return false;
    }

    std::vector<Tile> tiles;
    {
        const ProfClock::time_point t0 = prof ? ProfClock::now() : ProfClock::time_point{};
        segtrace::Scope sc("dll.collect");
        collectTiles(H, W, tiles);
        if (prof) lastTiming_.collectNs = nsBetween(t0, ProfClock::now());
    }
    lastTiles_ = tiles.size();
    if (prof) lastTiming_.tiles = static_cast<std::int32_t>(tiles.size());

    // 复用成员缓冲：尺寸/类型不变时 create 是空操作，只做 setTo(0)。
    // cv::Mat::create 不初始化内存，所以清零这一步不能省。
    {
        const ProfClock::time_point t0 = prof ? ProfClock::now() : ProfClock::time_point{};
        segtrace::Scope sc("dll.clear");
        if (static_cast<int>(acc_.size()) != cfg_.numClasses) acc_.resize(cfg_.numClasses);
        for (int c = 0; c < cfg_.numClasses; ++c) {
            acc_[c].create(H, W, CV_32F);
            acc_[c].setTo(0);
        }
        if (prof) lastTiming_.clearNs = nsBetween(t0, ProfClock::now());
        if (segtrace::enabled()) {
            sc.setNote("bytes=" + std::to_string(static_cast<long long>(H) * W *
                                                 cfg_.numClasses * 4));
        }
    }

    const auto t0 = prof_ ? std::chrono::high_resolution_clock::now()
                          : std::chrono::high_resolution_clock::time_point{};

    if (workers_.empty()) {
        err = "pipeline not initialized";
        return false;
    }

    const int tileHW = tile_ * tile_;
    const int planeHW = outH_ * outW_;
    const std::size_t upStride =
        static_cast<std::size_t>(cfg_.numClasses) * static_cast<std::size_t>(tileHW);

    // 每块的上采样结果先落到 upAll_[j]，供主线程最后按原顺序串行累加。
    // 每个位置都会被 cv::resize 整块覆盖，不需要清零。只在容量不足时扩容，
    // 避免图像尺寸变化后每帧重新分配（vector 只增不减，相当于高水位缓存）。
    const std::size_t upNeed = tiles.size() * upStride;
    if (upAll_.size() < upNeed) upAll_.resize(upNeed);

    const std::size_t nWorkers = threads_.size();
    if (nWorkers == 0) {
        err = "pipeline not initialized";
        return false;
    }

    if (prof_) {
        profPrepNs_.store(0);
        profGpuNs_.store(0);
        profUpNs_.store(0);
        profGpuMinNs_.store(LLONG_MAX);
        profGpuMaxNs_.store(0);
        for (auto& b : profBlocks_) b.store(0);
        profGpuPerBlock_.resize(tiles.size());
    }

    // 把本帧任务分发给常驻 worker 并等待全部完成。frameBgr_/frameTiles_ 指向本函数的
    // 局部变量，等待返回前一直有效；worker 线程不再每帧创建，消除「线程 1~4 波动」。
    {
        segtrace::Scope sc("dll.infer");
        const ProfClock::time_point tDispatch0 =
            prof ? ProfClock::now() : ProfClock::time_point{};
        {
            std::lock_guard<std::mutex> lk(poolMtx_);
            poolTask_ = PoolTask::Tiles;
            frameBgr_ = &bgr;
            frameTiles_ = &tiles;
            frameNext_.store(0, std::memory_order_relaxed);
            frameFailed_.store(false, std::memory_order_relaxed);
            done_ = 0;
            frameNotifyTp_ = prof ? ProfClock::now() : ProfClock::time_point{};
            {
                std::lock_guard<std::mutex> elk(errMtx_);
                frameErr_.clear();
            }
            ++poolSeq_;
        }
        poolCv_.notify_all();
        if (prof) lastTiming_.dispatchNs = nsBetween(tDispatch0, ProfClock::now());

        const ProfClock::time_point tWait0 =
            prof ? ProfClock::now() : ProfClock::time_point{};
        {
            std::unique_lock<std::mutex> lk(poolMtx_);
            doneCv_.wait(lk, [&] { return done_ == static_cast<int>(nWorkers); });
        }
        if (prof) {
            lastTiming_.waitNs = nsBetween(tWait0, ProfClock::now());
            const std::size_t nw =
                std::min<std::size_t>(workers_.size(), SEG_TIMING_MAX_WORKERS);
            for (std::size_t i = 0; i < nw; ++i) {
                const Worker& w = workers_[i];
                lastTiming_.prepNs[i] = w.pPrepNs;
                lastTiming_.gpuWallNs[i] = w.pGpuWallNs;
                lastTiming_.inferEnqueueNs[i] = w.pInferEnqueueNs;
                lastTiming_.syncNs[i] = w.pSyncNs;
                lastTiming_.h2dGpuNs[i] = w.pH2dNs;
                lastTiming_.inferGpuNs[i] = w.pInferNs;
                lastTiming_.d2hGpuNs[i] = w.pD2hNs;
                lastTiming_.upNs[i] = w.pUpNs;
                lastTiming_.wakeNs[i] = w.pWakeNs;
                lastTiming_.tailNs[i] =
                    w.pTiles > 0 ? (w.pDoneNs - w.pLastEndNs) : 0;
                lastTiming_.tileCount[i] = w.pTiles;
            }
        }
        if (segtrace::enabled()) {
            char buf[192];
            std::snprintf(buf, sizeof(buf),
                          "prep=%.1f gpu=%.1f up=%.1f blocks=%zu prof=%d",
                          profPrepNs_.load() * 1e-6, profGpuNs_.load() * 1e-6,
                          profUpNs_.load() * 1e-6, tiles.size(), prof_ ? 1 : 0);
            sc.setNote(buf);
        }
    }

    if (frameFailed_.load()) {
        std::lock_guard<std::mutex> lk(errMtx_);
        err = frameErr_.empty() ? "inference failed" : frameErr_;
        return false;
    }

    const auto tAcc0 = needT ? ProfClock::now() : ProfClock::time_point{};
    // 累加只能串行：stride(448) < tile(640)，相邻块的 ROI 是重叠的。
    // 仍按 tile 原始顺序累加，保证与 Python / 30 系 TRT8.6 逐像素一致。
    {
        segtrace::Scope sc("dll.post.acc");
        for (std::size_t j = 0; j < tiles.size(); ++j) {
            const Tile& t = tiles[j];
            const float* up = upAll_.data() + j * upStride;
            for (int c = 0; c < cfg_.numClasses; ++c) {
                cv::Mat upm(tile_, tile_, CV_32F,
                            const_cast<float*>(up + static_cast<std::size_t>(c) * tileHW));
                cv::Mat roi = acc_[c](cv::Rect(t.x0, t.y0, t.pw, t.ph));
                roi += upm(cv::Rect(0, 0, t.pw, t.ph));
            }
        }
    }
    const auto tAcc1 = needT ? ProfClock::now() : ProfClock::time_point{};

    // argmax：Python 是 argmax(acc/cov)，但 cov 对每个像素的三类相同，可约掉。
    // cov 为 0 的像素三类都是 0 -> argmax 取 0（背景），与 Python 一致。
    {
        segtrace::Scope sc("dll.post.argmax");
        labels.create(H, W, CV_8UC1);
        if (!runRows(H, [&](int y0, int y1) {
                for (int y = y0; y < y1; ++y) {
                    const float* a0 = acc_[0].ptr<float>(y);
                    const float* a1 = acc_[1].ptr<float>(y);
                    const float* a2 = acc_.size() > 2u ? acc_[2].ptr<float>(y) : nullptr;
                    uchar* out = labels.ptr<uchar>(y);
                    for (int x = 0; x < W; ++x) {
                        int best = 0;
                        float bv = a0[x];
                        if (a1[x] > bv) {
                            best = 1;
                            bv = a1[x];
                        }
                        if (a2 && a2[x] > bv) {
                            best = 2;
                            bv = a2[x];
                        }
                        out[x] = static_cast<uchar>(best);
                    }
                }
            }, err)) {
            return false;
        }
    }
    const auto tArg1 = needT ? ProfClock::now() : ProfClock::time_point{};

    if (prof) {
        lastTiming_.accNs = nsBetween(tAcc0, tAcc1);
        lastTiming_.argmaxNs = nsBetween(tAcc1, tArg1);
        lastTiming_.totalNs = nsBetween(tFrameStart, ProfClock::now());
    }

    const auto t1 = prof_ ? std::chrono::high_resolution_clock::now()
                          : std::chrono::high_resolution_clock::time_point{};
    if (prof_) lastMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (segtrace::enabled()) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "w=%d h=%d tiles=%zu", W, H, tiles.size());
        scTotal.setNote(buf);
    }

    if (prof_) {
        const double ms = 1.0 / 1e6;
        long long gpuMin = profGpuMinNs_.load();
        if (gpuMin == LLONG_MAX) gpuMin = 0;
        long long med = 0;
        if (!profGpuPerBlock_.empty()) {
            std::vector<long long> v = profGpuPerBlock_;
            const std::size_t mid = v.size() / 2;
            std::nth_element(v.begin(), v.begin() + mid, v.end());
            med = v[mid];
        }
        std::fprintf(stderr, "[profile] workers=%zu blocks=[", nWorkers);
        for (std::size_t i = 0; i < nWorkers && i < 16; ++i) {
            std::fprintf(stderr, "%s%d", i ? "," : "", profBlocks_[i].load());
        }
        std::fprintf(stderr,
                     "] prep=%.1f gpu=%.1f(min=%.1f med=%.1f max=%.1f) up=%.1f "
                     "acc=%.1f argmax=%.1f total=%.1f\n",
                     profPrepNs_.load() * ms, profGpuNs_.load() * ms, gpuMin * ms,
                     med * ms, profGpuMaxNs_.load() * ms, profUpNs_.load() * ms,
                     std::chrono::duration<double, std::milli>(tAcc1 - tAcc0).count(),
                     std::chrono::duration<double, std::milli>(tArg1 - tAcc1).count(),
                     lastMs_);
    }
    return true;
}
