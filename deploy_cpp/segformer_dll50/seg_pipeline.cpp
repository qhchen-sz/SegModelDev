// ★ segformer_dll50（RTX50 / TensorRT 10）专用：固定 batch=1 + 多路 context 并发。
//   与 ..\segformer_dll 及 ..\src 的版本已分叉，不要再互相同步（见 seg_pipeline.hpp 说明）。
#include "seg_pipeline.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>

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

// 多 worker 并发期间把 OpenCV 内部线程池压到 1：否则 4 路 worker 各自再开
// OpenCV 线程会造成严重过订阅（resize 本身是访存受限，单线程足够）。
// 仅在「worker 创建 + 推理」这段作用域内生效，argmax/叠加图仍用默认线程数。
struct CvThreadsToOne {
    int old;
    CvThreadsToOne() : old(cv::getNumThreads()) { cv::setNumThreads(1); }
    ~CvThreadsToOne() { cv::setNumThreads(old); }
};

}  // namespace

cv::Mat segOverlay(const cv::Mat& bgr, const cv::Mat& labels, float alpha) {
    const cv::Vec3f cols[3] = {cv::Vec3f(60, 60, 60), cv::Vec3f(0, 0, 255),
                               cv::Vec3f(255, 0, 0)};
    cv::Mat ov = bgr.clone();
    // 逐行独立，行并行不改变任何输出像素（叠加图在 DLL 的每帧路径上，值得并行）
    cv::parallel_for_(cv::Range(0, ov.rows), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            const uchar* lab = labels.ptr<uchar>(y);
            cv::Vec3b* row = ov.ptr<cv::Vec3b>(y);
            for (int x = 0; x < ov.cols; ++x) {
                const cv::Vec3f& c = cols[std::min<int>(lab[x], 2)];
                for (int k = 0; k < 3; ++k) {
                    row[x][k] = static_cast<uchar>(row[x][k] * (1.0f - alpha) + c[k] * alpha);
                }
            }
        }
    });
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
    for (auto& w : workers_) {
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
    const int nWant = std::max(1, cfg_.numWorkers);

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
            outH_ = oH;
            outW_ = oW;
            (void)outElems;
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
    collectTiles(H, W, tiles);

    std::vector<cv::Mat> acc(cfg_.numClasses);
    for (int c = 0; c < cfg_.numClasses; ++c) acc[c] = cv::Mat::zeros(H, W, CV_32F);

    const auto t0 = std::chrono::high_resolution_clock::now();

    if (workers_.empty()) {
        err = "pipeline not initialized";
        return false;
    }

    const int tileHW = tile_ * tile_;
    const int planeHW = outH_ * outW_;
    const std::size_t upStride =
        static_cast<std::size_t>(cfg_.numClasses) * static_cast<std::size_t>(tileHW);

    // 每块的上采样结果先落到 upAll_[j]，供主线程最后按原顺序串行累加。
    // 每个位置都会被 cv::resize 整块覆盖，不需要清零；resize 同尺寸时是空操作。
    upAll_.resize(tiles.size() * upStride);

    // 任务队列：worker 并行做「预处理 -> H2D -> 推理 -> D2H -> 上采样」，
    // 各 tile 完全独立；累加仍留给主线程串行完成（见下）。
    std::atomic<std::size_t> next{0};
    std::atomic<bool> failed{false};
    std::mutex errMutex;
    std::string workerErr;

    auto run = [&](Worker& w) {
        try {
            for (;;) {
                const std::size_t j = next.fetch_add(1);
                if (j >= tiles.size() || failed.load()) break;

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

                std::string e;
                if (!TrtEngine::cudaCheck(
                        cudaMemcpyAsync(w.dIn, w.hIn, w.dInBytes, cudaMemcpyHostToDevice,
                                        w.stream),
                        "cudaMemcpyAsync(H2D)", e) ||
                    !eng_->inferOn(w.ctx, w.dIn, w.dOut, w.stream, e) ||
                    !TrtEngine::cudaCheck(
                        cudaMemcpyAsync(w.hOut, w.dOut, w.dOutBytes, cudaMemcpyDeviceToHost,
                                        w.stream),
                        "cudaMemcpyAsync(D2H)", e) ||
                    !TrtEngine::cudaCheck(cudaStreamSynchronize(w.stream),
                                          "cudaStreamSynchronize", e)) {
                    std::lock_guard<std::mutex> lk(errMutex);
                    if (workerErr.empty()) workerErr = e;
                    failed = true;
                    return;
                }

                float* dst = upAll_.data() + j * upStride;
                for (int c = 0; c < cfg_.numClasses; ++c) {
                    cv::Mat src(outH_, outW_, CV_32F,
                                w.hOut + static_cast<std::size_t>(c) * planeHW);
                    cv::Mat d(tile_, tile_, CV_32F,
                              dst + static_cast<std::size_t>(c) * tileHW);
                    // 与 F.interpolate(..., bilinear, align_corners=False) 同为半像素中心约定
                    cv::resize(src, d, cv::Size(tile_, tile_), 0, 0, cv::INTER_LINEAR);
                }
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lk(errMutex);
            if (workerErr.empty()) workerErr = std::string("worker: ") + e.what();
            failed = true;
        }
    };

    std::vector<std::thread> ths;
    {
        CvThreadsToOne cvGuard;   // worker 内 cv::resize 单线程，避免与 4 路并发叠加
        ths.reserve(workers_.size());
        try {
            for (auto& w : workers_) ths.emplace_back(run, std::ref(w));
        } catch (const std::exception& e) {
            // 线程创建失败：先等已启动的线程结束，再返回错误（thread 析构前必须 join）
            for (auto& th : ths) {
                if (th.joinable()) th.join();
            }
            err = std::string("thread create failed: ") + e.what();
            return false;
        }
        for (auto& th : ths) th.join();
    }

    if (failed.load()) {
        err = workerErr.empty() ? "inference failed" : workerErr;
        return false;
    }

    // 累加只能串行：stride(448) < tile(640)，相邻块的 ROI 是重叠的。
    // 仍按 tile 原始顺序累加，保证与 Python / 30 系 TRT8.6 逐像素一致。
    for (std::size_t j = 0; j < tiles.size(); ++j) {
        const Tile& t = tiles[j];
        const float* up = upAll_.data() + j * upStride;
        for (int c = 0; c < cfg_.numClasses; ++c) {
            cv::Mat upm(tile_, tile_, CV_32F,
                        const_cast<float*>(up + static_cast<std::size_t>(c) * tileHW));
            cv::Mat roi = acc[c](cv::Rect(t.x0, t.y0, t.pw, t.ph));
            roi += upm(cv::Rect(0, 0, t.pw, t.ph));
        }
    }

    // argmax：Python 是 argmax(acc/cov)，但 cov 对每个像素的三类相同，可约掉。
    // cov 为 0 的像素三类都是 0 -> argmax 取 0（背景），与 Python 一致。
    labels.create(H, W, CV_8UC1);
    cv::parallel_for_(cv::Range(0, H), [&](const cv::Range& r) {
        for (int y = r.start; y < r.end; ++y) {
            const float* a0 = acc[0].ptr<float>(y);
            const float* a1 = acc[1].ptr<float>(y);
            const float* a2 = acc.size() > 2 ? acc[2].ptr<float>(y) : nullptr;
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
    });

    const auto t1 = std::chrono::high_resolution_clock::now();
    lastMs_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return true;
}
