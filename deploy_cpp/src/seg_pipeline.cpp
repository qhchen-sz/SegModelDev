#include "seg_pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>

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
// 每轮会漏掉约 160 MB（75 MiB 显存 + 75 MiB pinned 主机内存）以及一个 stream。
void SegPipeline::release() {
    // 显存是 cudaMalloc 在本进程 CUDA 上下文里分配的，经哪个 TrtEngine 释放都可以；
    // 仅用 eng_ 做非空判断（从未 init 过时这些指针本来就是空的）。
    if (eng_) {
        eng_->freeDevice(dIn_);
        eng_->freeDevice(dOut_);
    }
    dIn_ = nullptr;
    dOut_ = nullptr;
    dInBytes_ = 0;
    dOutBytes_ = 0;
    if (hIn_) {
        cudaFreeHost(hIn_);
        hIn_ = nullptr;
    }
    if (hOut_) {
        cudaFreeHost(hOut_);
        hOut_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    std::vector<float>().swap(upBuf_);   // 真正归还约 75 MB 的上采样中转缓冲
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
    if (cfg_.batch < 1) cfg_.batch = 1;
    if (cfg_.batch > eng.maxBatch()) cfg_.batch = eng.maxBatch();

    std::size_t outElems = 0;
    if (!eng.setBatch(cfg_.batch, outElems, err)) return false;
    const std::vector<int>& od = eng.outputDims();
    if (od.size() != 4 || od[1] != cfg_.numClasses) {
        err = "unexpected engine output dims";
        return false;
    }
    outH_ = od[2];
    outW_ = od[3];

    dInBytes_ = static_cast<std::size_t>(cfg_.batch) * 3 * tile_ * tile_ * sizeof(float);
    dOutBytes_ = outElems * sizeof(float);

    dIn_ = eng.allocDevice(dInBytes_, err);
    if (!dIn_) return false;
    dOut_ = eng.allocDevice(dOutBytes_, err);
    if (!dOut_) return false;
    if (!TrtEngine::cudaCheck(cudaStreamCreate(&stream_), "cudaStreamCreate", err)) return false;

    // pinned 主机内存：否则 cudaMemcpyAsync 对 pageable 内存会退化成同步拷贝
    if (!TrtEngine::cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&hIn_), dInBytes_,
                                            cudaHostAllocDefault),
                              "cudaHostAlloc(input)", err)) {
        return false;
    }
    if (!TrtEngine::cudaCheck(cudaHostAlloc(reinterpret_cast<void**>(&hOut_), dOutBytes_,
                                            cudaHostAllocDefault),
                              "cudaHostAlloc(output)", err)) {
        return false;
    }
    upBuf_.resize(static_cast<std::size_t>(cfg_.batch) * cfg_.numClasses * tile_ * tile_);

    return true;
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

    cv::Mat patch = bgr(cv::Rect(0, 0, std::min(tile_, bgr.cols), std::min(tile_, bgr.rows)));
    if (patch.rows != tile_ || patch.cols != tile_) {
        cv::Mat padded;
        cv::copyMakeBorder(patch, padded, 0, tile_ - patch.rows, 0, tile_ - patch.cols,
                           cv::BORDER_REFLECT);
        patch = padded;
    }
    fillNCHW(patch, hIn_, tile_);

    std::size_t outElems = 0;
    if (!eng_->setBatch(1, outElems, err)) return false;
    const std::size_t inBytes = static_cast<std::size_t>(3) * tile_ * tile_ * sizeof(float);
    const std::size_t outBytes = outElems * sizeof(float);

    if (!TrtEngine::cudaCheck(
            cudaMemcpyAsync(dIn_, hIn_, inBytes, cudaMemcpyHostToDevice, stream_),
            "cudaMemcpyAsync(H2D)", err)) {
        return false;
    }
    if (!eng_->infer(dIn_, dOut_, stream_, err)) return false;
    if (!TrtEngine::cudaCheck(
            cudaMemcpyAsync(hOut_, dOut_, outBytes, cudaMemcpyDeviceToHost, stream_),
            "cudaMemcpyAsync(D2H)", err)) {
        return false;
    }
    if (!TrtEngine::cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", err)) {
        return false;
    }
    logits.assign(hOut_, hOut_ + outElems);
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

    for (std::size_t i = 0; i < tiles.size(); i += cfg_.batch) {
        const int n = static_cast<int>(std::min<std::size_t>(cfg_.batch, tiles.size() - i));

        // 各 j 写 hIn_ 里互不重叠的一段，可并行
        cv::parallel_for_(cv::Range(0, n), [&](const cv::Range& r) {
            for (int j = r.start; j < r.end; ++j) {
                const Tile& t = tiles[i + j];
                cv::Mat patch = bgr(cv::Rect(t.x0, t.y0, t.pw, t.ph));
                if (t.ph != tile_ || t.pw != tile_) {
                    cv::Mat padded;
                    // cv2.BORDER_REFLECT (不是默认的 BORDER_REFLECT_101)
                    cv::copyMakeBorder(patch, padded, 0, tile_ - t.ph, 0, tile_ - t.pw,
                                       cv::BORDER_REFLECT);
                    patch = padded;
                }
                fillNCHW(patch, hIn_ + static_cast<std::size_t>(j) * 3 * tile_ * tile_, tile_);
            }
        });

        std::size_t outElems = 0;
        if (!eng_->setBatch(n, outElems, err)) return false;
        const std::size_t perImage = outElems / static_cast<std::size_t>(n);
        const std::size_t inBytes =
            static_cast<std::size_t>(n) * 3 * tile_ * tile_ * sizeof(float);
        const std::size_t outBytes = outElems * sizeof(float);

        if (!TrtEngine::cudaCheck(
                cudaMemcpyAsync(dIn_, hIn_, inBytes, cudaMemcpyHostToDevice, stream_),
                "cudaMemcpyAsync(H2D)", err)) {
            return false;
        }
        if (!eng_->infer(dIn_, dOut_, stream_, err)) return false;
        if (!TrtEngine::cudaCheck(
                cudaMemcpyAsync(hOut_, dOut_, outBytes, cudaMemcpyDeviceToHost, stream_),
                "cudaMemcpyAsync(D2H)", err)) {
            return false;
        }
        if (!TrtEngine::cudaCheck(cudaStreamSynchronize(stream_), "cudaStreamSynchronize", err)) {
            return false;
        }

        const int planeHW = outH_ * outW_;
        const int tileHW = tile_ * tile_;

        // 上采样并行化：每个 (j,c) 写 upBuf_ 里互不重叠的一段
        cv::parallel_for_(cv::Range(0, n * cfg_.numClasses), [&](const cv::Range& r) {
            for (int idx = r.start; idx < r.end; ++idx) {
                const int j = idx / cfg_.numClasses;
                const int c = idx % cfg_.numClasses;
                cv::Mat src(outH_, outW_, CV_32F,
                            const_cast<float*>(hOut_ + static_cast<std::size_t>(j) * perImage +
                                               static_cast<std::size_t>(c) * planeHW));
                cv::Mat dst(tile_, tile_, CV_32F,
                            upBuf_.data() + static_cast<std::size_t>(idx) * tileHW);
                // 与 F.interpolate(..., bilinear, align_corners=False) 同为半像素中心约定
                cv::resize(src, dst, cv::Size(tile_, tile_), 0, 0, cv::INTER_LINEAR);
            }
        });

        // 累加只能串行：stride(448) < tile(640)，相邻块的 ROI 是重叠的
        for (int j = 0; j < n; ++j) {
            const Tile& t = tiles[i + j];
            for (int c = 0; c < cfg_.numClasses; ++c) {
                const int idx = j * cfg_.numClasses + c;
                cv::Mat up(tile_, tile_, CV_32F,
                           upBuf_.data() + static_cast<std::size_t>(idx) * tileHW);
                cv::Mat roi = acc[c](cv::Rect(t.x0, t.y0, t.pw, t.ph));
                roi += up(cv::Rect(0, 0, t.pw, t.ph));
            }
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
