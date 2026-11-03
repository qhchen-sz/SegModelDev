#include "segformer_dll.h"

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

// 与 CMakeLists.txt 的 DLL 目标 / segformer_dll.vcxproj 保持一致：本 DLL 编译并使用
// 本目录下的自包含副本。若包装层包含 ..\src 的头文件而实现编译本目录副本，
// 两份头文件一旦漂移，同一 DLL 的不同 TU 会看到不同的类布局（ODR 违规）。
#include "seg_pipeline.hpp"
#include "trt_engine.hpp"

namespace {

// 与 sweep_eval.py 完全一致的口径：tile 由引擎决定(640)，stride 448，batch 16。
// 刻意不暴露成可调参数——这是已验证的配置，调歪了只会静默掉精度。
const SegConfig kConfig{};

thread_local std::string tlsError;

void setError(const std::string& msg) {
    tlsError = msg;
}

struct Handle {
    TrtEngine engine;
    SegPipeline pipe;
    cv::Mat lastInput;   // 原图副本（叠加图要用；copyTo 复用缓冲，不每帧重新分配）
    cv::Mat lastLabels;  // 最近一次 Forward 的 0/1/2 语义图
    cv::Mat lastOverlay; // 叠加图输出缓冲，SegGetOverlay 每帧复用，不再 clone 原图
    // computeStat 的复用缓冲：避免每帧重新分配数十 MB（statLabels 全图 CV_32S 约 31MB）
    cv::Mat statBin;        // CV_8U  H x W，(labels == c) 的二值图
    cv::Mat statLabels;     // CV_32S H x W，连通域标签
    cv::Mat statStats;      // CV_32S N x 5
    cv::Mat statCentroids;  // CV_64F N x 2
    bool loaded = false;
    bool statValid = false;
    SegStat stat{};
};

Handle* asHandle(void* p) {
    return static_cast<Handle*>(p);
}

// 对 lastLabels 做像素统计 + 连通域分析（按需调用，结果缓存在 handle 里）
void computeStat(Handle* h) {
    SegStat& s = h->stat;
    std::memset(&s, 0, sizeof(s));
    const cv::Mat& lab = h->lastLabels;

    for (int y = 0; y < lab.rows; ++y) {
        const uchar* p = lab.ptr<uchar>(y);
        for (int x = 0; x < lab.cols; ++x) {
            const int c = p[x] <= 2 ? p[x] : 0;
            s.px[c]++;
        }
    }
    s.ng = (s.px[1] + s.px[2]) > 0 ? 1 : 0;

    // 全背景（OK 品）：没有缺陷类像素，连通域结果必然为空，直接返回。
    // 这一步能省掉两遍全图连通的扫描，是每帧最大的单项优化之一。
    if (s.px[1] == 0 && s.px[2] == 0) {
        s.ncomp[1] = 0;
        s.ncomp[2] = 0;
        s.nbox = 0;
        return;
    }

    struct Comp {
        int cls, x, y, w, h;
        long long px;
    };
    std::vector<Comp> comps;

    for (int c = 1; c <= 2; ++c) {
        if (s.px[c] == 0) {  // 该类没有像素，不必扫全图
            s.ncomp[c] = 0;
            continue;
        }
        // 复用 statBin/statLabels/statStats/statCentroids，避免每帧几十 MB 的分配。
        // compare 对 8U 输入输出 8U（0/255），与原 (lab == c) 等价。
        cv::compare(lab, cv::Scalar(c), h->statBin, cv::CMP_EQ);
        const int n = cv::connectedComponentsWithStats(h->statBin, h->statLabels,
                                                       h->statStats, h->statCentroids, 8,
                                                       CV_32S);
        s.ncomp[c] = n > 0 ? n - 1 : 0;  // 0 号是背景连通域，不算
        for (int i = 1; i < n; ++i) {
            Comp cp;
            cp.cls = c;
            cp.x = h->statStats.at<int>(i, cv::CC_STAT_LEFT);
            cp.y = h->statStats.at<int>(i, cv::CC_STAT_TOP);
            cp.w = h->statStats.at<int>(i, cv::CC_STAT_WIDTH);
            cp.h = h->statStats.at<int>(i, cv::CC_STAT_HEIGHT);
            cp.px = h->statStats.at<int>(i, cv::CC_STAT_AREA);
            comps.push_back(cp);
        }
    }

    // 按面积降序：SEG_MAX_BOX 装不下时丢掉的是最小的那些，而 ncomp[] 报的是真实总数，
    // 所以截断不会被掩盖。
    std::sort(comps.begin(), comps.end(),
              [](const Comp& a, const Comp& b) { return a.px > b.px; });
    const int n = static_cast<int>(std::min<std::size_t>(comps.size(), SEG_MAX_BOX));
    s.nbox = n;
    for (int i = 0; i < n; ++i) {
        s.boxCls[i] = comps[i].cls;
        s.boxX[i] = comps[i].x;
        s.boxY[i] = comps[i].y;
        s.boxW[i] = comps[i].w;
        s.boxH[i] = comps[i].h;
        s.boxPx[i] = comps[i].px;
    }
}

// 把 cv::Mat 拷成调用方负责释放的裸缓冲
const char* copyOut(const cv::Mat& m, int& width, int& height, int& step, int& channels) {
    width = m.cols;
    height = m.rows;
    step = static_cast<int>(m.step);
    channels = m.channels();
    const std::size_t n = m.step * m.rows;
    char* buf = new char[n];
    std::memcpy(buf, m.data, n);
    return buf;
}

}  // namespace

// ============================================================================

void* SegCreate() {
    try {
        return new Handle();
    } catch (const std::exception& e) {
        setError(std::string("SegCreate: ") + e.what());
        return nullptr;
    }
}

void SegDestroy(void* model) {
    delete asHandle(model);
}

bool SegLoad(const char* enginePath, void* model) {
    Handle* h = asHandle(model);
    if (!h) {
        setError("SegLoad: null model handle");
        return false;
    }
    if (!enginePath || !*enginePath) {
        setError("SegLoad: empty engine path");
        return false;
    }
    // 从这一刻起模型处于“加载中”：下面任何一步失败，engine.load()/pipe.init()
    // 都可能已经释放了旧资源，绝不能让 SegForward 误以为模型仍可用。
    // 同时丢掉上一模型的结果，避免重新加载后 GetOverlay/GetMask 返回旧帧。
    h->loaded = false;
    h->statValid = false;
    h->lastInput.release();
    h->lastLabels.release();
    h->lastOverlay.release();
    h->statBin.release();
    h->statLabels.release();
    h->statStats.release();
    h->statCentroids.release();

    try {
        std::string err;
        if (!h->engine.load(enginePath, err)) {
            setError("SegLoad: " + err);
            return false;
        }
        if (!h->pipe.init(h->engine, kConfig, err)) {
            setError("SegLoad: pipeline init: " + err);
            return false;
        }
    } catch (const std::exception& e) {
        // 不允许异常穿过 C ABI（C# P/Invoke 侧是未定义行为）
        setError(std::string("SegLoad: ") + e.what());
        return false;
    }

    h->loaded = true;
    return true;
}

int SegForward(unsigned char* bgrData, int width, int height, int stride, void* model) {
    Handle* h = asHandle(model);
    if (!h) {
        setError("SegForward: null model handle");
        return 1;
    }
    if (!h->loaded) {
        setError("SegForward: model not loaded (call SegLoad first)");
        return 2;
    }
    if (!bgrData || width <= 0 || height <= 0 || stride <= 0) {
        setError("SegForward: bad image arguments");
        return 3;
    }

    try {
        const cv::Mat src(height, width, CV_8UC3, bgrData, static_cast<std::size_t>(stride));
        src.copyTo(h->lastInput);
        std::string err;
        if (!h->pipe.inferImage(h->lastInput, h->lastLabels, err)) {
            setError("SegForward: " + err);
            return 4;
        }
    } catch (const std::exception& e) {
        setError(std::string("SegForward: ") + e.what());
        return 5;
    }

    h->statValid = false;
    return 0;
}

// 只推理、不保存原图、不生成叠加图。省掉 lastInput.copyTo 与后续 overlay 的全部开销。
// 行为与 50 系列保持一致：不主动释放上一帧的 lastInput。
int SegForwardMask(unsigned char* bgrData, int width, int height, int stride, void* model) {
    Handle* h = asHandle(model);
    if (!h) {
        setError("SegForwardMask: null model handle");
        return 1;
    }
    if (!h->loaded) {
        setError("SegForwardMask: model not loaded (call SegLoad first)");
        return 2;
    }
    if (!bgrData || width <= 0 || height <= 0 || stride <= 0) {
        setError("SegForwardMask: bad image arguments");
        return 3;
    }

    try {
        // 直接引用调用方内存（LockBits 期间有效）；inferImage 内部不会长期持有它。
        const cv::Mat src(height, width, CV_8UC3, bgrData, static_cast<std::size_t>(stride));
        std::string err;
        if (!h->pipe.inferImage(src, h->lastLabels, err)) {
            setError("SegForwardMask: " + err);
            return 4;
        }
    } catch (const std::exception& e) {
        setError(std::string("SegForwardMask: ") + e.what());
        return 5;
    }

    h->statValid = false;
    return 0;
}

const char* SegGetOverlay(int& width, int& height, int& step, int& channels, void* model) {
    Handle* h = asHandle(model);
    if (!h || h->lastInput.empty() || h->lastLabels.empty()) {
        setError("SegGetOverlay: no result yet (call SegForward first)");
        return nullptr;
    }
    try {
        // 写入复用的 lastOverlay，避免每帧 clone 一份 23MB 原图
        segOverlayInto(h->lastInput, h->lastLabels, h->lastOverlay, 0.5f);
        return copyOut(h->lastOverlay, width, height, step, channels);
    } catch (const std::exception& e) {
        setError(std::string("SegGetOverlay: ") + e.what());
        return nullptr;
    }
}

const char* SegGetMask(int& width, int& height, int& step, int& channels, void* model) {
    Handle* h = asHandle(model);
    if (!h || h->lastLabels.empty()) {
        setError("SegGetMask: no result yet (call SegForward first)");
        return nullptr;
    }
    try {
        return copyOut(h->lastLabels, width, height, step, channels);
    } catch (const std::exception& e) {
        setError(std::string("SegGetMask: ") + e.what());
        return nullptr;
    }
}

void SegFreeImage(const char* imageData) {
    delete[] imageData;
}

void SegGetStat(void* model, SegStat* out) {
    if (!out) return;
    std::memset(out, 0, sizeof(SegStat));

    Handle* h = asHandle(model);
    if (!h || h->lastLabels.empty()) {
        setError("SegGetStat: no result yet (call SegForward first)");
        return;
    }
    try {
        if (!h->statValid) {
            computeStat(h);
            h->statValid = true;
        }
        *out = h->stat;
    } catch (const std::exception& e) {
        setError(std::string("SegGetStat: ") + e.what());
    }
}

int SegStatSize() {
    return static_cast<int>(sizeof(SegStat));
}

const char* SegLastError() {
    // 线程局部缓冲：调用方拿到后应立即拷走
    return tlsError.c_str();
}

BOOL APIENTRY DllMain(HMODULE hModule, DWORD ul_reason_for_call, LPVOID lpReserved) {
    (void)hModule;
    (void)ul_reason_for_call;
    (void)lpReserved;
    return TRUE;
}
