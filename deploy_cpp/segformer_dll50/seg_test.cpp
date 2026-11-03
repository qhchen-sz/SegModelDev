// ============================================================================
// seg_test.cpp —— segformer_Dll_func.dll 的测试用可执行程序。
//
// 刻意用 **动态加载**（LoadLibrary + GetProcAddress）调用 DLL 导出的纯 C ABI，
// 只依赖 DLL 本身与它的公开头文件 —— 这就是软件里 C# 经 P/Invoke 走的那套接口，
// 所以本 exe 通过 = DLL 的对外接口通过。
//
// 用法：
//   seg_test.exe <engine> <图 或 目录> <输出目录> [--mask] [--dll <dll路径>]
//   seg_test.exe <engine> --eval <val_full.txt> [--dll <dll路径>]
//   seg_test.exe <engine> --eval <val_full.txt> --no-forward-stat   (评测时跳过连通域统计，更快)
//   seg_test.exe <engine> --soak <图 或 目录> <迭代次数> [--dll <dll路径>]
//     浸泡测试：同一进程内反复 Forward / 取图 / 释放，并重复 SegLoad 多次，
//     定期打印进程工作集与私有提交量，用于排查逐帧泄漏与重复加载泄漏。
//
// val_full.txt 每行：<图片路径>\t<GT语义图路径>  （GT 像素 0/1/2）
// 输出：叠加图 <out>/<stem>.png；--mask 时再存 <out>/mask/<stem>.png
// ============================================================================

#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cctype>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <filesystem>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ---- 与 segformer_dll.h 完全一致的 SegStat 布局（sizeof 由 SegStatSize() 断言）----
#define SEG_MAX_BOX 64
struct SegStat {
    int       ng;
    int       ncomp[3];
    int       nbox;
    int       boxCls[SEG_MAX_BOX];
    int       boxX[SEG_MAX_BOX];
    int       boxY[SEG_MAX_BOX];
    int       boxW[SEG_MAX_BOX];
    int       boxH[SEG_MAX_BOX];
    long long px[3];
    long long boxPx[SEG_MAX_BOX];
};

// ---- 动态解析的 C ABI 函数指针 ----
typedef void*       (*SegCreate_t)();
typedef void        (*SegDestroy_t)(void*);
typedef bool        (*SegLoad_t)(const char*, void*);
typedef int         (*SegForward_t)(unsigned char*, int, int, int, void*);
typedef int         (*SegForwardMask_t)(unsigned char*, int, int, int, void*);
typedef const char* (*SegGetOverlay_t)(int&, int&, int&, int&, void*);
typedef const char* (*SegGetMask_t)(int&, int&, int&, int&, void*);
typedef void        (*SegFreeImage_t)(const char*);
typedef void        (*SegGetStat_t)(void*, SegStat*);
typedef int         (*SegStatSize_t)();
typedef const char* (*SegLastError_t)();

struct DllApi {
    HMODULE h = nullptr;
    SegCreate_t     Create = nullptr;
    SegDestroy_t    Destroy = nullptr;
    SegLoad_t       Load = nullptr;
    SegForward_t    Forward = nullptr;
    SegForwardMask_t ForwardMask = nullptr;   // 可选导出：旧 DLL 没有时回退到 Forward
    SegGetOverlay_t GetOverlay = nullptr;
    SegGetMask_t    GetMask = nullptr;
    SegFreeImage_t  FreeImage = nullptr;
    SegGetStat_t    GetStat = nullptr;
    SegStatSize_t   StatSize = nullptr;
    SegLastError_t  LastError = nullptr;

    template <typename T> bool grab(const char* name, T& out) {
        out = reinterpret_cast<T>(GetProcAddress(h, name));
        if (!out) { std::fprintf(stderr, "[ERR] DLL 缺少导出函数: %s\n", name); return false; }
        return true;
    }
    bool open(const std::string& dllPath) {
        h = LoadLibraryA(dllPath.c_str());
        if (!h) {
            std::fprintf(stderr, "[ERR] LoadLibrary 失败 (err=%lu): %s\n"
                                 "      该 DLL 依赖 nvinfer.dll / opencv_world480.dll 等，"
                                 "请先设好 PATH（见 run_seg_test.bat）。\n",
                         GetLastError(), dllPath.c_str());
            return false;
        }
        // 可选导出：缺失时不报错，--runmask 自动回退到 SegForward
        ForwardMask = reinterpret_cast<SegForwardMask_t>(GetProcAddress(h, "SegForwardMask"));
        return grab("SegCreate", Create) && grab("SegDestroy", Destroy) &&
               grab("SegLoad", Load) && grab("SegForward", Forward) &&
               grab("SegGetOverlay", GetOverlay) && grab("SegGetMask", GetMask) &&
               grab("SegFreeImage", FreeImage) && grab("SegGetStat", GetStat) &&
               grab("SegStatSize", StatSize) && grab("SegLastError", LastError);
    }
    ~DllApi() { if (h) FreeLibrary(h); }
};

static std::string dllErr(DllApi& a) {
    const char* e = a.LastError ? a.LastError() : "(no LastError)";
    return e ? std::string(e) : std::string();
}

// 把 DLL 返回的裸缓冲拷成 cv::Mat（深拷贝后释放原缓冲）
static cv::Mat takeImage(DllApi& a, const char* buf, int w, int h, int step, int ch) {
    if (!buf) return cv::Mat();
    const int type = CV_MAKETYPE(CV_8U, ch);
    cv::Mat view(h, w, type, const_cast<char*>(buf), step);
    cv::Mat out = view.clone();
    a.FreeImage(buf);
    return out;
}

static std::vector<std::string> listImages(const std::string& path) {
    std::vector<std::string> files;
    if (fs::is_directory(path)) {
        for (auto& e : fs::directory_iterator(path)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (auto& c : ext) c = (char)std::tolower((unsigned char)c);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(path);
    }
    return files;
}

static void printStat(const SegStat& s) {
    std::printf("    ng=%d  像素: bg=%lld silie=%lld fanzhe=%lld\n",
                s.ng, s.px[0], s.px[1], s.px[2]);
    std::printf("    连通域数: silie=%d fanzhe=%d (boxes=%d)\n",
                s.ncomp[1], s.ncomp[2], s.nbox);
    for (int i = 0; i < s.nbox; ++i) {
        std::printf("      #%d cls=%d box=(%d,%d,%d,%d) px=%lld\n",
                    i, s.boxCls[i], s.boxX[i], s.boxY[i], s.boxW[i], s.boxH[i], s.boxPx[i]);
    }
}

// 进程内存快照：工作集看物理占用，私有提交量看真实申请量（pinned 内存会计入）
static void printMem(const char* tag) {
    PROCESS_MEMORY_COUNTERS_EX pmc;
    std::memset(&pmc, 0, sizeof(pmc));
    if (!GetProcessMemoryInfo(GetCurrentProcess(),
                              reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc),
                              static_cast<DWORD>(sizeof(pmc)))) {
        std::fprintf(stderr, "[mem] GetProcessMemoryInfo 失败 (err=%lu)\n", GetLastError());
        return;
    }
    std::printf("[mem] %-14s 工作集=%4llu MB  私有=%4llu MB\n", tag,
                static_cast<unsigned long long>(pmc.WorkingSetSize) >> 20,
                static_cast<unsigned long long>(pmc.PrivateUsage) >> 20);
}

// 逐图推理 + 存图 + 统计。返回 0 成功。
static int runImages(DllApi& a, const std::string& engine, const std::vector<std::string>& files,
                     const std::string& outDir, bool wantMask, bool wantStat) {
    void* model = a.Create();
    if (!model) { std::fprintf(stderr, "[ERR] SegCreate 失败: %s\n", dllErr(a).c_str()); return 1; }
    if (!a.Load(engine.c_str(), model)) {
        std::fprintf(stderr, "[ERR] SegLoad 失败: %s\n", dllErr(a).c_str());
        a.Destroy(model); return 1;
    }
    std::printf("[OK] 已加载引擎: %s\n", engine.c_str());
    if (a.StatSize) {
        const int sz = a.StatSize();
        if (sz != (int)sizeof(SegStat)) {
            std::fprintf(stderr, "[ERR] SegStat 布局不匹配: DLL=%d, 本程序=%d\n",
                         sz, (int)sizeof(SegStat));
            a.Destroy(model); return 1;
        }
        std::printf("[OK] SegStat 布局自检通过 (%d 字节)\n", sz);
    }
    if (!outDir.empty()) fs::create_directories(outDir);
    if (wantMask && !outDir.empty()) fs::create_directories(fs::path(outDir) / "mask");

    double totalMs = 0.0;
    for (size_t i = 0; i < files.size(); ++i) {
        cv::Mat img = cv::imread(files[i], cv::IMREAD_COLOR);
        if (img.empty()) { std::fprintf(stderr, "  [skip] 读图失败: %s\n", files[i].c_str()); continue; }
        const std::string stem = fs::path(files[i]).stem().string();

        LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f);
        QueryPerformanceCounter(&t0);
        const int rc = a.Forward(img.data, img.cols, img.rows, (int)img.step, model);
        QueryPerformanceCounter(&t1);
        const double ms = (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart;
        totalMs += ms;

        if (rc != 0) { std::fprintf(stderr, "  [ERR] SegForward rc=%d: %s\n", rc, dllErr(a).c_str()); continue; }
        std::printf("  [%zu/%zu] %s  %.0f ms\n", i + 1, files.size(), stem.c_str(), ms);

        if (!outDir.empty()) {
            int w, h, step, ch;
            const char* ov = a.GetOverlay(w, h, step, ch, model);
            cv::Mat ovm = takeImage(a, ov, w, h, step, ch);
            if (!ovm.empty()) cv::imwrite((fs::path(outDir) / (stem + ".png")).string(), ovm);
            if (wantMask) {
                const char* mk = a.GetMask(w, h, step, ch, model);
                cv::Mat mkm = takeImage(a, mk, w, h, step, ch);
                if (!mkm.empty()) cv::imwrite((fs::path(outDir) / "mask" / (stem + ".png")).string(), mkm);
            }
        }
        if (wantStat) { SegStat s; a.GetStat(model, &s); printStat(s); }
    }
    std::printf("[done] %zu 张, 平均 %.1f ms/图\n", files.size(),
                files.empty() ? 0.0 : totalMs / files.size());
    a.Destroy(model);
    return 0;
}

// 浸泡测试：同一进程内反复 Forward -> 取图 -> 释放，并在结束时重复 SegLoad 多次。
// 判别标准：iter 10 之后各次内存快照应基本持平；reload 前后也应基本持平。
// （修复前每次 SegLoad 会漏掉整份引擎 + 约 160 MB 流水线缓冲，快照会阶梯式上涨。）
static int runSoak(DllApi& a, const std::string& engine, const std::string& input,
                   int iters, bool useMask) {
    std::vector<cv::Mat> imgs;
    for (const auto& f : listImages(input)) {
        cv::Mat m = cv::imread(f, cv::IMREAD_COLOR);
        if (!m.empty()) imgs.push_back(m);
    }
    if (imgs.empty()) {
        std::fprintf(stderr, "[ERR] soak 没有可用图片: %s\n", input.c_str());
        return 1;
    }

    const bool mask = useMask && a.ForwardMask != nullptr;
    if (useMask && !a.ForwardMask) {
        std::printf("[soak] 当前 DLL 没有 SegForwardMask 导出，回退到 SegForward\n");
    }

    void* model = a.Create();
    if (!model) {
        std::fprintf(stderr, "[ERR] SegCreate 失败: %s\n", dllErr(a).c_str());
        return 1;
    }
    if (!a.Load(engine.c_str(), model)) {
        std::fprintf(stderr, "[ERR] SegLoad 失败: %s\n", dllErr(a).c_str());
        a.Destroy(model);
        return 1;
    }
    std::printf("[soak] %d 次迭代, %zu 张图, 路径=%s\n", iters, imgs.size(),
                mask ? "SegForwardMask" : "SegForward");
    printMem("load 后");

    int errors = 0;
    for (int i = 0; i < iters; ++i) {
        const cv::Mat& img = imgs[static_cast<std::size_t>(i) % imgs.size()];
        const int rc = mask
            ? a.ForwardMask(img.data, img.cols, img.rows, static_cast<int>(img.step), model)
            : a.Forward(img.data, img.cols, img.rows, static_cast<int>(img.step), model);
        if (rc != 0) {
            std::fprintf(stderr, "[ERR] SegForward rc=%d: %s\n", rc, dllErr(a).c_str());
            ++errors;
        } else {
            int w = 0, h = 0, step = 0, ch = 0;
            const char* ov = a.GetOverlay(w, h, step, ch, model);
            if (ov) a.FreeImage(ov);
            const char* mk = a.GetMask(w, h, step, ch, model);
            if (mk) a.FreeImage(mk);
            SegStat st;
            a.GetStat(model, &st);
        }
        if (i == 9 || (i + 1) % 20 == 0 || i + 1 == iters) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "iter %d", i + 1);
            printMem(tag);
        }
    }

    printMem("reload 前");
    for (int r = 1; r <= 4; ++r) {
        if (!a.Load(engine.c_str(), model)) {
            std::fprintf(stderr, "[ERR] 重新加载失败: %s\n", dllErr(a).c_str());
            ++errors;
            break;
        }
        char tag[32];
        std::snprintf(tag, sizeof(tag), "reload %d 后", r);
        printMem(tag);
        const int rc = mask
            ? a.ForwardMask(imgs[0].data, imgs[0].cols, imgs[0].rows,
                            static_cast<int>(imgs[0].step), model)
            : a.Forward(imgs[0].data, imgs[0].cols, imgs[0].rows,
                        static_cast<int>(imgs[0].step), model);
        if (rc != 0) {
            std::fprintf(stderr, "[ERR] reload 后 Forward 失败: %s\n", dllErr(a).c_str());
            ++errors;
        }
    }

    a.Destroy(model);
    printMem("destroy 后");
    return errors == 0 ? 0 : 1;
}

// 评测：读 val_full.txt，逐图算 per-class IoU + 平均耗时
static int runEval(DllApi& a, const std::string& engine, const std::string& valFull, bool wantStat) {
    std::vector<std::pair<std::string, std::string>> lines;
    { std::ifstream in(valFull); std::string l;
      while (std::getline(in, l)) {
          if (l.empty()) continue;
          auto p = l.find('\t');
          if (p == std::string::npos) continue;
          lines.emplace_back(l.substr(0, p), l.substr(p + 1));
      } }
    if (lines.empty()) { std::fprintf(stderr, "[ERR] val_full 为空或读不到: %s\n", valFull.c_str()); return 1; }

    void* model = a.Create();
    if (!model) {
        std::fprintf(stderr, "[ERR] SegCreate 失败: %s\n", dllErr(a).c_str());
        return 1;
    }
    if (!a.Load(engine.c_str(), model)) {
        // Create 成功后 Load 失败也必须 Destroy，否则整个 Handle（含已部分加载的引擎）泄漏
        std::fprintf(stderr, "[ERR] 加载失败: %s\n", dllErr(a).c_str());
        a.Destroy(model);
        return 1;
    }

    double inter[3] = {0, 0, 0}, uni[3] = {0, 0, 0}, totalMs = 0.0;
    int counted = 0;
    for (size_t i = 0; i < lines.size(); ++i) {
        cv::Mat img = cv::imread(lines[i].first, cv::IMREAD_COLOR);
        cv::Mat gt = cv::imread(lines[i].second, cv::IMREAD_UNCHANGED);
        if (img.empty() || gt.empty()) continue;

        LARGE_INTEGER f, t0, t1; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&t0);
        const int rc = a.Forward(img.data, img.cols, img.rows, (int)img.step, model);
        QueryPerformanceCounter(&t1);
        totalMs += (t1.QuadPart - t0.QuadPart) * 1000.0 / f.QuadPart;
        if (rc != 0) continue;

        int w, h, step, ch;
        const char* mk = a.GetMask(w, h, step, ch, model);
        cv::Mat pred = takeImage(a, mk, w, h, step, ch);
        if (pred.empty()) continue;
        if (pred.size() != gt.size()) cv::resize(gt, gt, pred.size(), 0, 0, cv::INTER_NEAREST);

        for (int c = 0; c < 3; ++c) {
            cv::Mat pm = (pred == c), gm = (gt == c);
            inter[c] += cv::countNonZero(pm & gm);
            uni[c]   += cv::countNonZero(pm | gm);
        }
        if (wantStat) { SegStat s; a.GetStat(model, &s); if (i < 3) printStat(s); }
        counted++;
        if (i < 3 || (i + 1) % 20 == 0) std::printf("  [%zu/%zu]\n", i + 1, lines.size());
    }
    double iou[3], miou = 0;
    for (int c = 0; c < 3; ++c) { iou[c] = inter[c] / (uni[c] + 1e-9); miou += iou[c]; }
    miou /= 3.0;
    std::printf("\n=== seg_test --eval (%d 张) ===\n", counted);
    std::printf("background IoU=%.4f\nsilie IoU=%.4f\nfanzhe IoU=%.4f\nmIoU=%.4f\n",
                iou[0], iou[1], iou[2], miou);
    std::printf("avg %.1f ms/图\n", counted ? totalMs / counted : 0.0);
    a.Destroy(model);
    return 0;
}

int main(int argc, char** argv) {
    // 控制台按 UTF-8 渲染：本文件是 UTF-8 编码，printf 输出的是 UTF-8 字节，
    // 若不改代码页，控制台会用默认 GBK(936) 解码导致中文乱码。
    SetConsoleOutputCP(CP_UTF8);

    std::vector<std::string> pos;
    std::string dllPath, valFull, soakInput;
    bool mask = false, eval = false, stat = true, soak = false, runMask = false;
    int soakIters = 0;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--mask") mask = true;
        else if (a == "--eval") { eval = true; if (i + 1 < argc) valFull = argv[++i]; }
        else if (a == "--dll") { if (i + 1 < argc) dllPath = argv[++i]; }
        else if (a == "--no-forward-stat") stat = false;
        else if (a == "--runmask") runMask = true;   // soak 走软件实际路径（不生成叠加图）
        else if (a == "--soak") {
            soak = true;
            if (i + 1 < argc) soakInput = argv[++i];
            if (i + 1 < argc) soakIters = std::atoi(argv[++i]);
        }
        else pos.push_back(a);
    }
    if (dllPath.empty()) {
        // 默认：exe 同级目录
        char exe[4096]; GetModuleFileNameA(nullptr, exe, sizeof(exe));
        dllPath = (fs::path(exe).parent_path() / "segformer_Dll_func.dll").string();
    }
    if (pos.empty()) {
        std::printf("用法:\n"
                    "  seg_test.exe <engine> <图或目录> <输出目录> [--mask] [--dll <dll>]\n"
                    "  seg_test.exe <engine> --eval <val_full.txt> [--dll <dll>]\n"
                    "  seg_test.exe <engine> --soak <图或目录> <迭代次数> [--dll <dll>] "
                    "[--runmask]\n");
        return 2;
    }
    const std::string engine = pos[0];

    DllApi api;
    if (!api.open(dllPath)) return 1;
    std::printf("[OK] 动态加载 DLL: %s\n", dllPath.c_str());

    if (soak) {
        if (soakInput.empty() || soakIters < 1) {
            std::fprintf(stderr, "[ERR] 用法: seg_test.exe <engine> --soak <图或目录> <迭代次数>\n");
            return 2;
        }
        return runSoak(api, engine, soakInput, soakIters, runMask);
    }
    if (eval) return runEval(api, engine, valFull, stat);
    if (pos.size() < 3) { std::fprintf(stderr, "[ERR] 需要 <engine> <输入> <输出目录>\n"); return 2; }
    return runImages(api, engine, listImages(pos[1]), pos[2], mask, stat);
}
