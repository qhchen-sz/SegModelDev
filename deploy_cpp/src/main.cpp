// SegFormer 语义分割 (TensorRT) 的 C++ 推理程序。
//
// 用法:
//   seg_infer.exe --model <name> --input <图或目录> --out <目录> [--mask]
//   seg_infer.exe --model <name> --eval <val_full.txt> [--out <目录>] [--limit N]
//   seg_infer.exe --model <name> --dump-logits <out.bin> --input <单张图>
//
//   <name> 解析为 models/segformer_<name>_b1_fp16_640.engine
//   （sw_s4 / v2_strat / ...）；也可用 --engine <完整路径> 直接指定。
//   两者后写的生效。
//
// 设计约束: 输出必须与 infer_fast_strat.py 逐条对齐，见 seg_pipeline.hpp 的注释。

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
// 只为 GetModuleFileNameA —— 用来定位 exe 目录，从而不依赖当前工作目录。
// CMakeLists.txt 已全局定义 WIN32_LEAN_AND_MEAN / NOMINMAX。
#include <windows.h>
#endif

#include <opencv2/opencv.hpp>

#include "seg_pipeline.hpp"
#include "trt_engine.hpp"

namespace {

const char* kUsage =
    "SegFormer 语义分割 (TensorRT) 推理\n"
    "\n"
    "  --model  <name>       模型名，解析为 models/segformer_<name>_b1_fp16_640.engine\n"
    "                        例：--model sw_s4 / v2_strat\n"
    "                        先用 tools\\build_engine.bat <name> 构建该引擎\n"
    "  --engine <path>       TensorRT .engine 完整路径\n"
    "                        (--model 与 --engine 二选一，后写的生效)\n"
    "  --input  <path>       单张图片或目录\n"
    "  --out    <dir>        输出目录：叠加图 <stem>.png；--mask 时再存 mask/<stem>.png\n"
    "  --mask                额外输出 0/1/2 语义图\n"
    "  --eval   <val_full.txt>  评测模式：逐图算 IoU，打印 summary\n"
    "  --dump-logits <out.bin>  调试：对输入图左上角单个 tile 前向一次，dump 原始\n"
    "                           logits (float32, CxHxW)，用于和 PyTorch 做数值对齐\n"
    "  --limit  <N>          只处理前 N 张（调试用）\n"
    "  --stride <N>          滑窗步长，默认 448（须与 infer_fast_strat.py 一致）\n"
    "  --batch  <N>          单次前向块数，默认 16\n"
    "  --confidence-threshold <float>  Softmax 置信度阈值 [0,1]，默认 0.5\n"
    "  --warmup <N>          预热次数，不计入耗时，默认 1\n";

// exe 所在目录；失败返回空串。用来在不依赖当前工作目录的前提下定位 models\。
std::string exeDir() {
#if defined(_WIN32)
    char buf[MAX_PATH] = {0};
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::string();
    std::string p(buf, n);
    const std::size_t s = p.find_last_of("\\/");
    return s == std::string::npos ? std::string() : p.substr(0, s);
#else
    return std::string();
#endif
}

bool fileExists(const std::string& p) {
    std::error_code ec;
    return std::filesystem::is_regular_file(p, ec);
}

// "sw_s4" -> 第一个存在的 models\segformer_sw_s4_b1_fp16_640.engine
// 依次尝试：当前工作目录、exe 目录的 ../../models（源码树里 build/Release/ 往上两级）、
// exe 目录的 ../models（若将来 exe 被直接放到 deploy_cpp\ 下）。
// 找不到返回空串。
std::string resolveModel(const std::string& model) {
    const std::string file = "segformer_" + model + "_b1_fp16_640.engine";
    std::vector<std::string> cands;
    cands.push_back((std::filesystem::path("models") / file).string());
    const std::string ed = exeDir();
    if (!ed.empty()) {
        cands.push_back((std::filesystem::path(ed) / ".." / ".." / "models" / file)
                            .lexically_normal()
                            .string());
        cands.push_back(
            (std::filesystem::path(ed) / ".." / "models" / file).lexically_normal().string());
    }
    for (const std::string& c : cands) {
        if (fileExists(c)) return c;
    }
    return std::string();
}

// 叠加图配色来自 seg_pipeline.hpp 的 segOverlay()，与 DLL 共用同一份实现。

bool hasImageExt(std::string p) {
    std::transform(p.begin(), p.end(), p.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    for (const char* e : {".jpg", ".jpeg", ".png", ".bmp"}) {
        const std::size_t n = std::strlen(e);
        if (p.size() > n && p.compare(p.size() - n, n, e) == 0) return true;
    }
    return false;
}

std::vector<std::string> listImages(const std::string& dir) {
    std::vector<std::string> out;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string p = e.path().string();
        if (hasImageExt(p)) out.push_back(p);
    }
    std::sort(out.begin(), out.end());
    return out;
}

bool readValFull(const std::string& txt,
                 std::vector<std::pair<std::string, std::string>>& out,
                 std::string& err) {
    std::ifstream f(txt);
    if (!f) {
        err = "cannot open eval list: " + txt;
        return false;
    }
    std::string line;
    while (std::getline(f, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line.empty()) continue;
        const std::size_t tab = line.find('\t');
        if (tab == std::string::npos) continue;
        out.emplace_back(line.substr(0, tab), line.substr(tab + 1));
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string enginePath, inputPath, outDir, evalList, dumpLogits;
    bool saveMask = false;
    int limit = -1;
    int warmup = 1;
    SegConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> std::string {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "[ERROR] %s needs a value\n", what);
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--engine") enginePath = next("--engine");
        else if (a == "--model") {
            // 立即解析，让「后写的生效」与 --engine 保持同一语义
            const std::string modelName = next("--model");
            enginePath = resolveModel(modelName);
            if (enginePath.empty()) {
                std::fprintf(stderr,
                             "[ERROR] no engine for model '%s'\n"
                             "        expected models\\segformer_%s_b1_fp16_640.engine\n"
                             "        build it first:  tools\\build_engine.bat %s\n",
                             modelName.c_str(), modelName.c_str(), modelName.c_str());
                return 1;
            }
        }
        else if (a == "--input") inputPath = next("--input");
        else if (a == "--out") outDir = next("--out");
        else if (a == "--eval") evalList = next("--eval");
        else if (a == "--dump-logits") dumpLogits = next("--dump-logits");
        else if (a == "--mask") saveMask = true;
        else if (a == "--limit") limit = std::stoi(next("--limit"));
        else if (a == "--stride") cfg.stride = std::stoi(next("--stride"));
        else if (a == "--batch") cfg.batch = std::stoi(next("--batch"));
        else if (a == "--confidence-threshold") {
            cfg.confidenceThreshold = std::stof(next("--confidence-threshold"));
        }
        else if (a == "--warmup") warmup = std::stoi(next("--warmup"));
        else if (a == "-h" || a == "--help") { std::printf("%s", kUsage); return 0; }
        else { std::fprintf(stderr, "[ERROR] unknown arg: %s\n\n%s", a.c_str(), kUsage); return 2; }
    }

    if (enginePath.empty()) {
        std::fprintf(stderr, "[ERROR] --model or --engine is required\n\n%s", kUsage);
        return 2;
    }

    std::string err;
    TrtEngine engine;
    if (!engine.load(enginePath, err)) {
        std::fprintf(stderr, "[ERROR] %s\n", err.c_str());
        return 1;
    }
    std::printf("engine: %s\n", enginePath.c_str());
    engine.describe();

    SegPipeline pipe;
    if (!pipe.init(engine, cfg, err)) {
        std::fprintf(stderr, "[ERROR] pipeline init: %s\n", err.c_str());
        return 1;
    }
    std::printf("  pipeline: tile=%d stride=%d batch=%d classes=%d confidence_threshold=%.4f\n\n",
                pipe.tileSize(), cfg.stride, cfg.batch, cfg.numClasses, cfg.confidenceThreshold);

    // ---- dump-logits: 单 tile 原始输出，用于和 PyTorch 做数值对齐 ----
    if (!dumpLogits.empty()) {
        if (inputPath.empty()) {
            std::fprintf(stderr, "[ERROR] --dump-logits requires --input\n");
            return 2;
        }
        cv::Mat img = cv::imread(inputPath, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "[ERROR] cannot read %s\n", inputPath.c_str());
            return 1;
        }
        std::vector<float> logits;
        if (!pipe.debugForwardTile(img, logits, err)) {
            std::fprintf(stderr, "[ERROR] %s\n", err.c_str());
            return 1;
        }
        std::ofstream o(dumpLogits, std::ios::binary);
        o.write(reinterpret_cast<const char*>(logits.data()),
                static_cast<std::streamsize>(logits.size() * sizeof(float)));
        std::printf("wrote raw logits -> %s  shape=[%d,%d,%d] float32 (%zu floats)\n",
                    dumpLogits.c_str(), cfg.numClasses, pipe.outHeight(), pipe.outWidth(),
                    logits.size());
        return 0;
    }

    // ---- 评测模式 ----
    if (!evalList.empty()) {
        std::vector<std::pair<std::string, std::string>> items;
        if (!readValFull(evalList, items, err)) {
            std::fprintf(stderr, "[ERROR] %s\n", err.c_str());
            return 1;
        }
        if (limit > 0 && static_cast<int>(items.size()) > limit) items.resize(limit);
        std::printf("eval: %zu images from %s\n", items.size(), evalList.c_str());

        if (!outDir.empty()) {
            std::filesystem::create_directories(outDir);
            if (saveMask) std::filesystem::create_directories(std::filesystem::path(outDir) / "mask");
        }

        for (int w = 0; w < warmup && !items.empty(); ++w) {
            cv::Mat img = cv::imread(items[0].first, cv::IMREAD_COLOR);
            cv::Mat labels;
            if (img.empty() || !pipe.inferImage(img, labels, err)) {
                std::fprintf(stderr, "[ERROR] warmup: %s\n", err.c_str());
                return 1;
            }
        }

        double inter[3] = {0, 0, 0};
        double uni[3] = {0, 0, 0};
        double totalMs = 0.0;
        int bad = 0;

        for (std::size_t k = 0; k < items.size(); ++k) {
            const std::string& ip = items[k].first;
            const std::string& mp = items[k].second;
            cv::Mat img = cv::imread(ip, cv::IMREAD_COLOR);
            cv::Mat gt = cv::imread(mp, cv::IMREAD_UNCHANGED);
            if (img.empty() || gt.empty()) {
                std::fprintf(stderr, "  [skip] cannot read %s\n", ip.c_str());
                ++bad;
                continue;
            }
            if (gt.channels() != 1) cv::cvtColor(gt, gt, cv::COLOR_BGR2GRAY);
            if (gt.size() != img.size()) cv::resize(gt, gt, img.size(), 0, 0, cv::INTER_NEAREST);

            cv::Mat labels;
            if (!pipe.inferImage(img, labels, err)) {
                std::fprintf(stderr, "[ERROR] %s\n", err.c_str());
                return 1;
            }
            totalMs += pipe.lastMs();

            for (int y = 0; y < labels.rows; ++y) {
                const uchar* p = labels.ptr<uchar>(y);
                const uchar* t = gt.ptr<uchar>(y);
                for (int x = 0; x < labels.cols; ++x) {
                    for (int c = 0; c < 3; ++c) {
                        const bool pp = (p[x] == c);
                        const bool tt = (t[x] == c);
                        if (pp && tt) inter[c] += 1;
                        if (pp || tt) uni[c] += 1;
                    }
                }
            }

            if (!outDir.empty()) {
                const std::string stem = std::filesystem::path(ip).stem().string();
                cv::imwrite((std::filesystem::path(outDir) / (stem + ".png")).string(),
                            segOverlay(img, labels));
                if (saveMask) {
                    cv::imwrite(
                        (std::filesystem::path(outDir) / "mask" / (stem + ".png")).string(),
                        labels);
                }
            }
            if (k < 3 || k % 10 == 0) {
                std::printf("  [%zu/%zu] %s %.0fms\n", k + 1, items.size(),
                            std::filesystem::path(ip).filename().string().c_str(), pipe.lastMs());
            }
        }

        const int n = static_cast<int>(items.size()) - bad;
        double iou[3];
        for (int c = 0; c < 3; ++c) iou[c] = inter[c] / (uni[c] + 1e-9);
        const double miou = (iou[0] + iou[1] + iou[2]) / 3.0;
        const double avgMs = n > 0 ? totalMs / n : 0.0;

        std::printf("\n=== SegFormer (TensorRT) tile=%d stride=%d batch=%d confidence_threshold=%.4f ===\n",
                    pipe.tileSize(), cfg.stride, cfg.batch, cfg.confidenceThreshold);
        std::printf("engine: %s\n", enginePath.c_str());
        std::printf("background IoU=%.4f\nsilie IoU=%.4f\nfanzhe IoU=%.4f\nmIoU=%.4f\n", iou[0],
                    iou[1], iou[2], miou);
        std::printf("avg inference = %.1f ms/image (n=%d)\n", avgMs, n);
        if (bad) std::printf("skipped %d unreadable images\n", bad);
        return 0;
    }

    // ---- 普通推理 ----
    std::vector<std::string> files;
    if (std::filesystem::is_directory(inputPath)) {
        files = listImages(inputPath);
    } else if (!inputPath.empty()) {
        files.push_back(inputPath);
    } else {
        std::fprintf(stderr, "[ERROR] need --input, --eval or --dump-logits\n\n%s", kUsage);
        return 2;
    }
    if (limit > 0 && static_cast<int>(files.size()) > limit) files.resize(limit);
    if (files.empty()) {
        std::fprintf(stderr, "[ERROR] no input images\n");
        return 1;
    }
    if (!outDir.empty()) std::filesystem::create_directories(outDir);
    if (saveMask && !outDir.empty()) {
        std::filesystem::create_directories(std::filesystem::path(outDir) / "mask");
    }

    double totalMs = 0.0;
    int ok = 0;
    for (const std::string& f : files) {
        cv::Mat img = cv::imread(f, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::fprintf(stderr, "  [skip] cannot read %s\n", f.c_str());
            continue;
        }
        cv::Mat labels;
        if (!pipe.inferImage(img, labels, err)) {
            std::fprintf(stderr, "[ERROR] %s\n", err.c_str());
            return 1;
        }
        totalMs += pipe.lastMs();
        ++ok;
        const std::string stem = std::filesystem::path(f).stem().string();
        if (!outDir.empty()) {
            cv::imwrite((std::filesystem::path(outDir) / (stem + ".png")).string(),
                        segOverlay(img, labels));
            if (saveMask) {
                cv::imwrite((std::filesystem::path(outDir) / "mask" / (stem + ".png")).string(),
                            labels);
            }
        }
        std::printf("  %-52s %4dx%-4d  %.0fms\n", std::filesystem::path(f).filename().string().c_str(),
                    img.cols, img.rows, pipe.lastMs());
    }
    if (ok) std::printf("\ndone: %d images, avg %.1f ms/image\n", ok, totalMs / ok);
    return 0;
}
