#pragma once
// segformer_Dll_func.dll —— SegFormer 语义分割模型的 C ABI 导出层。
//
// 接口约定刻意对齐 D:\CaoJiaoJie\gpu_Dll_func 的既有做法：
//   - 不透明句柄 void*，Create/Destroy 成对
//   - 图像入参是原始指针：unsigned char* + width/height/stride（BGR、24bpp、带 stride）
//   - 结果图出参是 const char*，由调用方用 SegFreeImage 释放
//   - 统计结构体用「扁平定长数组」而非嵌套结构体数组（对齐 ResultPara 的封送风格）
//
// 推理逻辑**不在这里**——复用已验证的 TrtEngine / SegPipeline。
// （本工程已自包含：trt_engine.* / seg_pipeline.* 是 deploy_cpp/src 下同名文件的副本，
//   两处需同步改；封装层本身不含任何推理逻辑。）
// sw_s4 fp16 实测 mIoU 0.7594 / 155 ms（PyTorch 基线 0.7593 / 864 ms）。

#include <cstdint>

// 一次 Forward 最多回报这么多个缺陷连通域（按面积降序，超出部分丢弃最小的一些；
// 真实总数见 SegStat::ncomp）。
#define SEG_MAX_BOX 64

#ifdef SEGFORMER_DLL_EXPORTS
#define SEG_API extern "C" __declspec(dllexport)
#else
#define SEG_API extern "C" __declspec(dllimport)
#endif

struct SegStat {
    int       ng;                  // 是否有缺陷：任一前景类（1..numClasses-1）像素 > 0
    int       nbox;                // 下面 box* 数组中有效的个数，<= SEG_MAX_BOX
    int       boxCls[SEG_MAX_BOX]; // 连通域类别 id：0=背景，1..numClasses-1=前景类（含义由模型 id2label 决定）
    int       boxX[SEG_MAX_BOX];
    int       boxY[SEG_MAX_BOX];
    int       boxW[SEG_MAX_BOX];
    int       boxH[SEG_MAX_BOX];
    long long boxPx[SEG_MAX_BOX];  // 各连通域的像素数，与 box* 一一对应
};

// ---- 生命周期 ----------------------------------------------------------
SEG_API void* SegCreate();
SEG_API void  SegDestroy(void* model);

// ---- 加载 / 推理 -------------------------------------------------------
// 加载 .engine；成功返回 true。失败时用 SegLastError() 取原因。
// 类别数以引擎输出通道数为准（>=2 即可），同一 DLL 兼容任意类别数的分割模型。
SEG_API bool  SegLoad(const char* enginePath, void* model);

// 设置 Softmax 最大置信度阈值 [0,1]，默认 0.5；低于阈值判为背景，等于阈值保留。
// 可在 SegLoad 前或后调用；成功返回 true，非法阈值返回 false，原因见 SegLastError。
// 注意：二分类模型下阈值 0.5 等价于纯 argmax（不过滤），压假阳需设 > 0.5。
SEG_API bool  SegSetConfidenceThreshold(void* model, float threshold);

// 整图滑窗推理。bgrData 为 BGR/24bpp/带 stride 的原始像素。
// 返回 0 表示成功，非 0 失败（原因见 SegLastError）。
SEG_API int   SegForward(unsigned char* bgrData, int width, int height, int stride, void* model);

// 同 SegForward，但只做推理，不保存原图、不生成叠加图。调用方只需要 mask / 统计时用它，
// 每帧可省掉一次原图拷贝和一次叠加图生成（约 23MB 级）。
// 注意：与 50 系列行为保持一致——不主动清空上一帧的 lastInput。
SEG_API int   SegForwardMask(unsigned char* bgrData, int width, int height, int stride, void* model);

// ---- 取结果图（调用方负责 SegFreeImage）--------------------------------
// 叠加图：原图上把缺陷区域上色（与 seg_infer.exe --eval 输出同一实现）
SEG_API const char* SegGetOverlay(int& width, int& height, int& step, int& channels, void* model);
// 纯语义图：单通道，像素值 0..numClasses-1（0=背景）
SEG_API const char* SegGetMask(int& width, int& height, int& step, int& channels, void* model);
SEG_API void SegFreeImage(const char* imageData);

// ---- 缺陷统计 ----------------------------------------------------------
// 引擎类别数（SegLoad 成功后可用，未加载返回 0）。
// 用于让调用方按实际长度分配下面 SegGetStat 的 px / ncomp 数组。
SEG_API int SegGetNumClasses(void* model);

// 对最近一次 SegForward 的结果做连通域统计（按需计算并缓存，不调用就没有开销）。
// 不做任何最小面积过滤，小连通域原样报出，过滤交给调用方。
// px / ncomp 由调用方分配并通过 pxCap / ncompCap 传容量，实际填充 min(cap, 类别数) 项；
// 传 nullptr 则只取定长部分。按类别的统计不进 SegStat 结构体，类别数变化不影响 ABI
// （与 YOLO 部署 ResultPara 的设计一致）。
SEG_API void SegGetStat(void* model, SegStat* out,
                        long long* px, int pxCap, int* ncomp, int ncompCap);

// sizeof(SegStat)。C# 侧启动时断言 Marshal.SizeOf 与它一致，布局漂移会立刻暴露。
SEG_API int SegStatSize();

// 最近一次错误的文字说明（线程局部，无需释放）。
SEG_API const char* SegLastError();
