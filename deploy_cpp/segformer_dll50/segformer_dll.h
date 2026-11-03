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

#include "seg_profile.h"

// 一次 Forward 最多回报这么多个缺陷连通域（按面积降序，超出部分丢弃最小的一些；
// 真实总数见 SegStat::ncomp）。
#define SEG_MAX_BOX 64

#ifdef SEGFORMER_DLL_EXPORTS
#define SEG_API extern "C" __declspec(dllexport)
#else
#define SEG_API extern "C" __declspec(dllimport)
#endif

struct SegStat {
    int       ng;                  // 是否有缺陷：px[1] + px[2] > 0
    int       ncomp[3];            // 各类别连通域真实总数（ncomp[0] 恒为 0，背景不计）
    int       nbox;                // 下面 box* 数组中有效的个数，<= SEG_MAX_BOX
    int       boxCls[SEG_MAX_BOX]; // 连通域类别：1=silie(撕裂) 2=fanzhe(翻折)
    int       boxX[SEG_MAX_BOX];
    int       boxY[SEG_MAX_BOX];
    int       boxW[SEG_MAX_BOX];
    int       boxH[SEG_MAX_BOX];
    long long px[3];               // 各类别像素总数：0=背景 1=silie 2=fanzhe
    long long boxPx[SEG_MAX_BOX];  // 各连通域的像素数，与 box* 一一对应
};

// ---- 生命周期 ----------------------------------------------------------
SEG_API void* SegCreate();
SEG_API void  SegDestroy(void* model);

// ---- 加载 / 推理 -------------------------------------------------------
// 加载 .engine；成功返回 true。失败时用 SegLastError() 取原因。
SEG_API bool  SegLoad(const char* enginePath, void* model);

// 整图滑窗推理。bgrData 为 BGR/24bpp/带 stride 的原始像素。
// 返回 0 表示成功，非 0 失败（原因见 SegLastError）。
SEG_API int   SegForward(unsigned char* bgrData, int width, int height, int stride, void* model);

// 同 SegForward，但只做推理，不保存原图、不生成叠加图。调用方只需要 mask / 统计时用它，
// 每帧可省掉一次原图拷贝和一次叠加图生成（约 23MB 级）。之后调用 SegGetOverlay 会报
// “no result yet”，SegGetMask / SegGetStat 正常。
SEG_API int   SegForwardMask(unsigned char* bgrData, int width, int height, int stride, void* model);

// ---- 取结果图（调用方负责 SegFreeImage）--------------------------------
// 叠加图：原图上把缺陷区域上色（与 seg_infer.exe --eval 输出、与 PyTorch 参考叠加图同一实现）
SEG_API const char* SegGetOverlay(int& width, int& height, int& step, int& channels, void* model);
// 纯语义图：单通道，像素值 0/1/2
SEG_API const char* SegGetMask(int& width, int& height, int& step, int& channels, void* model);
SEG_API void SegFreeImage(const char* imageData);

// ---- 缺陷统计 ----------------------------------------------------------
// 对最近一次 SegForward 的结果做连通域统计（按需计算并缓存，不调用就没有开销）。
// 不做任何最小面积过滤，小连通域原样报出，过滤交给调用方。
SEG_API void SegGetStat(void* model, SegStat* out);

// sizeof(SegStat)。C# 侧启动时断言 Marshal.SizeOf 与它一致，布局漂移会立刻暴露。
SEG_API int SegStatSize();

// 最近一次错误的文字说明（线程局部，无需释放）。
SEG_API const char* SegLastError();

// ---- 诊断：最近一帧的结构化耗时（默认关闭，不影响推理结果）---------------
// csvPath 目前只做保留参数（CSV 仍由 SEGFORMER_PROFILE / SEGFORMER_PROFILE_FILE 控制）；
// enable!=0 时开启 SegFrameTiming 快照。可在 SegLoad 之前调用。
SEG_API int  SegProfileConfigure(const char* csvPath, int enable);
// 取最近一帧的耗时。0=成功，1=模型未加载/无数据，2=参数错误。
// 仅应在同一次 SegForward/SegForwardMask 返回之后、同一线程上调用。
SEG_API int  SegGetLastTiming(void* model, SegFrameTiming* out);
// sizeof(SegFrameTiming)，C# 启动时用于布局自检。
SEG_API int  SegTimingSize();
// 仅诊断实验用：覆盖并发路数（1~16），下次 SegLoad 生效。
SEG_API void SegSetWorkers(int n);
