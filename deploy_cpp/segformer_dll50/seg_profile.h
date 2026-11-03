#pragma once
// 最近一帧的原生耗时快照。C 兼容，供 DLL 导出层与 C# 封装共用。
//
// 所有段都是纳秒；[N] 数组是 per-worker 的并行区间，不能直接相加。
// 字段顺序必须与 C# EVDll_Segformer.SegFrameTiming 严格一致，
// 启动时用 SegTimingSize() 与 Marshal.SizeOf 做布局自检。

#include <cstdint>

#define SEG_TIMING_MAX_WORKERS 16
#define SEG_TIMING_VERSION 1

typedef struct SegFrameTiming {
    int32_t version;
    int32_t size;

    int64_t frameSeq;
    int64_t totalNs;      // SegForward / SegForwardMask 入口到返回
    int64_t collectNs;    // collectTiles
    int64_t clearNs;      // acc_ setTo(0)
    int64_t dispatchNs;   // 主线程加锁派发到 notify_all
    int64_t waitNs;       // 主线程等所有 worker 完成
    int64_t accNs;        // 重叠累加
    int64_t argmaxNs;
    int64_t maskCopyNs;   // 结果图拷贝（由调用方取图时填写，可为 0）

    // per-worker 并行区间（不是总和）
    int64_t prepNs[SEG_TIMING_MAX_WORKERS];
    int64_t gpuWallNs[SEG_TIMING_MAX_WORKERS];      // CPU 观测：H2D→sync 全段
    int64_t inferEnqueueNs[SEG_TIMING_MAX_WORKERS]; // inferOn 宿主耗时（含内部 sync）
    int64_t syncNs[SEG_TIMING_MAX_WORKERS];         // 末次 cudaStreamSynchronize 阻塞
    int64_t h2dGpuNs[SEG_TIMING_MAX_WORKERS];       // CUDA event 实测
    int64_t inferGpuNs[SEG_TIMING_MAX_WORKERS];     // CUDA event 实测：纯推理
    int64_t d2hGpuNs[SEG_TIMING_MAX_WORKERS];       // CUDA event 实测
    int64_t upNs[SEG_TIMING_MAX_WORKERS];
    int64_t wakeNs[SEG_TIMING_MAX_WORKERS];         // notify → worker 开始处理
    int64_t tailNs[SEG_TIMING_MAX_WORKERS];         // 末块完成 → worker 退出

    int32_t tileCount[SEG_TIMING_MAX_WORKERS];

    int32_t tiles;
    int32_t nWorkers;
} SegFrameTiming;
