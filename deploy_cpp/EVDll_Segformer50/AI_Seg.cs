// EVDll_Segformer —— SegFormer 语义分割模型的 C# 调用层。
//
// 与 D:\CaoJiaoJie\EVDll\AI_Gpu.cs 里的 AI_Gpu1 同构：
//   软件(Halcon 脚本) → 本 DLL → segformer_Dll_func.dll → TensorRT
//
// 用法（与 YOLO 版对照）：
//     using EVDll_Segformer;
//     AI_Seg1.Init(@"...\segformer_sw_s4_b1_fp16_640.engine", ref erro);
//     var image = AI_Seg1.Run(GetBitMap("采集图像.图像"), ref erro);
//     SetBitMap(image, "采集图像.图像");
//
//     var st = AI_Seg1.GetStat();     // 缺陷判定：st.ng / st.px[] / st.box*
//
// Softmax 置信度阈值默认 0.5，可在 Init 后通过 AI_SegN.SetConfidenceThreshold
// 或 SegModel.SetConfidenceThreshold 调整；Run 仍不带 (conf, nms) 参数，不做 NMS。

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace EVDll_Segformer
{
    /// <summary>segformer_Dll_func.dll 的导出声明。</summary>
    internal static class Native
    {
        private const string Dll = "segformer_Dll_func.dll";

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SegCreate();

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void SegDestroy(IntPtr model);

        // C++ 的 bool 是 1 字节；不加 I1 的话 .NET 默认按 4 字节 Win32 BOOL 封送，返回值不可靠。
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool SegLoad(string enginePath, IntPtr model);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool SegSetConfidenceThreshold(IntPtr model, float threshold);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegForward(IntPtr bgrData, int width, int height, int stride,
                                            IntPtr model);

        // 只推理、不生成叠加图；只取 mask / 统计时用它可省掉每帧一次原图拷贝和叠加图生成。
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegForwardMask(IntPtr bgrData, int width, int height, int stride,
                                                IntPtr model);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SegGetOverlay(ref int width, ref int height, ref int step,
                                                  ref int channels, IntPtr model);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SegGetMask(ref int width, ref int height, ref int step,
                                               ref int channels, IntPtr model);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void SegFreeImage(IntPtr imageData);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern void SegGetStat(IntPtr model, ref SegStat stat);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegStatSize();

        // 诊断用；旧版原生 DLL 可能没有这两个导出，调用方需捕获 EntryPointNotFoundException。
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegGetLastTiming(IntPtr model, ref SegFrameTiming timing);

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegTimingSize();

        // 诊断：开启结构化计时（C# 侧 SegTrace 启用时调用）。旧版 DLL 没有此导出。
        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern int SegProfileConfigure(string csvPath, int enable);

        [DllImport("kernel32.dll")]
        public static extern int GetCurrentThreadId();

        [DllImport(Dll, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr SegLastError();
    }

    /// <summary>
    /// 缺陷统计。字段顺序/类型必须与 segformer_dll.h 的 SegStat 严格一致。
    /// 刻意用扁平定长数组（对齐 EVDll 里 ResultPara 的封送风格），避免嵌套结构体的坑。
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct SegStat
    {
        public const int SegMaxBox = 64;

        public int ng;                                          // 是否有缺陷
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public int[] ncomp;                                     // 各类别连通域真实总数（[0] 恒为 0）
        public int nbox;                                        // 下面数组中有效的个数
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public int[] boxCls;                                    // 1=silie(撕裂) 2=fanzhe(翻折)
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public int[] boxX;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public int[] boxY;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public int[] boxW;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public int[] boxH;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 3)]
        public long[] px;                                       // 各类别像素总数
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = SegMaxBox)]
        public long[] boxPx;                                    // 各连通域像素数
    }

    /// <summary>
    /// 最近一帧的原生耗时快照，字段顺序/类型必须与 segformer_dll.h 的 SegFrameTiming 一致。
    /// 所有 [16] 数组是 per-worker 的并行区间，不能直接相加。
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 8)]
    public struct SegFrameTiming
    {
        public const int MaxWorkers = 16;

        public int version;
        public int size;

        public long frameSeq;
        public long totalNs;
        public long collectNs;
        public long clearNs;
        public long dispatchNs;
        public long waitNs;
        public long accNs;
        public long argmaxNs;
        public long maskCopyNs;

        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] prepNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] gpuWallNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] inferEnqueueNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] syncNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] h2dGpuNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] inferGpuNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] d2hGpuNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] upNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] wakeNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public long[] tailNs;
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = MaxWorkers)]
        public int[] tileCount;

        public int tiles;
        public int nWorkers;
    }

    /// <summary>
    /// 封装层自带的时间线日志，脚本无需任何改动，由本 DLL 自己落盘。
    /// 启用方式（二选一）：
    ///   1) 环境变量 EVDLL_SEG_TRACE_FILE = 输出路径；
    ///   2) 存在标记文件 D:\trace\ev_trace.on（或 %TEMP%\ev_trace.on），第一行可写输出路径，
    ///      只写空行则输出到 %TEMP%\ev_trace_&lt;pid&gt;.csv。
    /// 关闭时只多一次 bool 判断，生产路径无感。
    /// </summary>
    internal static class SegTrace
    {
        private const int FlushEvery = 32;
        private const int MaxRows = 200000;   // 约 20MB；防止现场忘记关闭导致文件无限增长
        private static readonly object Lock = new object();
        private static readonly double MsPerTick = 1000.0 / Stopwatch.Frequency;
        private static readonly System.Globalization.CultureInfo Inv =
            System.Globalization.CultureInfo.InvariantCulture;
        private static readonly Process Proc = Process.GetCurrentProcess();
        private static readonly List<string> Pending = new List<string>();

        private static bool _active;
        private static string _path = "";
        private static StreamWriter _writer;
        private static long _lastEnd;
        private static int _frame;
        private static int _seq;
        private static string _lastError = "";
        private static bool _timingUnavailable;

        internal static bool Enabled { get { return _active; } }
        internal static int CurrentFrame { get { return _frame; } }
        internal static string LastError { get { return _lastError; } }

        static SegTrace()
        {
            try
            {
                string target = ResolveTarget();
                if (target != null) Open(target);
            }
            catch (Exception e)
            {
                _lastError = e.Message;
                _active = false;
            }
            AppDomain.CurrentDomain.ProcessExit += delegate { Flush(true); Close(); };
        }

        /// <summary>返回 null=未启用；空字符串=启用并使用默认路径；其它=启用并写入该路径。</summary>
        private static string ResolveTarget()
        {
            string env = Environment.GetEnvironmentVariable("EVDLL_SEG_TRACE_FILE");
            if (!string.IsNullOrEmpty(env)) return env;

            string[] markers = { @"D:\trace\ev_trace.on", Path.Combine(Path.GetTempPath(), "ev_trace.on") };
            foreach (string m in markers)
            {
                if (!File.Exists(m)) continue;
                using (StreamReader r = new StreamReader(m))
                {
                    string line;
                    while ((line = r.ReadLine()) != null)
                    {
                        line = line.Trim();
                        if (line.Length > 0) return line;
                    }
                }
                return "";
            }
            return null;
        }

        private static void Open(string target)
        {
            string def = Path.Combine(Path.GetTempPath(), "ev_trace_" + Proc.Id + ".csv");
            string[] candidates = string.IsNullOrEmpty(target)
                ? new string[] { def }
                : new string[] { target, def,
                                 Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                                              "ev_trace_" + Proc.Id + ".csv") };

            foreach (string p in candidates)
            {
                try
                {
                    string dir = Path.GetDirectoryName(p);
                    if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
                    _writer = new StreamWriter(p, false, new UTF8Encoding(false));
                    _writer.WriteLine("frame,event,wall,threadId,seq,durMs,gapMs,srcW,srcH,srcFmt,to24," +
                                      "gc0,gc1,gc2,heapMB,procCpuMs,note");
                    _writer.Flush();
                    _path = p;
                    _active = true;

                    int nativeSize = -1;
                    try { nativeSize = Native.SegTimingSize(); }
                    catch (EntryPointNotFoundException) { }
                    catch (DllNotFoundException) { }
                    EmitMetaRow(-1, 0, 0, "", false,
                                "traceStart;file=" + p +
                                ";timingManaged=" + Marshal.SizeOf(typeof(SegFrameTiming)) +
                                ";timingNative=" + nativeSize);
                    return;
                }
                catch (Exception e)
                {
                    _lastError = p + " : " + e.Message;
                }
            }
            _active = false;
        }

        internal static long Stamp()
        {
            return _active ? Stopwatch.GetTimestamp() : 0L;
        }

        internal static int BeginFrame()
        {
            if (!_active) return 0;
            lock (Lock) { return ++_frame; }
        }

        /// <summary>记录一个叶子时间段；start/end 为 Stopwatch 时间戳。诊断代码永不抛异常。</summary>
        internal static void Emit(int frame, string ev, long start, long end)
        {
            if (!_active || start == 0 || end < start) return;
            try
            {
                lock (Lock)
                {
                    if (!_active) return;
                    double dur = (end - start) * MsPerTick;
                    double gap = _lastEnd == 0 ? 0 : (start - _lastEnd) * MsPerTick;
                    _lastEnd = end;
                    string[] f = {
                        frame.ToString(Inv), ev, Wall(), ThreadId(),
                        (++_seq).ToString(Inv),
                        dur.ToString("F3", Inv), gap.ToString("F3", Inv),
                        "", "", "", "", "", "", "", "", "", ""
                    };
                    Add(string.Join(",", f));
                }
            }
            catch (Exception e)
            {
                _lastError = e.Message;
            }
        }

        /// <summary>在 RunMask 结束后调用：记录输入图信息、GC/进程指标与原生分段耗时。</summary>
        internal static void EmitMeta(int frame, Bitmap src, bool to24, IntPtr handle)
        {
            if (!_active || frame == 0) return;
            try
            {
                EmitMetaCore(frame, src, to24, handle);
            }
            catch (Exception e)
            {
                _lastError = e.Message;
            }
        }

        private static void EmitMetaCore(int frame, Bitmap src, bool to24, IntPtr handle)
        {
            long t0 = Stamp();
            string note = "to24=" + (to24 ? "1" : "0");
            SegFrameTiming t = NewTiming();
            if (handle != IntPtr.Zero && TryGetTiming(handle, ref t))
            {
                long h2d = 0, infer = 0, d2h = 0, sync = 0, enq = 0, prep = 0, up = 0, wall = 0;
                long wakeMax = 0, tailMax = 0;
                int nw = t.nWorkers < 0 ? 0 : (t.nWorkers > SegFrameTiming.MaxWorkers
                                                   ? SegFrameTiming.MaxWorkers : t.nWorkers);
                for (int i = 0; i < nw; ++i)
                {
                    h2d += t.h2dGpuNs[i];
                    infer += t.inferGpuNs[i];
                    d2h += t.d2hGpuNs[i];
                    sync += t.syncNs[i];
                    enq += t.inferEnqueueNs[i];
                    prep += t.prepNs[i];
                    up += t.upNs[i];
                    wall += t.gpuWallNs[i];
                    if (t.wakeNs[i] > wakeMax) wakeMax = t.wakeNs[i];
                    if (t.tailNs[i] > tailMax) tailMax = t.tailNs[i];
                }

                note += ";nativeTotal=" + Ms(t.totalNs)
                      + ";nativeCollect=" + Ms(t.collectNs)
                      + ";nativeClear=" + Ms(t.clearNs)
                      + ";nativeDispatch=" + Ms(t.dispatchNs)
                      + ";nativeWait=" + Ms(t.waitNs)
                      + ";nativeAcc=" + Ms(t.accNs)
                      + ";nativeArgmax=" + Ms(t.argmaxNs)
                      + ";nativeTiles=" + t.tiles
                      + ";nativeWorkers=" + t.nWorkers
                      // 以下 *Sum 是各 worker 并行区间之和（GPU 忙碌总时间），不是墙钟
                      + ";gpuInferSum=" + Ms(infer)
                      + ";gpuH2dSum=" + Ms(h2d)
                      + ";gpuD2hSum=" + Ms(d2h)
                      + ";gpuWallSum=" + Ms(wall)
                      + ";inferEnqSum=" + Ms(enq)
                      + ";syncSum=" + Ms(sync)
                      + ";prepSum=" + Ms(prep)
                      + ";upSum=" + Ms(up)
                      + ";wakeMax=" + Ms(wakeMax)
                      + ";tailMax=" + Ms(tailMax);
            }

            lock (Lock)
            {
                if (!_active) return;
                EmitMetaRow(frame, src == null ? 0 : src.Width, src == null ? 0 : src.Height,
                            src == null ? "" : src.PixelFormat.ToString(), to24, note);
            }

            // 把诊断自身耗时也作为一段时间记下来，保证“各段 + 空档 = 帧间隔”仍然成立。
            long t1 = Stamp();
            Emit(frame, "trace.meta", t0, t1);
        }

        private static SegFrameTiming NewTiming()
        {
            SegFrameTiming t = new SegFrameTiming();
            t.prepNs = new long[SegFrameTiming.MaxWorkers];
            t.gpuWallNs = new long[SegFrameTiming.MaxWorkers];
            t.inferEnqueueNs = new long[SegFrameTiming.MaxWorkers];
            t.syncNs = new long[SegFrameTiming.MaxWorkers];
            t.h2dGpuNs = new long[SegFrameTiming.MaxWorkers];
            t.inferGpuNs = new long[SegFrameTiming.MaxWorkers];
            t.d2hGpuNs = new long[SegFrameTiming.MaxWorkers];
            t.upNs = new long[SegFrameTiming.MaxWorkers];
            t.wakeNs = new long[SegFrameTiming.MaxWorkers];
            t.tailNs = new long[SegFrameTiming.MaxWorkers];
            t.tileCount = new int[SegFrameTiming.MaxWorkers];
            return t;
        }

        private static void EmitMetaRow(int frame, int w, int h, string fmt, bool to24, string note)
        {
            int gc0 = GC.CollectionCount(0);
            int gc1 = GC.CollectionCount(1);
            int gc2 = GC.CollectionCount(2);
            double heap = GC.GetTotalMemory(false) / (1024.0 * 1024.0);
            double cpu = Proc.TotalProcessorTime.TotalMilliseconds;

            string[] f = {
                frame.ToString(Inv), "meta", Wall(), ThreadId(), (++_seq).ToString(Inv),
                "0.000", "0.000",
                w.ToString(Inv), h.ToString(Inv), fmt, to24 ? "1" : "0",
                gc0.ToString(Inv), gc1.ToString(Inv), gc2.ToString(Inv),
                heap.ToString("F1", Inv), cpu.ToString("F1", Inv), Csv(note)
            };
            Add(string.Join(",", f));
        }

        private static void Add(string row)
        {
            if (!_active) return;
            Pending.Add(row);
            if (_seq >= MaxRows)
            {
                Pending.Add(FinalRow());
                _active = false;
                Flush(true);
                Close();
                return;
            }
            if (Pending.Count >= FlushEvery) Flush(false);
        }

        private static string FinalRow()
        {
            string[] f = {
                "-1", "meta", Wall(), ThreadId(), (++_seq).ToString(Inv),
                "0.000", "0.000", "", "", "", "", "", "", "", "", "",
                Csv("traceStop;maxRows=" + MaxRows)
            };
            return string.Join(",", f);
        }

        private static void Close()
        {
            lock (Lock)
            {
                try
                {
                    if (_writer != null)
                    {
                        _writer.Flush();
                        _writer.Dispose();
                        _writer = null;
                    }
                }
                catch (Exception e)
                {
                    _lastError = e.Message;
                }
                _active = false;
                Pending.Clear();
            }
        }

        private static void Flush(bool force)
        {
            lock (Lock)
            {
                if (_writer == null || Pending.Count == 0) return;
                try
                {
                    for (int i = 0; i < Pending.Count; ++i) _writer.WriteLine(Pending[i]);
                    _writer.Flush();
                    Pending.Clear();
                }
                catch (Exception e)
                {
                    _lastError = "write " + _path + " : " + e.Message;
                    _active = false;
                    Pending.Clear();
                }
            }
        }

        private static bool TryGetTiming(IntPtr handle, ref SegFrameTiming t)
        {
            if (_timingUnavailable) return false;
            try
            {
                return Native.SegGetLastTiming(handle, ref t) == 0;
            }
            catch (EntryPointNotFoundException) { _timingUnavailable = true; return false; }
            catch (DllNotFoundException) { _timingUnavailable = true; return false; }
        }

        private static string Ms(long ns)
        {
            return (ns / 1e6).ToString("F3", Inv);
        }

        private static string Wall()
        {
            return DateTime.Now.ToString("HH:mm:ss.fff", Inv);
        }

        private static string ThreadId()
        {
            return Native.GetCurrentThreadId().ToString(Inv);
        }

        private static string Csv(string s)
        {
            if (string.IsNullOrEmpty(s)) return "";
            if (s.IndexOf(',') < 0 && s.IndexOf('"') < 0 &&
                s.IndexOf('\n') < 0 && s.IndexOf('\r') < 0) return s;
            return "\"" + s.Replace("\"", "\"\"") + "\"";
        }
    }

    /// <summary>一个模型实例。可同时 new 多个（每个各自持有引擎与 CUDA 上下文）。</summary>
    public sealed class SegModel : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

        // 取结果图时复用，避免每帧把几十 MB 的 byte[] 分配进 LOH（3000x2600 的图
        // overlay 约 23MB、mask 约 7.8MB）。数组只增不减，尺寸变化时按需扩容。
        private byte[] _overlayBuf;
        private byte[] _maskBuf;

        public SegModel() { _handle = IntPtr.Zero; }

        public SegModel(string enginePath) { Init(enginePath); }

        /// <summary>最近一次底层调用失败的原因。</summary>
        public static string LastError()
        {
            IntPtr p = Native.SegLastError();
            return p == IntPtr.Zero ? "" : (Marshal.PtrToStringAnsi(p) ?? "");
        }

        /// <summary>底层 SegStat 的字节数，供布局自检。</summary>
        public static int NativeStatSize() { return Native.SegStatSize(); }

        public bool IsOpen { get { return _handle != IntPtr.Zero; } }

        /// <summary>设置 Softmax 最大置信度阈值 [0,1]，默认 0.5；低于阈值判为背景。</summary>
        public void SetConfidenceThreshold(float threshold)
        {
            if (_disposed) throw new ObjectDisposedException("SegModel");
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            if (!Native.SegSetConfidenceThreshold(_handle, threshold))
                throw new ArgumentException("置信度阈值必须是 [0,1] 内的有限数：" + LastError());
        }

        public void Init(string enginePath)
        {
            if (_handle != IntPtr.Zero) return;
            if (!File.Exists(enginePath)) throw new FileNotFoundException("模型路径不正确：" + enginePath);

            IntPtr h = Native.SegCreate();
            if (h == IntPtr.Zero) throw new InvalidOperationException("SegCreate 失败：" + LastError());
            if (SegTrace.Enabled) EnableNativeTiming();
            try
            {
                if (!Native.SegLoad(enginePath, h))
                    throw new InvalidOperationException("加载引擎失败：" + LastError());
            }
            catch
            {
                Native.SegDestroy(h);
                throw;
            }
            _handle = h;
        }

        // 旧版原生 DLL 没有 SegProfileConfigure 时不报错，只是拿不到原生分段。
        private static void EnableNativeTiming()
        {
            try { Native.SegProfileConfigure(null, 1); }
            catch { }   // 诊断功能任何异常都不能影响 native handle 的创建/释放
        }

        /// <summary>整图滑窗推理，返回叠加图（原图上色标出缺陷）。</summary>
        public Bitmap Run(Bitmap bitmap)
        {
            if (_disposed) throw new ObjectDisposedException("SegModel");
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            if (bitmap == null || bitmap.Width <= 0 || bitmap.Height <= 0)
                throw new ArgumentException("传入图片为空，请检查传入图片是否正确");

            int frame = SegTrace.BeginFrame();
            long t0 = SegTrace.Stamp();
            bool to24 = bitmap.PixelFormat != PixelFormat.Format24bppRgb;
            Bitmap input = to24 ? To24bppRgb(bitmap) : bitmap;
            long t1 = SegTrace.Stamp();
            if (to24) SegTrace.Emit(frame, "run.to24", t0, t1);

            try
            {
                // 直接锁定位图并把 Scan0 交给 native：省掉每帧 23MB 的托管 byte[] 分配、
                // 一次整幅拷贝和一个 Pinned GCHandle。SegForward 返回前会把像素拷进
                // lastInput，因此 UnlockBits 之后原数据失效没有影响。
                long t2 = SegTrace.Stamp();
                BitmapData bd = input.LockBits(new Rectangle(0, 0, input.Width, input.Height),
                                               ImageLockMode.ReadOnly,
                                               PixelFormat.Format24bppRgb);
                long t3 = SegTrace.Stamp();
                SegTrace.Emit(frame, "run.lock", t2, t3);
                try
                {
                    long t4 = SegTrace.Stamp();
                    int rc = Native.SegForward(bd.Scan0, input.Width, input.Height, bd.Stride,
                                               _handle);
                    long t5 = SegTrace.Stamp();
                    SegTrace.Emit(frame, "run.call", t4, t5);
                    if (rc != 0)
                        throw new InvalidOperationException("SegForward 返回 " + rc + "：" + LastError());
                }
                finally
                {
                    long t6 = SegTrace.Stamp();
                    input.UnlockBits(bd);
                    long t7 = SegTrace.Stamp();
                    SegTrace.Emit(frame, "run.unlock", t6, t7);
                }
            }
            finally
            {
                if (!ReferenceEquals(input, bitmap)) input.Dispose();
            }

            SegTrace.EmitMeta(frame, bitmap, to24, _handle);
            return FetchImage(false);
        }

        /// <summary>
        /// 只推理、不生成叠加图。只需要 mask / 统计时用它，省掉每帧一次原图拷贝、
        /// 一次叠加图生成（约 23MB）以及叠加图 Bitmap 的分配/回收。
        /// 之后调用 GetOverlay 会失败（没有原图），GetMask / GetStat 正常。
        /// </summary>
        public bool RunMask(Bitmap bitmap)
        {
            if (_disposed) throw new ObjectDisposedException("SegModel");
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            if (bitmap == null || bitmap.Width <= 0 || bitmap.Height <= 0)
                throw new ArgumentException("传入图片为空，请检查传入图片是否正确");

            int frame = SegTrace.BeginFrame();
            long t0 = SegTrace.Stamp();
            bool to24 = bitmap.PixelFormat != PixelFormat.Format24bppRgb;
            Bitmap input = to24 ? To24bppRgb(bitmap) : bitmap;
            long t1 = SegTrace.Stamp();
            if (to24) SegTrace.Emit(frame, "runmask.to24", t0, t1);

            try
            {
                // 与 Run 相同的零拷贝路径：把 LockBits 的 Scan0 直接交给 native。
                long t2 = SegTrace.Stamp();
                BitmapData bd = input.LockBits(new Rectangle(0, 0, input.Width, input.Height),
                                               ImageLockMode.ReadOnly,
                                               PixelFormat.Format24bppRgb);
                long t3 = SegTrace.Stamp();
                SegTrace.Emit(frame, "runmask.lock", t2, t3);
                try
                {
                    long t4 = SegTrace.Stamp();
                    int rc = Native.SegForwardMask(bd.Scan0, input.Width, input.Height, bd.Stride,
                                                   _handle);
                    long t5 = SegTrace.Stamp();
                    SegTrace.Emit(frame, "runmask.call", t4, t5);
                    if (rc != 0)
                        throw new InvalidOperationException("SegForwardMask 返回 " + rc + "：" + LastError());
                }
                finally
                {
                    long t6 = SegTrace.Stamp();
                    input.UnlockBits(bd);
                    long t7 = SegTrace.Stamp();
                    SegTrace.Emit(frame, "runmask.unlock", t6, t7);
                }
            }
            finally
            {
                if (!ReferenceEquals(input, bitmap)) input.Dispose();
            }

            SegTrace.EmitMeta(frame, bitmap, to24, _handle);
            return true;
        }

        /// <summary>取最近一次推理的纯语义图：单通道，像素值 0/1/2。</summary>
        public Bitmap GetMask()
        {
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            return FetchImage(true);
        }

        /// <summary>取最近一次推理的缺陷统计（原样返回，不做最小面积过滤）。</summary>
        public SegStat GetStat()
        {
            SegStat st = new SegStat();
            if (_handle == IntPtr.Zero) return st;
            int frame = SegTrace.CurrentFrame;
            long t0 = SegTrace.Stamp();
            Native.SegGetStat(_handle, ref st);
            long t1 = SegTrace.Stamp();
            SegTrace.Emit(frame, "getstat.call", t0, t1);
            return st;
        }

        // 兜底：调用方忘了 Dispose 时，由 GC 终结器释放 native 引擎
        //（显存 + pinned 主机内存 + CUDA stream）。这些资源对 GC 不可见，
        // 没有这层兜底就是进程级泄漏。
        ~SegModel()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }

        private void Dispose(bool disposing)
        {
            if (_disposed) return;
            _disposed = true;

            IntPtr h = _handle;
            _handle = IntPtr.Zero;
            if (h == IntPtr.Zero) return;

            // 进程关闭阶段 DLL 可能正在卸载，不要再进 native 析构，交给 OS 回收。
            if (!disposing && Environment.HasShutdownStarted) return;

            if (disposing)
            {
                Native.SegDestroy(h);
                return;
            }
            try
            {
                Native.SegDestroy(h);
            }
            catch
            {
                // 终结器线程上不允许异常逃逸
            }
        }

        // ---- 内部 ----

        private Bitmap FetchImage(bool wantMask)
        {
            int frame = SegTrace.CurrentFrame;
            // 字面量三元选择返回 interned 字符串；trace 关闭时不产生任何字符串分配。
            string evNative = wantMask ? "getmask.native" : "getoverlay.native";
            string evMarshal = wantMask ? "getmask.marshal" : "getoverlay.marshal";
            string evNewBmp = wantMask ? "getmask.newbmp" : "getoverlay.newbmp";
            string evRowCopy = wantMask ? "getmask.rowcopy" : "getoverlay.rowcopy";
            string evFree = wantMask ? "getmask.free" : "getoverlay.free";
            int w = 0, h = 0, step = 0, ch = 0;

            long t0 = SegTrace.Stamp();
            IntPtr ptr = wantMask
                ? Native.SegGetMask(ref w, ref h, ref step, ref ch, _handle)
                : Native.SegGetOverlay(ref w, ref h, ref step, ref ch, _handle);
            long t1 = SegTrace.Stamp();
            SegTrace.Emit(frame, evNative, t0, t1);
            if (ptr == IntPtr.Zero) throw new InvalidOperationException("取结果图失败：" + LastError());

            try
            {
                // 复用缓冲：尺寸不变时不再每帧 new byte[]（23MB / 7.8MB 进 LOH 会引发随机 GC 抖动）
                long t2 = SegTrace.Stamp();
                int need = step * h;
                byte[] data = wantMask ? _maskBuf : _overlayBuf;
                if (data == null || data.Length < need)
                {
                    data = new byte[need];
                    if (wantMask) _maskBuf = data; else _overlayBuf = data;
                }
                Marshal.Copy(ptr, data, 0, need);
                long t3 = SegTrace.Stamp();
                SegTrace.Emit(frame, evMarshal, t2, t3);

                long t4 = SegTrace.Stamp();
                PixelFormat fmt = (ch == 1) ? PixelFormat.Format8bppIndexed : PixelFormat.Format24bppRgb;
                Bitmap bmp = new Bitmap(w, h, fmt);
                long t5 = SegTrace.Stamp();
                SegTrace.Emit(frame, evNewBmp, t4, t5);
                try
                {
                    long t6 = SegTrace.Stamp();
                    if (ch == 1) ApplySegPalette(bmp);
                    BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, fmt);
                    try
                    {
                        for (int y = 0; y < h; y++)
                            Marshal.Copy(data, y * step, bd.Scan0 + y * bd.Stride, step);
                    }
                    finally { bmp.UnlockBits(bd); }
                    long t7 = SegTrace.Stamp();
                    SegTrace.Emit(frame, evRowCopy, t6, t7);
                    return bmp;
                }
                catch
                {
                    bmp.Dispose();
                    throw;
                }
            }
            finally
            {
                long t8 = SegTrace.Stamp();
                Native.SegFreeImage(ptr);
                long t9 = SegTrace.Stamp();
                SegTrace.Emit(frame, evFree, t8, t9);
            }
        }

        // 掩膜图的像素值仍是原始 0/1/2（Halcon 侧可直接读取），调色板只是让它看起来可解释
        private static void ApplySegPalette(Bitmap bmp)
        {
            ColorPalette pal = bmp.Palette;
            for (int i = 0; i < pal.Entries.Length; i++) pal.Entries[i] = Color.Black;
            pal.Entries[0] = Color.Black;   // 背景
            pal.Entries[1] = Color.Red;     // silie 撕裂
            pal.Entries[2] = Color.Blue;    // fanzhe 翻折
            bmp.Palette = pal;
        }

        private static Bitmap To24bppRgb(Bitmap src)
        {
            Bitmap dst = new Bitmap(src.Width, src.Height, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.DrawImageUnscaled(src, 0, 0);
            }
            return dst;
        }

    }

    /// <summary>
    /// 静态模型槽 1，与 EVDll 里的 AI_Gpu1 写法一致（一个进程内每个槽一个模型）。
    /// 共 48 个槽：AI_Seg1 在本文件，AI_Seg2…AI_Seg48 见 AI_Seg_Slots.cs，用法完全相同。
    /// </summary>
    public static class AI_Seg1
    {
        private static SegModel _model;

        /// <summary>本 DLL 的版本标识（构建时取自最近一次 git 提交，见 GitVersion.g.cs）。</summary>
        public static string Version { get { return SegVersionInfo.Current; } }

        public static void Init(string AIPath)
        {
            string dummy = null;
            Init(AIPath, ref dummy);
        }

        public static void Init(string AIPath, ref string ErroMessage)
        {
            if (_model != null) return;
            try
            {
                if (!File.Exists(AIPath))
                {
                    ErroMessage = "AI_Seg1:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg1:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg1:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg1:AI Init报错：" + e;
                _model = null;
            }
        }

        public static Bitmap Run(Bitmap bitmap)
        {
            string dummy = null;
            return Run(bitmap, ref dummy);
        }

        public static Bitmap Run(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg1:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg1:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        // 只推理、不生成叠加图：只需要 mask / 统计时用它，省掉每帧的叠加图开销。
        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg1:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg1:AI RunMask报错：" + e;
                return false;
            }
        }

        public static Bitmap GetMask()
        {
            return _model == null ? null : _model.GetMask();
        }

        public static SegStat GetStat()
        {
            return _model == null ? new SegStat() : _model.GetStat();
        }

        /// <summary>设置 Softmax 最大置信度阈值 [0,1]，默认 0.5；低于阈值判为背景。</summary>
        public static bool SetConfidenceThreshold(float threshold)
        {
            string dummy = null;
            return SetConfidenceThreshold(threshold, ref dummy);
        }

        public static bool SetConfidenceThreshold(float threshold, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "模型未打开";
                return false;
            }
            try
            {
                _model.SetConfidenceThreshold(threshold);
                return true;
            }
            catch (Exception e)
            {
                ErroMessage = "设置置信度阈值报错：" + e;
                return false;
            }
        }

        public static void Close()
        {
            if (_model != null)
            {
                _model.Dispose();
                _model = null;
            }
        }

        private static bool ContainsChinese(string path)
        {
            foreach (char c in path)
            {
                if (c >= 0x4E00 && c <= 0x9FA5) return true;
            }
            return false;
        }
    }

    /// <summary>
    /// 读取编译时注入的程序集版本标识（AssemblyInformationalVersion，由 csproj 的
    /// GenerateGitVersion 目标取自最近一次 git 提交），供所有 AI_SegN.Version 使用。
    /// </summary>
    internal static class SegVersionInfo
    {
        private static readonly string _current = Read();

        public static string Current { get { return _current; } }

        private static string Read()
        {
            Assembly asm = typeof(SegVersionInfo).Assembly;
            object[] attrs = asm.GetCustomAttributes(typeof(AssemblyInformationalVersionAttribute), false);
            if (attrs.Length > 0)
                return ((AssemblyInformationalVersionAttribute)attrs[0]).InformationalVersion;
            return asm.GetName().Version.ToString();
        }
    }
}
