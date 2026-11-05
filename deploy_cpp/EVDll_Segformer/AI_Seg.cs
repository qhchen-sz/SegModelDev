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
// Softmax 置信度阈值默认 0.5，可在 Init 后通过 SegModel.SetConfidenceThreshold 调整；
// Run 仍不带 (conf, nms) 参数，不做 NMS。

using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

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

    /// <summary>一个模型实例。可同时 new 多个（每个各自持有引擎与 CUDA 上下文）。</summary>
    public sealed class SegModel : IDisposable
    {
        private IntPtr _handle;
        private bool _disposed;

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

        /// <summary>整图滑窗推理，返回叠加图（原图上色标出缺陷）。</summary>
        public Bitmap Run(Bitmap bitmap)
        {
            if (_disposed) throw new ObjectDisposedException("SegModel");
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            if (bitmap == null || bitmap.Width <= 0 || bitmap.Height <= 0)
                throw new ArgumentException("传入图片为空，请检查传入图片是否正确");

            Bitmap input = bitmap.PixelFormat == PixelFormat.Format24bppRgb
                ? bitmap
                : To24bppRgb(bitmap);

            try
            {
                int rc = ForwardCore(input, false);
                if (rc != 0)
                    throw new InvalidOperationException("SegForward 返回 " + rc + "：" + LastError());
            }
            finally
            {
                if (!ReferenceEquals(input, bitmap)) input.Dispose();
            }

            return FetchImage(false);
        }

        /// <summary>
        /// 只推理、不生成叠加图。只需要 mask / 统计时用它，省掉每帧一次原图拷贝、
        /// 一次叠加图生成（约 23MB）以及叠加图 Bitmap 的分配/回收。
        /// 行为与 50 系列一致：不主动清空上一帧原图。
        /// </summary>
        public bool RunMask(Bitmap bitmap)
        {
            if (_disposed) throw new ObjectDisposedException("SegModel");
            if (_handle == IntPtr.Zero) throw new InvalidOperationException("模型未打开，请先 Init");
            if (bitmap == null || bitmap.Width <= 0 || bitmap.Height <= 0)
                throw new ArgumentException("传入图片为空，请检查传入图片是否正确");

            Bitmap input = bitmap.PixelFormat == PixelFormat.Format24bppRgb
                ? bitmap
                : To24bppRgb(bitmap);

            try
            {
                int rc = ForwardCore(input, true);
                if (rc != 0)
                    throw new InvalidOperationException("SegForwardMask 返回 " + rc + "：" + LastError());
            }
            finally
            {
                if (!ReferenceEquals(input, bitmap)) input.Dispose();
            }
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
            Native.SegGetStat(_handle, ref st);
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

        // 直接锁定位图并把 Scan0 交给 native：省掉每帧 23MB 的托管 byte[] 分配、
        // 一次整幅拷贝和一个 Pinned GCHandle。SegForward 返回前会把像素拷进
        // lastInput，因此 UnlockBits 之后原数据失效没有影响。
        private int ForwardCore(Bitmap input, bool maskOnly)
        {
            BitmapData bd = input.LockBits(new Rectangle(0, 0, input.Width, input.Height),
                                           ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                if (bd.Stride > 0)
                {
                    return maskOnly
                        ? Native.SegForwardMask(bd.Scan0, input.Width, input.Height, bd.Stride, _handle)
                        : Native.SegForward(bd.Scan0, input.Width, input.Height, bd.Stride, _handle);
                }
            }
            finally
            {
                input.UnlockBits(bd);
            }

            // 负步长（自底向上的 DIB）不能直接交给 native，回退到托管 byte[] 路径。
            byte[] pixelData;
            int width, height, stride;
            GetBitmapData(input, out pixelData, out width, out height, out stride);
            GCHandle pinned = GCHandle.Alloc(pixelData, GCHandleType.Pinned);
            try
            {
                return maskOnly
                    ? Native.SegForwardMask(pinned.AddrOfPinnedObject(), width, height, stride, _handle)
                    : Native.SegForward(pinned.AddrOfPinnedObject(), width, height, stride, _handle);
            }
            finally
            {
                if (pinned.IsAllocated) pinned.Free();
            }
        }

        private Bitmap FetchImage(bool wantMask)
        {
            int w = 0, h = 0, step = 0, ch = 0;
            IntPtr ptr = wantMask
                ? Native.SegGetMask(ref w, ref h, ref step, ref ch, _handle)
                : Native.SegGetOverlay(ref w, ref h, ref step, ref ch, _handle);
            if (ptr == IntPtr.Zero) throw new InvalidOperationException("取结果图失败：" + LastError());

            try
            {
                byte[] data = new byte[step * h];
                Marshal.Copy(ptr, data, 0, data.Length);

                PixelFormat fmt = (ch == 1) ? PixelFormat.Format8bppIndexed : PixelFormat.Format24bppRgb;
                Bitmap bmp = new Bitmap(w, h, fmt);
                try
                {
                    if (ch == 1) ApplySegPalette(bmp);
                    BitmapData bd = bmp.LockBits(new Rectangle(0, 0, w, h), ImageLockMode.WriteOnly, fmt);
                    try
                    {
                        for (int y = 0; y < h; y++)
                            Marshal.Copy(data, y * step, bd.Scan0 + y * bd.Stride, step);
                    }
                    finally { bmp.UnlockBits(bd); }
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
                Native.SegFreeImage(ptr);
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

        private static void GetBitmapData(Bitmap bitmap, out byte[] pixelData, out int width,
                                          out int height, out int stride)
        {
            BitmapData bd = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height),
                                            ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                width = bitmap.Width;
                height = bitmap.Height;
                stride = bd.Stride;
                pixelData = new byte[bd.Stride * height];
                Marshal.Copy(bd.Scan0, pixelData, 0, pixelData.Length);
            }
            finally
            {
                bitmap.UnlockBits(bd);
            }
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
