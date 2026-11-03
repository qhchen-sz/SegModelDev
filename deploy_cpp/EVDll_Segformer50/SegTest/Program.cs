// SegTest —— EVDll_Segformer 的端到端自测台，同时充当用法示例。
//
// 它走的是与视觉软件**完全相同**的调用路径：C# → P/Invoke → segformer_Dll_func.dll → TensorRT。
//
// 用法：
//   SegTest <engine> --eval <val_full.txt> <outDir>   逐图推理并导出 mask（供算 IoU）
//   SegTest <engine> <图片或目录>  <outDir> [--mask]  普通推理，导出叠加图（--mask 时再导 mask）

using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using EVDll_Segformer;

namespace SegTest
{
    internal static class Program
    {
        private static int Main(string[] args)
        {
            try
            {
                return Run(args);
            }
            catch (Exception e)
            {
                Console.Error.WriteLine("[ERROR] " + e);
                return 1;
            }
        }

        private static int Run(string[] args)
        {
            if (args.Length < 3)
            {
                Console.WriteLine("用法:");
                Console.WriteLine("  SegTest <engine> --eval <val_full.txt> <outDir>");
                Console.WriteLine("  SegTest <engine> <图片或目录> <outDir> [--mask]");
                Console.WriteLine("  SegTest <engine> --trace <image> <frames> [--each] [--writeback] [--dispose] [--gc] [--nomask]");
                return 2;
            }

            string enginePath = args[0];

            // ---- 布局自检：C# 结构体必须与 C++ 的 SegStat 逐字节一致 ----
            int nativeSize = SegModel.NativeStatSize();
            int managedSize = Marshal.SizeOf(typeof(SegStat));
            Console.WriteLine(string.Format("SegStat 布局: C++ {0} 字节, C# {1} 字节  -> {2}",
                nativeSize, managedSize, nativeSize == managedSize ? "OK" : "不一致！"));
            if (nativeSize != managedSize) return 3;

            using (SegModel model = new SegModel())
            {
                model.Init(enginePath);
                Console.WriteLine("引擎已加载: " + enginePath);

                if (args[1] == "--eval") return EvalMode(model, args[2], args[3]);
                if (args[1] == "--trace") return TraceMode(model, args);
                return ImageMode(model, args[1], args[2],
                                 Array.IndexOf(args, "--mask") >= 0);
            }
        }

        private static int EvalMode(SegModel model, string valFullTxt, string outDir)
        {
            Directory.CreateDirectory(outDir);
            string[] lines = File.ReadAllLines(valFullTxt);
            int n = 0;
            double totalMs = 0;
            Stopwatch sw = new Stopwatch();

            foreach (string line in lines)
            {
                if (line.Trim().Length == 0) continue;
                string[] parts = line.Split('\t');
                string imgPath = parts[0].Trim();

                using (Bitmap bmp = new Bitmap(imgPath))
                {
                    sw.Restart();
                    // Run 返回的是新建的全图 Bitmap；这里只要它触发推理，但必须释放，
                    // 否则每张图都会留一块 GDI+ 内存在等 GC 终结器。
                    using (model.Run(bmp)) { }
                    sw.Stop();
                    totalMs += sw.Elapsed.TotalMilliseconds;

                    using (Bitmap mask = model.GetMask())
                    {
                        string stem = Path.GetFileNameWithoutExtension(imgPath);
                        mask.Save(Path.Combine(outDir, stem + ".png"), ImageFormat.Png);
                    }
                }
                n++;
                if (n <= 3 || n % 10 == 0)
                    Console.WriteLine(string.Format("  [{0}] {1}  {2:F0}ms",
                        n, Path.GetFileName(imgPath), sw.Elapsed.TotalMilliseconds));
            }

            Console.WriteLine(string.Format("\n完成 {0} 张, 平均 {1:F1} ms/图", n, totalMs / n));
            return 0;
        }

        // 软件外复刻现场脚本：
        //   GetBitMap      ≈ ImageSource.Load（优先用 EasyVision 同款 FreeImageNET）
        //   SetBitMap      ≈ LockBits + Marshal.Copy 到 byte[]（拷贝成本下限）
        // 用法: SegTest <engine> --trace <image> <frames> [--each] [--writeback] [--dispose] [--gc] [--nomask]
        private static int TraceMode(SegModel model, string[] args)
        {
            if (args.Length < 4 || !int.TryParse(args[3], out int frames) || frames < 1)
            {
                Console.WriteLine("用法: SegTest <engine> --trace <image> <frames> " +
                                  "[--each] [--writeback] [--dispose] [--gc] [--nomask]");
                return 2;
            }

            string imagePath = args[2];
            bool each = Array.IndexOf(args, "--each") >= 0;
            bool writeback = Array.IndexOf(args, "--writeback") >= 0;
            bool dispose = Array.IndexOf(args, "--dispose") >= 0;
            bool forceGc = Array.IndexOf(args, "--gc") >= 0;
            bool wantMask = Array.IndexOf(args, "--nomask") < 0;

            string outPath = Environment.GetEnvironmentVariable("SEGTEST_TRACE_FILE");
            if (string.IsNullOrEmpty(outPath))
                outPath = Path.Combine(Path.GetTempPath(),
                                       "segtest_trace_" + Process.GetCurrentProcess().Id + ".csv");
            string dir = Path.GetDirectoryName(outPath);
            if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);

            Console.WriteLine("trace -> " + outPath);
            Console.WriteLine("取图方式: " + ImageSource.Description);

            long prev = 0;
            byte[] wb = null;
            int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);

            using (StreamWriter w = new StreamWriter(outPath, false))
            {
                w.WriteLine("frame,loadMs,runMaskMs,getMaskMs,writebackMs,totalMs,intervalMs," +
                            "gc0,gc1,gc2,heapMB,w,h,fmt,note");
                Bitmap cached = each ? null : ImageSource.Load(imagePath);
                for (int i = 0; i < frames; ++i)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    Bitmap src = each ? ImageSource.Load(imagePath) : cached;
                    int srcW = src == null ? 0 : src.Width;
                    int srcH = src == null ? 0 : src.Height;
                    string srcFmt = src == null ? "" : src.PixelFormat.ToString();
                    long t1 = Stopwatch.GetTimestamp();

                    model.RunMask(src);
                    long t2 = Stopwatch.GetTimestamp();

                    Bitmap mask = wantMask ? model.GetMask() : null;
                    long t3 = Stopwatch.GetTimestamp();

                    if (writeback && mask != null)
                    {
                        BitmapData bd = mask.LockBits(new Rectangle(0, 0, mask.Width, mask.Height),
                                                      ImageLockMode.ReadOnly, mask.PixelFormat);
                        try
                        {
                            int need = bd.Stride * bd.Height;
                            if (wb == null || wb.Length < need) wb = new byte[need];
                            Marshal.Copy(bd.Scan0, wb, 0, need);
                        }
                        finally { mask.UnlockBits(bd); }
                    }
                    long t4 = Stopwatch.GetTimestamp();

                    if (dispose)
                    {
                        if (mask != null) mask.Dispose();
                        if (each && src != null) src.Dispose();
                    }
                    if (forceGc)
                    {
                        GC.Collect();
                        GC.WaitForPendingFinalizers();
                    }
                    long t5 = Stopwatch.GetTimestamp();

                    double interval = prev == 0 ? 0 : Ms(prev, t0);
                    prev = t0;

                    int n0 = GC.CollectionCount(0), n1 = GC.CollectionCount(1), n2 = GC.CollectionCount(2);
                    w.WriteLine(string.Join(",", new string[] {
                        i.ToString(),
                        Ms(t0, t1).ToString("F3"),
                        Ms(t1, t2).ToString("F3"),
                        Ms(t2, t3).ToString("F3"),
                        Ms(t3, t4).ToString("F3"),
                        Ms(t0, t5).ToString("F3"),
                        interval.ToString("F3"),
                        (n0 - gc0).ToString(), (n1 - gc1).ToString(), (n2 - gc2).ToString(),
                        (GC.GetTotalMemory(false) / 1048576.0).ToString("F1"),
                        srcW.ToString(), srcH.ToString(), srcFmt,
                        "each=" + (each ? 1 : 0) + ";wb=" + (writeback ? 1 : 0) +
                        ";dispose=" + (dispose ? 1 : 0) + ";gc=" + (forceGc ? 1 : 0)
                    }));
                    gc0 = n0; gc1 = n1; gc2 = n2;
                }
                if (cached != null) cached.Dispose();
            }

            Console.WriteLine("完成 " + frames + " 帧");
            return 0;
        }

        private static double Ms(long a, long b)
        {
            return (b - a) * 1000.0 / Stopwatch.Frequency;
        }

        private static int ImageMode(SegModel model, string input, string outDir, bool wantMask)
        {
            Directory.CreateDirectory(outDir);
            string[] files;
            if (Directory.Exists(input))
            {
                files = Directory.GetFiles(input);
                Array.Sort(files, StringComparer.Ordinal);
            }
            else
            {
                files = new string[] { input };
            }

            int n = 0;
            foreach (string f in files)
            {
                string ext = Path.GetExtension(f).ToLowerInvariant();
                if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" && ext != ".bmp") continue;

                using (Bitmap bmp = new Bitmap(f))
                {
                    Stopwatch sw = Stopwatch.StartNew();
                    using (Bitmap overlay = model.Run(bmp))
                    {
                        sw.Stop();
                        string stem = Path.GetFileNameWithoutExtension(f);
                        overlay.Save(Path.Combine(outDir, stem + ".png"), ImageFormat.Png);
                        if (wantMask)
                        {
                            Directory.CreateDirectory(Path.Combine(outDir, "mask"));
                            using (Bitmap mask = model.GetMask())
                                mask.Save(Path.Combine(outDir, "mask", stem + ".png"), ImageFormat.Png);
                        }
                        SegStat st = model.GetStat();
                        Console.WriteLine(string.Format(
                            "  {0,-52} {1}x{2} {3:F0}ms  ng={4} px[bg={5} silie={6} fanzhe={7}] comp[bg={8} silie={9} fanzhe={10}] box={11}",
                            Path.GetFileName(f), bmp.Width, bmp.Height, sw.Elapsed.TotalMilliseconds,
                            st.ng, st.px[0], st.px[1], st.px[2],
                            st.ncomp[0], st.ncomp[1], st.ncomp[2], st.nbox));
                    }
                }
                n++;
            }
            Console.WriteLine("\n完成 " + n + " 张 -> " + outDir);
            return 0;
        }
    }

    /// <summary>
    /// 取图模拟：优先用 EasyVision 同款 FreeImageNET（FreeImageBitmap.ToBitmap），
    /// 找不到时退回 new Bitmap(path)。仅用于软件外复刻 GetBitMap 的拷贝成本。
    /// </summary>
    internal static class ImageSource
    {
        private static bool _init;
        private static ConstructorInfo _ctor;
        private static MethodInfo _toBitmap;
        private static MethodInfo _dispose;
        private static string _desc = "System.Drawing.Bitmap";

        public static string Description { get { Init(); return _desc; } }

        private static void Init()
        {
            if (_init) return;
            _init = true;

            string[] candidates = {
                Environment.GetEnvironmentVariable("FREEIMAGE_NET_DLL"),
                @"D:\00_EasyVision\01_Soft\Release - 14.0.18\FreeImageNET.dll",
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "FreeImageNET.dll")
            };

            foreach (string p in candidates)
            {
                if (string.IsNullOrEmpty(p) || !File.Exists(p)) continue;
                try
                {
                    Assembly asm = Assembly.LoadFrom(p);
                    Type t = asm.GetType("FreeImageAPI.FreeImageBitmap", true);
                    _ctor = t.GetConstructor(new Type[] { typeof(string) });
                    _toBitmap = t.GetMethod("ToBitmap", Type.EmptyTypes);
                    _dispose = t.GetMethod("Dispose", Type.EmptyTypes);
                    if (_ctor != null && _toBitmap != null)
                    {
                        _desc = "FreeImageNET: " + p;
                        return;
                    }
                }
                catch (Exception e)
                {
                    Console.WriteLine("[ImageSource] " + p + " 不可用: " + e.Message);
                }
            }

            _ctor = null;
            _toBitmap = null;
            _dispose = null;
        }

        public static Bitmap Load(string path)
        {
            Init();
            if (_ctor != null)
            {
                object fib = _ctor.Invoke(new object[] { path });
                try { return (Bitmap)_toBitmap.Invoke(fib, null); }
                finally { if (_dispose != null) _dispose.Invoke(fib, null); }
            }
            return new Bitmap(path);
        }
    }
}
