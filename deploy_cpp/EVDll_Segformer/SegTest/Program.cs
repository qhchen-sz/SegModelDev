// SegTest —— EVDll_Segformer 的端到端自测台，同时充当用法示例。
//
// 它走的是与视觉软件**完全相同**的调用路径：C# → P/Invoke → segformer_Dll_func.dll → TensorRT。
//
// 用法：
//   SegTest <engine> --eval <val_full.txt> <outDir>   逐图推理并导出 mask（供算 IoU）
//   SegTest <engine> <图片或目录>  <outDir> [--mask]  普通推理，导出叠加图（--mask 时再导 mask）
//   SegTest <engine> <图片或目录>  <outDir> --maskonly  只推理，不生成叠加图，只导出 mask

using System;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
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
                Console.WriteLine("  SegTest <engine> <图片或目录> <outDir> [--mask] [--maskonly]");
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
                return ImageMode(model, args[1], args[2],
                                 Array.IndexOf(args, "--mask") >= 0,
                                 Array.IndexOf(args, "--maskonly") >= 0);
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

        private static int ImageMode(SegModel model, string input, string outDir, bool wantMask,
                                     bool maskOnly)
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
                    string stem = Path.GetFileNameWithoutExtension(f);
                    Stopwatch sw = Stopwatch.StartNew();
                    if (maskOnly)
                    {
                        model.RunMask(bmp);
                        sw.Stop();
                        Directory.CreateDirectory(Path.Combine(outDir, "mask"));
                        using (Bitmap mask = model.GetMask())
                            mask.Save(Path.Combine(outDir, "mask", stem + ".png"), ImageFormat.Png);
                    }
                    else
                    {
                        using (Bitmap overlay = model.Run(bmp))
                        {
                            sw.Stop();
                            overlay.Save(Path.Combine(outDir, stem + ".png"), ImageFormat.Png);
                            if (wantMask)
                            {
                                Directory.CreateDirectory(Path.Combine(outDir, "mask"));
                                using (Bitmap mask = model.GetMask())
                                    mask.Save(Path.Combine(outDir, "mask", stem + ".png"), ImageFormat.Png);
                            }
                        }
                    }
                    SegStat st = model.GetStat();
                    Console.WriteLine(string.Format(
                        "  {0,-52} {1}x{2} {3:F0}ms  ng={4} px[bg={5} silie={6} fanzhe={7}] comp[bg={8} silie={9} fanzhe={10}] box={11}",
                        Path.GetFileName(f), bmp.Width, bmp.Height, sw.Elapsed.TotalMilliseconds,
                        st.ng, st.px[0], st.px[1], st.px[2],
                        st.ncomp[0], st.ncomp[1], st.ncomp[2], st.nbox));
                }
                n++;
            }
            Console.WriteLine("\n完成 " + n + " 张 -> " + outDir);
            return 0;
        }
    }
}
