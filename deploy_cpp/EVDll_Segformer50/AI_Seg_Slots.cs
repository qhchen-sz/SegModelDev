// EVDll_Segformer —— 静态模型槽 AI_Seg2 … AI_Seg48（共 47 个）。
//
// 与 AI_Seg.cs 里的 AI_Seg1 逐份复制、完全同构：一个进程内每个槽一个模型，
// 各槽持有独立的 SegModel，错误消息前缀为各自的槽名。
// 注意：逻辑修改必须同步全部 48 个槽（这是与 EVDll 的 AI_Gpu1/2/3… 一致的写法）。

using System;
using System.Drawing;
using System.IO;

namespace EVDll_Segformer
{
    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg2
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
                    ErroMessage = "AI_Seg2:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg2:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg2:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg2:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg2:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg2:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg2:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg2:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg3
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
                    ErroMessage = "AI_Seg3:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg3:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg3:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg3:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg3:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg3:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg3:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg3:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg4
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
                    ErroMessage = "AI_Seg4:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg4:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg4:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg4:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg4:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg4:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg4:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg4:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg5
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
                    ErroMessage = "AI_Seg5:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg5:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg5:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg5:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg5:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg5:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg5:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg5:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg6
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
                    ErroMessage = "AI_Seg6:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg6:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg6:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg6:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg6:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg6:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg6:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg6:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg7
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
                    ErroMessage = "AI_Seg7:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg7:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg7:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg7:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg7:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg7:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg7:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg7:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg8
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
                    ErroMessage = "AI_Seg8:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg8:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg8:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg8:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg8:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg8:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg8:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg8:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg9
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
                    ErroMessage = "AI_Seg9:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg9:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg9:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg9:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg9:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg9:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg9:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg9:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg10
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
                    ErroMessage = "AI_Seg10:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg10:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg10:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg10:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg10:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg10:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg10:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg10:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg11
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
                    ErroMessage = "AI_Seg11:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg11:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg11:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg11:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg11:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg11:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg11:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg11:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg12
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
                    ErroMessage = "AI_Seg12:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg12:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg12:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg12:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg12:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg12:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg12:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg12:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg13
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
                    ErroMessage = "AI_Seg13:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg13:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg13:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg13:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg13:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg13:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg13:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg13:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg14
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
                    ErroMessage = "AI_Seg14:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg14:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg14:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg14:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg14:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg14:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg14:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg14:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg15
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
                    ErroMessage = "AI_Seg15:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg15:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg15:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg15:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg15:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg15:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg15:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg15:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg16
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
                    ErroMessage = "AI_Seg16:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg16:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg16:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg16:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg16:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg16:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg16:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg16:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg17
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
                    ErroMessage = "AI_Seg17:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg17:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg17:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg17:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg17:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg17:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg17:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg17:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg18
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
                    ErroMessage = "AI_Seg18:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg18:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg18:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg18:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg18:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg18:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg18:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg18:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg19
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
                    ErroMessage = "AI_Seg19:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg19:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg19:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg19:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg19:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg19:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg19:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg19:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg20
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
                    ErroMessage = "AI_Seg20:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg20:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg20:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg20:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg20:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg20:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg20:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg20:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg21
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
                    ErroMessage = "AI_Seg21:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg21:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg21:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg21:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg21:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg21:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg21:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg21:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg22
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
                    ErroMessage = "AI_Seg22:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg22:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg22:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg22:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg22:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg22:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg22:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg22:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg23
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
                    ErroMessage = "AI_Seg23:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg23:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg23:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg23:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg23:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg23:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg23:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg23:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg24
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
                    ErroMessage = "AI_Seg24:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg24:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg24:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg24:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg24:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg24:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg24:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg24:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg25
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
                    ErroMessage = "AI_Seg25:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg25:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg25:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg25:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg25:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg25:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg25:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg25:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg26
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
                    ErroMessage = "AI_Seg26:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg26:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg26:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg26:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg26:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg26:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg26:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg26:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg27
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
                    ErroMessage = "AI_Seg27:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg27:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg27:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg27:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg27:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg27:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg27:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg27:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg28
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
                    ErroMessage = "AI_Seg28:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg28:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg28:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg28:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg28:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg28:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg28:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg28:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg29
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
                    ErroMessage = "AI_Seg29:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg29:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg29:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg29:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg29:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg29:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg29:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg29:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg30
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
                    ErroMessage = "AI_Seg30:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg30:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg30:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg30:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg30:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg30:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg30:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg30:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg31
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
                    ErroMessage = "AI_Seg31:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg31:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg31:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg31:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg31:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg31:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg31:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg31:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg32
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
                    ErroMessage = "AI_Seg32:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg32:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg32:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg32:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg32:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg32:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg32:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg32:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg33
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
                    ErroMessage = "AI_Seg33:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg33:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg33:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg33:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg33:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg33:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg33:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg33:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg34
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
                    ErroMessage = "AI_Seg34:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg34:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg34:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg34:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg34:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg34:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg34:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg34:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg35
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
                    ErroMessage = "AI_Seg35:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg35:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg35:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg35:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg35:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg35:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg35:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg35:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg36
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
                    ErroMessage = "AI_Seg36:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg36:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg36:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg36:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg36:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg36:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg36:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg36:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg37
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
                    ErroMessage = "AI_Seg37:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg37:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg37:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg37:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg37:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg37:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg37:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg37:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg38
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
                    ErroMessage = "AI_Seg38:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg38:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg38:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg38:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg38:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg38:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg38:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg38:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg39
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
                    ErroMessage = "AI_Seg39:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg39:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg39:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg39:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg39:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg39:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg39:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg39:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg40
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
                    ErroMessage = "AI_Seg40:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg40:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg40:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg40:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg40:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg40:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg40:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg40:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg41
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
                    ErroMessage = "AI_Seg41:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg41:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg41:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg41:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg41:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg41:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg41:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg41:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg42
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
                    ErroMessage = "AI_Seg42:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg42:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg42:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg42:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg42:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg42:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg42:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg42:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg43
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
                    ErroMessage = "AI_Seg43:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg43:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg43:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg43:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg43:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg43:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg43:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg43:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg44
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
                    ErroMessage = "AI_Seg44:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg44:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg44:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg44:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg44:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg44:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg44:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg44:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg45
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
                    ErroMessage = "AI_Seg45:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg45:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg45:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg45:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg45:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg45:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg45:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg45:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg46
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
                    ErroMessage = "AI_Seg46:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg46:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg46:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg46:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg46:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg46:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg46:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg46:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg47
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
                    ErroMessage = "AI_Seg47:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg47:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg47:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg47:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg47:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg47:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg47:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg47:AI RunMask报错：" + e;
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

    /// <summary>静态模型槽，与 AI_Seg1 完全同构（一个进程内每个槽一个模型）。</summary>
    public static class AI_Seg48
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
                    ErroMessage = "AI_Seg48:模型路径不正确";
                    return;
                }
                if (ContainsChinese(AIPath))
                {
                    ErroMessage = "AI_Seg48:模型路径包含中文路径";
                    return;
                }
                if (Path.GetExtension(AIPath) != ".engine")
                {
                    ErroMessage = "AI_Seg48:GPU 模型必须是 .engine 文件";
                    return;
                }
                _model = new SegModel(AIPath);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg48:AI Init报错：" + e;
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
                ErroMessage = "AI_Seg48:模型未打开";
                return new Bitmap(10, 10);
            }
            try
            {
                return _model.Run(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg48:AI Run报错：" + e;
                return new Bitmap(10, 10);
            }
        }

        public static bool RunMask(Bitmap bitmap)
        {
            string dummy = null;
            return RunMask(bitmap, ref dummy);
        }

        public static bool RunMask(Bitmap bitmap, ref string ErroMessage)
        {
            if (_model == null)
            {
                ErroMessage = "AI_Seg48:模型未打开";
                return false;
            }
            try
            {
                return _model.RunMask(bitmap);
            }
            catch (Exception e)
            {
                ErroMessage = "AI_Seg48:AI RunMask报错：" + e;
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

}
