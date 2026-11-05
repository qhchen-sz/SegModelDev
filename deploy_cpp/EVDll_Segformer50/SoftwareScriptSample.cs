// SegFormer 语义分割 —— 视觉软件脚本模板（可直接粘到软件的脚本编辑器里）。
//
// 与 YOLO 版（using EVDll; AI_Gpu1.Run(bmp, ref erro, conf, nms)）的唯一区别：
// 分割模型的 Softmax 置信度阈值默认 0.5，低于阈值判为背景；Run 只接收 Bitmap，不做 NMS。
//
// 注意：Run 返回的是 3 通道叠加图（只能显示），斑点分析要用 GetMask 的
// 单通道掩膜（0=背景 1=撕裂 2=翻折），阈值设为 1~2 即可。
//
// 部署文件（详见 deploy_cpp/README.md §10）：
//   EVDll_Segformer.dll    -> 00Exe\Plugins\   （与 EVDll.dll 同目录，脚本的 using 才能解析）
//   segformer_Dll_func.dll -> 00Exe\           （与 gpu_Dll_func.dll 同目录，P/Invoke 在此查找）
//   *.engine               -> 无中文路径，例如 00Exe\models\
//   依赖 nvinfer.dll / nvinfer_plugin.dll / cudart64_12.dll / opencv_world480.dll，
//   软件目录（00Exe 或 Plugins）已有，无需自带。

using System;
using MyVision.Script.Method;
using EVDll_Segformer;

public class MyScript : ScriptMethods
{
    // 引擎路径不能含中文（AI_Seg1.Init 会直接拒绝）；.engine 与 TensorRT 版本、显卡绑定，
    // 换机器请先在目标机上用 deploy_cpp\tools\build_engine.bat 重建。
    private const string EnginePath =
        @"C:\Users\Univo\Desktop\zm_b1_fixed_batch1_fp16_ws1024.engine";

    // Softmax 置信度阈值，合法范围 [0,1]；等于阈值保留，低于阈值判背景。
    private const float ConfidenceThreshold = 0.5f;
    private static bool _thresholdSet;

    // 脚本执行该方法
    public bool Process()
    {//@
        try
        {
            string erro = "";

            // 首次调用真正加载引擎，之后每次进来直接返回，可以安全地放在 Process 里
            AI_Seg1.Init(EnginePath, ref erro);
            if (erro != "")
            {
                LogError(GetLogPre() + erro);
                return false;
            }

            // 阈值持续生效到 Close；若脚本里显式 Close 后重新 Init，需要同时重置 _thresholdSet。
            if (!_thresholdSet)
            {
                if (!AI_Seg1.SetConfidenceThreshold(ConfidenceThreshold, ref erro))
                {
                    LogError(GetLogPre() + erro);
                    return false;
                }
                _thresholdSet = true;
            }

            var src = GetBitMap("采集图像.图像");

            // 只推理、不生成叠加图：只需要 mask / 统计时用它，省掉每帧一次原图拷贝
            // 和一次叠加图生成（约 23MB 级）。需要叠加图显示时改回：
            //     var overlay = AI_Seg1.Run(src, ref erro);
            //     SetBitMap(overlay, "显示图像.图像");
            AI_Seg1.RunMask(src, ref erro);
            if (erro != "")
            {
                LogError(GetLogPre() + erro);
                return false;
            }

            // 斑点分析输入：单通道 8bpp 语义图（0=背景 1=silie/撕裂 2=fanzhe/翻折）。
            // 直接写回 采集图像.图像，斑点分析沿用原来的图像链接即可；
            // 若要保留原图，请给掩膜单独建一个图像变量，例如：
            // SetBitMap(AI_Seg1.GetMask(), "C#脚本.掩膜");
            SetBitMap(AI_Seg1.GetMask(), "采集图像.图像");

            // ---- 可选：缺陷统计（只统计最近一次 Run 的结果）----
            // SegStat st = AI_Seg1.GetStat();
            // bool ng = st.ng == 1;
            // long siliePx = st.px[1];       // 撕裂像素总数
            // long fanzhePx = st.px[2];      // 翻折像素总数
            // int silieComp = st.ncomp[1];   // 撕裂连通域总数（未做最小面积过滤）
            // int fanzheComp = st.ncomp[2];  // 翻折连通域总数
            // int nbox = st.nbox;            // st.box* 中有效的连通域个数，<= 64

            // 需要过滤小噪点时按像素数自行判断。box* 已按面积降序排列，
            // 超出 64 个时丢掉的是最小的那些，真实总数仍看 ncomp[]：
            // for (int i = 0; i < st.nbox; i++)
            // {
            //     if (st.boxPx[i] < 50) continue;          // 忽略小于 50 像素的连通域
            //     int cls = st.boxCls[i];                  // 1=撕裂 2=翻折
            //     int x = st.boxX[i], y = st.boxY[i];
            //     int w = st.boxW[i], h = st.boxH[i];
            // }

            // 把结果输出给流程（方法名以软件实际的脚本 API 为准）：
            // SetInt("分割结果.NG", ng ? 1 : 0);
            // SetInt("分割结果.撕裂像素数", (int)siliePx);
            // SetInt("分割结果.翻折像素数", (int)fanzhePx);
            // SetInt("分割结果.撕裂连通域数", silieComp);
            // SetInt("分割结果.翻折连通域数", fanzheComp);

            return true;
        }
        catch (Exception ex)
        {
            LogError(GetLogPre() + ex.ToString());
            return false;
        }
    }
}
