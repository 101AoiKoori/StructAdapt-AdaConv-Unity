using UnityEngine;

public static class ComputeShaderUtils
{
    private static bool debugLogging = false;

    /// <summary>
    /// 调度计算着色器，自动计算线程组大小
    /// </summary>
    /// <param name="shader">计算着色器</param>
    /// <param name="kernelIndex">内核索引</param>
    /// <param name="width">处理宽度</param>
    /// <param name="height">处理高度</param>
    public static void DispatchComputeShader(ComputeShader shader, int kernelIndex, int width, int height)
    {
        if (shader == null)
        {
            Debug.LogError("DispatchComputeShader: 计算着色器为空");
            return;
        }

        if (width <= 0 || height <= 0)
        {
            Debug.LogError($"DispatchComputeShader: 无效的尺寸 {width}x{height}");
            return;
        }

        try
        {
            // 获取线程组大小
            uint threadGroupSizeX, threadGroupSizeY, threadGroupSizeZ;
            shader.GetKernelThreadGroupSizes(kernelIndex, out threadGroupSizeX, out threadGroupSizeY, out threadGroupSizeZ);

            // 确保线程组大小合理
            if (threadGroupSizeX == 0 || threadGroupSizeY == 0)
            {
                Debug.LogError($"计算着色器线程组大小无效: [{threadGroupSizeX}, {threadGroupSizeY}, {threadGroupSizeZ}]");
                return;
            }

            // 计算调度大小
            int dispatchX = Mathf.CeilToInt(width / (float)threadGroupSizeX);
            int dispatchY = Mathf.CeilToInt(height / (float)threadGroupSizeY);

            // 确保调度大小至少为1
            dispatchX = Mathf.Max(1, dispatchX);
            dispatchY = Mathf.Max(1, dispatchY);

            if (debugLogging)
            {
                Debug.Log($"调度计算着色器: 尺寸={width}x{height}, 线程组=[{dispatchX},{dispatchY},1], 线程大小=[{threadGroupSizeX},{threadGroupSizeY},{threadGroupSizeZ}]");
            }

            // 执行调度
            shader.Dispatch(kernelIndex, dispatchX, dispatchY, 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"调度计算着色器时发生异常: {e.Message}");
        }
    }

    /// <summary>
    /// 调度计算着色器，允许指定线程组大小
    /// </summary>
    public static void DispatchComputeShaderCustom(ComputeShader shader, int kernelIndex, int groupsX, int groupsY, int groupsZ = 1)
    {
        if (shader == null)
        {
            Debug.LogError("DispatchComputeShaderCustom: 计算着色器为空");
            return;
        }

        try
        {
            // 确保调度大小至少为1
            groupsX = Mathf.Max(1, groupsX);
            groupsY = Mathf.Max(1, groupsY);
            groupsZ = Mathf.Max(1, groupsZ);

            // 执行调度
            shader.Dispatch(kernelIndex, groupsX, groupsY, groupsZ);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"调度计算着色器时发生异常: {e.Message}");
        }
    }

    /// <summary>
    /// 检查计算着色器是否可用
    /// </summary>
    public static bool IsComputeShaderSupported()
    {
        return SystemInfo.supportsComputeShaders;
    }

    /// <summary>
    /// 获取计算着色器最大线程组大小
    /// </summary>
    public static Vector3Int GetMaxThreadGroupSize()
    {
        int maxComputeWorkGroupSize = SystemInfo.maxComputeWorkGroupSize;
        return new Vector3Int(maxComputeWorkGroupSize, maxComputeWorkGroupSize, maxComputeWorkGroupSize);
    }
}