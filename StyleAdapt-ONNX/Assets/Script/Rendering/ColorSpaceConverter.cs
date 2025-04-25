using UnityEngine;

public class ColorSpaceConverter
{
    private ComputeShader converterCS;
    private int rgbToSRGBKernel;
    private int srgbToRGBKernel;

    // 状态
    private bool isInitialized = false;
    private bool debugLogging = false;

    public ColorSpaceConverter()
    {
        try
        {
            // 加载 Compute Shader
            converterCS = Resources.Load<ComputeShader>("Shaders/ColorSpaceConverter");
            if (converterCS == null)
            {
                Debug.LogError("无法加载 ColorSpaceConverter 计算着色器，请确保它位于 Resources/Shaders 目录下");
                return;
            }

            // 查找内核
            rgbToSRGBKernel = converterCS.FindKernel("CSRGBToSRGB");
            srgbToRGBKernel = converterCS.FindKernel("CSSRGBToRGB");

            if (rgbToSRGBKernel == -1)
            {
                Debug.LogError("无法找到计算着色器内核 CSRGBToSRGB");
            }
            if (srgbToRGBKernel == -1)
            {
                Debug.LogError("无法找到计算着色器内核 CSSRGBToRGB");
            }

            isInitialized = rgbToSRGBKernel != -1 && srgbToRGBKernel != -1;

            if (debugLogging && isInitialized)
            {
                Debug.Log("ColorSpaceConverter 初始化成功");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ColorSpaceConverter 初始化错误: {e.Message}");
            isInitialized = false;
        }
    }

    /// <summary>
    /// 将RGB线性空间转换为sRGB空间
    /// </summary>
    public void ConvertRGBToSRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        if (!isInitialized || converterCS == null)
        {
            Debug.LogError("ColorSpaceConverter 计算着色器未初始化");
            SafeBlit(source, destination);
            return;
        }

        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("ColorSpaceConverter: 源或目标纹理无效");
            SafeBlit(source, destination);
            return;
        }

        try
        {
            // 设置计算着色器参数
            converterCS.SetTexture(rgbToSRGBKernel, "_InputTexture", source);
            converterCS.SetTexture(rgbToSRGBKernel, "_OutputTexture", destination);
            converterCS.SetVector("_ColorBias", colorBias);

            // 确保目标纹理启用随机写入
            if (!destination.enableRandomWrite)
            {
                destination.enableRandomWrite = true;
                destination.Create();
            }

            // 调度计算着色器
            DispatchComputeShader(converterCS, rgbToSRGBKernel, destination.width, destination.height);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ConvertRGBToSRGB 错误: {e.Message}");
            SafeBlit(source, destination);
        }
    }

    /// <summary>
    /// 将sRGB空间转换为RGB线性空间
    /// </summary>
    public void ConvertSRGBToRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        if (!isInitialized || converterCS == null)
        {
            Debug.LogError("ColorSpaceConverter 计算着色器未初始化");
            SafeBlit(source, destination);
            return;
        }

        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("ColorSpaceConverter: 源或目标纹理无效");
            SafeBlit(source, destination);
            return;
        }

        try
        {
            // 设置计算着色器参数
            converterCS.SetTexture(srgbToRGBKernel, "_InputTexture", source);
            converterCS.SetTexture(srgbToRGBKernel, "_OutputTexture", destination);
            converterCS.SetVector("_ColorBias", colorBias);

            // 启用随机写入
            if (!destination.enableRandomWrite)
            {
                destination.enableRandomWrite = true;
                destination.Create();
            }

            // 调度计算着色器
            DispatchComputeShader(converterCS, srgbToRGBKernel, source.width, source.height);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ConvertSRGBToRGB 错误: {e.Message}");
            SafeBlit(source, destination);
        }
    }

    /// <summary>
    /// 安全的Blit操作，包含错误处理
    /// </summary>
    private void SafeBlit(RenderTexture source, RenderTexture destination)
    {
        if (source == null || destination == null)
            return;

        if (!source.IsCreated() || !destination.IsCreated())
            return;

        try
        {
            Graphics.Blit(source, destination);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Graphics.Blit 失败: {e.Message}");
        }
    }

    /// <summary>
    /// 辅助方法：调度计算着色器
    /// </summary>
    private void DispatchComputeShader(ComputeShader shader, int kernelIndex, int width, int height)
    {
        if (shader == null)
            return;

        try
        {
            uint x, y, z;
            shader.GetKernelThreadGroupSizes(kernelIndex, out x, out y, out z);
            int threadGroupsX = Mathf.CeilToInt(width / (float)x);
            int threadGroupsY = Mathf.CeilToInt(height / (float)y);
            shader.Dispatch(kernelIndex, threadGroupsX, threadGroupsY, 1);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"调度计算着色器错误: {e.Message}");
        }
    }
}