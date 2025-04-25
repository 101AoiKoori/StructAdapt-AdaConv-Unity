using UnityEngine;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;

public static class TextureUtils
{
    private static ComputeShader textureConversionShader;
    private static int textureToTensorKernel;
    private static int tensorToTextureKernel;
    private static bool debugLogging = false;

    // 缓冲区池 - 避免反复创建释放ComputeBuffer
    private static Dictionary<int, ComputeBuffer> bufferPool = new Dictionary<int, ComputeBuffer>();

    // 临时渲染纹理池 - 避免频繁分配RenderTexture
    private static Dictionary<string, RenderTexture> rtPool = new Dictionary<string, RenderTexture>();

    static TextureUtils()
    {
        textureConversionShader = Resources.Load<ComputeShader>("Shaders/TextureConversion");
        if (textureConversionShader == null)
        {
            Debug.LogError("无法加载纹理转换计算着色器，请确保它在Resources/Shaders目录中");
            return;
        }

        textureToTensorKernel = textureConversionShader.FindKernel("CSTextureToTensor");
        tensorToTextureKernel = textureConversionShader.FindKernel("CSTensorToTexture");

        if (debugLogging)
        {
            Debug.Log("TextureUtils 初始化成功");
        }
    }

    /// <summary>
    /// 从缓冲区池获取或创建ComputeBuffer
    /// </summary>
    private static ComputeBuffer GetOrCreateBuffer(int size, int stride)
    {
        int key = size * 1000 + stride; // 简单的哈希键
        if (bufferPool.TryGetValue(key, out ComputeBuffer buffer))
        {
            if (buffer != null && buffer.count == size && buffer.stride == stride)
            {
                return buffer;
            }
            // 如果大小不匹配，释放旧的
            if (buffer != null)
            {
                buffer.Release();
            }
        }

        // 创建新的缓冲区
        buffer = new ComputeBuffer(size, stride);
        bufferPool[key] = buffer;
        return buffer;
    }

    /// <summary>
    /// 从渲染纹理池获取或创建RenderTexture
    /// </summary>
    private static RenderTexture GetOrCreateRenderTexture(int width, int height, RenderTextureFormat format)
    {
        string key = $"{width}x{height}_{format}";
        if (rtPool.TryGetValue(key, out RenderTexture rt))
        {
            if (rt != null && rt.width == width && rt.height == height && rt.format == format)
            {
                return rt;
            }
            // 如果尺寸或格式不匹配，释放旧的
            if (rt != null)
            {
                rt.Release();
                Object.Destroy(rt);
            }
        }

        // 创建新的渲染纹理
        rt = new RenderTexture(width, height, 0, format);
        rt.enableRandomWrite = true;
        rt.Create();
        rtPool[key] = rt;
        return rt;
    }

    public static Tensor<float> TextureToTensor(Texture texture, int channels, int targetWidth, int targetHeight, int batchSize)
    {
        if (texture == null || textureConversionShader == null)
        {
            Debug.LogError("纹理或计算着色器无效");
            return null;
        }

        try
        {
            // 获取临时渲染纹理
            RenderTexture tempRT = GetOrCreateRenderTexture(targetWidth, targetHeight, RenderTextureFormat.ARGBFloat);

            // 将输入纹理缩放到目标尺寸
            Graphics.Blit(texture, tempRT);

            // 获取足够大的缓冲区
            int bufferSize = batchSize * channels * targetWidth * targetHeight;
            ComputeBuffer tensorBuffer = GetOrCreateBuffer(bufferSize, sizeof(float));

            // 设置计算着色器参数
            textureConversionShader.SetTexture(textureToTensorKernel, "_InputTexture", tempRT);
            textureConversionShader.SetBuffer(textureToTensorKernel, "_OutputBuffer", tensorBuffer);
            textureConversionShader.SetInt("_Width", targetWidth);
            textureConversionShader.SetInt("_Height", targetHeight);
            textureConversionShader.SetInt("_Channels", channels);
            textureConversionShader.SetInt("_BatchSize", batchSize);

            // 优化线程组大小 - 16是计算着色器中的线程组尺寸
            int threadGroupsX = Mathf.CeilToInt(targetWidth / 16f);
            int threadGroupsY = Mathf.CeilToInt(targetHeight / 16f);
            ComputeShaderUtils.DispatchComputeShaderCustom(textureConversionShader, textureToTensorKernel, threadGroupsX, threadGroupsY);

            // 获取结果数据
            float[] tensorData = new float[bufferSize];
            tensorBuffer.GetData(tensorData);

            // 创建ONNX运行时需要的DenseTensor
            var tensor = new DenseTensor<float>(new[] { batchSize, channels, targetHeight, targetWidth });

            // 填充张量数据 - 优化的索引计算
            for (int n = 0; n < batchSize; n++)
            {
                for (int c = 0; c < channels; c++)
                {
                    int channelOffset = c * targetWidth * targetHeight;
                    for (int h = 0; h < targetHeight; h++)
                    {
                        int rowOffset = h * targetWidth;
                        for (int w = 0; w < targetWidth; w++)
                        {
                            int index = n * channels * targetWidth * targetHeight + channelOffset + rowOffset + w;
                            tensor[n, c, h, w] = tensorData[index];
                        }
                    }
                }
            }

            return tensor;
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TextureToTensor错误: {e.Message}");
            return null;
        }
    }

    public static void TensorToRenderTexture(Tensor<float> tensor, RenderTexture target, Vector4 colorBias, int targetWidth, int targetHeight)
    {
        if (tensor == null || target == null || textureConversionShader == null)
        {
            Debug.LogError("张量、目标纹理或计算着色器无效");
            return;
        }

        try
        {
            // 确保目标纹理尺寸正确
            if (target.width != targetWidth || target.height != targetHeight)
            {
                Debug.LogWarning($"目标纹理尺寸({target.width}x{target.height})与期望尺寸({targetWidth}x{targetHeight})不匹配");
                // 仍继续执行，因为内容会被缩放
            }

            // 计算张量大小
            int tensorSize = 1;
            foreach (var dim in tensor.Dimensions)
            {
                tensorSize *= dim;
            }

            // 获取计算缓冲区
            ComputeBuffer tensorBuffer = GetOrCreateBuffer(tensorSize, sizeof(float));
            tensorBuffer.SetData(tensor.ToArray());

            // 确保目标纹理启用随机写入
            if (!target.enableRandomWrite)
            {
                target.enableRandomWrite = true;
                target.Create();
            }

            // 设置计算着色器参数
            textureConversionShader.SetBuffer(tensorToTextureKernel, "_InputBuffer", tensorBuffer);
            textureConversionShader.SetTexture(tensorToTextureKernel, "_OutputTexture", target);
            textureConversionShader.SetVector("_ColorBias", colorBias);
            textureConversionShader.SetInt("_Width", targetWidth);
            textureConversionShader.SetInt("_Height", targetHeight);
            textureConversionShader.SetInt("_Channels", tensor.Dimensions[1]);

            // 优化线程组大小
            int threadGroupsX = Mathf.CeilToInt(targetWidth / 16f);
            int threadGroupsY = Mathf.CeilToInt(targetHeight / 16f);
            ComputeShaderUtils.DispatchComputeShaderCustom(textureConversionShader, tensorToTextureKernel, threadGroupsX, threadGroupsY);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"TensorToRenderTexture错误: {e.Message}");
        }
    }

    /// <summary>
    /// 清理所有缓冲区和渲染纹理
    /// </summary>
    public static void Cleanup()
    {
        // 释放所有缓冲区
        foreach (var buffer in bufferPool.Values)
        {
            if (buffer != null)
            {
                buffer.Release();
            }
        }
        bufferPool.Clear();

        // 释放所有渲染纹理
        foreach (var rt in rtPool.Values)
        {
            if (rt != null)
            {
                rt.Release();
                Object.Destroy(rt);
            }
        }
        rtPool.Clear();
    }
}