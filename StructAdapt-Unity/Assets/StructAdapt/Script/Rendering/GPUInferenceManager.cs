using System;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using System.Linq;

/// <summary>
/// GPU加速的推理管理器 - 优化版本
/// </summary>
public class GPUInferenceManager : IDisposable
{
    // 计算着色器和内核索引
    private ComputeShader computeShader;
    private int textureToTensorKernel;
    private int tensorToTextureKernel;
    private int rgbToSRGBKernel;
    private int srgbToRGBKernel;
    private int combinedTextureToTensorKernel; // 组合版本的内核

    // 推理执行器
    private InferenceExecutor inferenceExecutor;

    // 缓冲区池 - 使用ComputeBuffer池避免频繁分配和释放
    private Dictionary<int, ComputeBuffer> bufferPool = new Dictionary<int, ComputeBuffer>();
    private Dictionary<string, RenderTexture> rtPool = new Dictionary<string, RenderTexture>();

    // 性能监控
    private System.Diagnostics.Stopwatch performanceWatch = new System.Diagnostics.Stopwatch();
    private float lastInferenceTime = 0f;
    private int inferenceCount = 0;
    private float totalInferenceTime = 0f;
    private float averageInferenceTime = 0f;

    // 配置
    private int batchSize = 1;
    private int channels = 3;
    private int inputWidth = 256;
    private int inputHeight = 256;
    private bool debugLogging = false;
    private bool isDisposed = false;

    // 启用并行处理
    private bool enableParallelProcessing = true;

    // 固定大小的缓冲区 - 预分配以避免GC压力
    private ComputeBuffer contentBuffer;
    private ComputeBuffer styleBuffer;
    private ComputeBuffer outputBuffer;

    public GPUInferenceManager(InferenceExecutor executor, int width = 256, int height = 256, int channels = 3)
    {
        this.inferenceExecutor = executor;
        this.inputWidth = width;
        this.inputHeight = height;
        this.channels = channels;

        InitializeCompute();
        InitializeBuffers();
    }

    private void InitializeCompute()
    {
        // 加载计算着色器
        computeShader = Resources.Load<ComputeShader>("Shaders/StyleTransfer");
        if (computeShader == null)
        {
            Debug.LogError("无法加载StyleTransfer计算着色器。确保其位于Resources/Shaders目录下。");
            return;
        }

        // 查找内核
        textureToTensorKernel = computeShader.FindKernel("CSTextureToTensor");
        tensorToTextureKernel = computeShader.FindKernel("CSTensorToTexture");
        rgbToSRGBKernel = computeShader.FindKernel("CSRGBToSRGB");
        srgbToRGBKernel = computeShader.FindKernel("CSSRGBToRGB");
        combinedTextureToTensorKernel = computeShader.FindKernel("CSCombinedTextureToTensor");

        if (debugLogging)
        {
            Debug.Log("GPUInferenceManager: 计算着色器初始化成功");
        }
    }

    private void InitializeBuffers()
    {
        // 预分配固定大小的缓冲区
        int bufferSize = batchSize * channels * inputWidth * inputHeight;
        contentBuffer = new ComputeBuffer(bufferSize, sizeof(float));
        styleBuffer = new ComputeBuffer(bufferSize, sizeof(float));
        outputBuffer = new ComputeBuffer(bufferSize, sizeof(float));

        if (debugLogging)
        {
            Debug.Log($"GPUInferenceManager: 预分配缓冲区大小: {bufferSize} 浮点数");
        }
    }

    /// <summary>
    /// 执行风格迁移推理 - 优化版本
    /// </summary>
    public RenderTexture RunStyleTransfer(RenderTexture contentTexture, Texture styleTexture, RenderTexture outputTexture, Vector4 colorBias)
    {
        if (isDisposed || computeShader == null || inferenceExecutor == null)
        {
            Debug.LogError("GPUInferenceManager 已被释放或未正确初始化");
            return outputTexture;
        }

        performanceWatch.Restart();

        try
        {
            // 1. 快速准备输入张量 - 使用新的组合内核同时处理内容和风格纹理
            Tensor<float> contentTensor, styleTensor;
            PrepareInputTensors(contentTexture, styleTexture, out contentTensor, out styleTensor);

            // 2. 执行推理
            var outputTensor = inferenceExecutor.RunInference(contentTensor, styleTensor);
            if (outputTensor == null)
            {
                Debug.LogError("推理返回空张量");
                return outputTexture;
            }

            // 3. 将输出张量转换回纹理
            TensorToTexture(outputTensor, outputTexture, colorBias);

            // 4. 更新性能统计
            performanceWatch.Stop();
            lastInferenceTime = performanceWatch.ElapsedMilliseconds / 1000f;
            totalInferenceTime += lastInferenceTime;
            inferenceCount++;
            averageInferenceTime = totalInferenceTime / inferenceCount;

            // 定期更新性能日志
            if (debugLogging && inferenceCount % 10 == 0)
            {
                Debug.Log($"风格迁移统计: 平均时间 = {averageInferenceTime * 1000:F2}ms, 最后一次 = {lastInferenceTime * 1000:F2}ms, FPS = {1 / lastInferenceTime:F1}");
            }

            return outputTexture;
        }
        catch (Exception e)
        {
            Debug.LogError($"执行风格迁移时出错: {e.Message}\n{e.StackTrace}");
            return outputTexture;
        }
    }

    /// <summary>
    /// 准备输入张量 - 使用GPU加速并行处理内容和风格纹理
    /// </summary>
    private void PrepareInputTensors(RenderTexture contentTexture, Texture styleTexture, out Tensor<float> contentTensor, out Tensor<float> styleTensor)
    {
        if (enableParallelProcessing && combinedTextureToTensorKernel >= 0)
        {
            // 获取或创建风格纹理的临时渲染纹理
            RenderTexture styleRT = GetOrCreateRenderTexture(inputWidth, inputHeight, RenderTextureFormat.ARGBFloat);
            Graphics.Blit(styleTexture, styleRT);

            // 配置组合内核
            computeShader.SetTexture(combinedTextureToTensorKernel, "_InputTexture", contentTexture);
            computeShader.SetTexture(combinedTextureToTensorKernel, "_StyleTexture", styleRT);
            computeShader.SetBuffer(combinedTextureToTensorKernel, "_ContentBuffer", contentBuffer);
            computeShader.SetBuffer(combinedTextureToTensorKernel, "_StyleBuffer", styleBuffer);
            computeShader.SetInt("_Width", inputWidth);
            computeShader.SetInt("_Height", inputHeight);

            // 调度计算着色器 - 一次性处理内容和风格
            int threadGroupsX = Mathf.CeilToInt(inputWidth / 16f);
            int threadGroupsY = Mathf.CeilToInt(inputHeight / 16f);
            computeShader.Dispatch(combinedTextureToTensorKernel, threadGroupsX, threadGroupsY, 1);

            // 从缓冲区创建张量
            contentTensor = CreateTensorFromBuffer(contentBuffer);
            styleTensor = CreateTensorFromBuffer(styleBuffer);
        }
        else
        {
            // 单独处理内容和风格 - 回退方法
            contentTensor = TextureToTensor(contentTexture, contentBuffer);
            styleTensor = TextureToTensor(styleTexture, styleBuffer);
        }
    }

    /// <summary>
    /// 从缓冲区创建张量
    /// </summary>
    private Tensor<float> CreateTensorFromBuffer(ComputeBuffer buffer)
    {
        int bufferSize = batchSize * channels * inputWidth * inputHeight;
        float[] tensorData = new float[bufferSize];
        buffer.GetData(tensorData);

        // 创建并填充张量
        var tensor = new DenseTensor<float>(new[] { batchSize, channels, inputHeight, inputWidth });

        // 使用并行填充来提高性能
        Parallel.For(0, batchSize, n =>
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * inputWidth * inputHeight;
                for (int h = 0; h < inputHeight; h++)
                {
                    int rowOffset = h * inputWidth;
                    for (int w = 0; w < inputWidth; w++)
                    {
                        int index = n * channels * inputWidth * inputHeight + channelOffset + rowOffset + w;
                        tensor[n, c, h, w] = tensorData[index];
                    }
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// 纹理转张量 - 优化版本
    /// </summary>
    private Tensor<float> TextureToTensor(Texture texture, ComputeBuffer outputBuffer)
    {
        // 创建临时纹理并调整大小
        RenderTexture tempRT = GetOrCreateRenderTexture(inputWidth, inputHeight, RenderTextureFormat.ARGBFloat);
        Graphics.Blit(texture, tempRT);

        // 设置计算着色器参数
        computeShader.SetTexture(textureToTensorKernel, "_InputTexture", tempRT);
        computeShader.SetBuffer(textureToTensorKernel, "_OutputBuffer", outputBuffer);
        computeShader.SetInt("_Width", inputWidth);
        computeShader.SetInt("_Height", inputHeight);
        computeShader.SetInt("_Channels", channels);
        computeShader.SetInt("_BatchSize", batchSize);

        // 调度计算着色器
        int threadGroupsX = Mathf.CeilToInt(inputWidth / 16f);
        int threadGroupsY = Mathf.CeilToInt(inputHeight / 16f);
        computeShader.Dispatch(textureToTensorKernel, threadGroupsX, threadGroupsY, 1);

        // 从ComputeBuffer读取数据
        int bufferSize = batchSize * channels * inputWidth * inputHeight;
        float[] tensorData = new float[bufferSize];
        outputBuffer.GetData(tensorData);

        // 创建张量并填充数据
        var tensor = new DenseTensor<float>(new[] { batchSize, channels, inputHeight, inputWidth });

        // 使用并行处理提高性能
        Parallel.For(0, batchSize, n =>
        {
            for (int c = 0; c < channels; c++)
            {
                int channelOffset = c * inputWidth * inputHeight;
                for (int h = 0; h < inputHeight; h++)
                {
                    int rowOffset = h * inputWidth;
                    for (int w = 0; w < inputWidth; w++)
                    {
                        int index = n * channels * inputWidth * inputHeight + channelOffset + rowOffset + w;
                        tensor[n, c, h, w] = tensorData[index];
                    }
                }
            }
        });

        return tensor;
    }

    /// <summary>
    /// 张量转纹理 - 优化版本
    /// </summary>
    private void TensorToTexture(Tensor<float> tensor, RenderTexture target, Vector4 colorBias)
    {
        // 确保目标纹理已启用随机写入
        if (!target.enableRandomWrite)
        {
            target.enableRandomWrite = true;
            target.Create();
        }

        // 将张量数据复制到计算缓冲区
        outputBuffer.SetData(tensor.ToArray());

        // 设置计算着色器参数
        computeShader.SetBuffer(tensorToTextureKernel, "_InputBuffer", outputBuffer);
        computeShader.SetTexture(tensorToTextureKernel, "_OutputTexture", target);
        computeShader.SetVector("_ColorBias", colorBias);
        computeShader.SetInt("_Width", inputWidth);
        computeShader.SetInt("_Height", inputHeight);
        computeShader.SetInt("_Channels", channels);

        // 调度计算着色器
        int threadGroupsX = Mathf.CeilToInt(inputWidth / 16f);
        int threadGroupsY = Mathf.CeilToInt(inputHeight / 16f);
        computeShader.Dispatch(tensorToTextureKernel, threadGroupsX, threadGroupsY, 1);
    }

    /// <summary>
    /// 颜色空间转换: RGB 到 sRGB
    /// </summary>
    public void ConvertRGBToSRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        // 边界检查
        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("无效的纹理");
            return;
        }

        // 确保目标纹理启用随机写入
        if (!destination.enableRandomWrite)
        {
            destination.enableRandomWrite = true;
            destination.Create();
        }

        // 设置计算着色器参数
        computeShader.SetTexture(rgbToSRGBKernel, "_InputTexture", source);
        computeShader.SetTexture(rgbToSRGBKernel, "_OutputTexture", destination);
        computeShader.SetVector("_ColorBias", colorBias);

        // 调度计算着色器
        int threadGroupsX = Mathf.CeilToInt(source.width / 16f);
        int threadGroupsY = Mathf.CeilToInt(source.height / 16f);
        computeShader.Dispatch(rgbToSRGBKernel, threadGroupsX, threadGroupsY, 1);
    }

    /// <summary>
    /// 颜色空间转换: sRGB 到 RGB
    /// </summary>
    public void ConvertSRGBToRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        // 边界检查
        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("无效的纹理");
            return;
        }

        // 确保目标纹理启用随机写入
        if (!destination.enableRandomWrite)
        {
            destination.enableRandomWrite = true;
            destination.Create();
        }

        // 设置计算着色器参数
        computeShader.SetTexture(srgbToRGBKernel, "_InputTexture", source);
        computeShader.SetTexture(srgbToRGBKernel, "_OutputTexture", destination);
        computeShader.SetVector("_ColorBias", colorBias);

        // 调度计算着色器
        int threadGroupsX = Mathf.CeilToInt(source.width / 16f);
        int threadGroupsY = Mathf.CeilToInt(source.height / 16f);
        computeShader.Dispatch(srgbToRGBKernel, threadGroupsX, threadGroupsY, 1);
    }

    /// <summary>
    /// 获取或创建渲染纹理 - 避免频繁分配
    /// </summary>
    private RenderTexture GetOrCreateRenderTexture(int width, int height, RenderTextureFormat format)
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
                UnityEngine.Object.Destroy(rt);
            }
        }

        // 创建新的渲染纹理
        rt = new RenderTexture(width, height, 0, format);
        rt.enableRandomWrite = true;
        rt.Create();
        rtPool[key] = rt;
        return rt;
    }

    /// <summary>
    /// 获取性能统计信息
    /// </summary>
    public Dictionary<string, string> GetPerformanceStats()
    {
        var stats = new Dictionary<string, string>();
        stats.Add("推理次数", inferenceCount.ToString());
        stats.Add("平均推理时间", $"{averageInferenceTime * 1000:F2} ms");
        stats.Add("最后一次推理时间", $"{lastInferenceTime * 1000:F2} ms");
        stats.Add("估计FPS", $"{1 / averageInferenceTime:F1}");
        stats.Add("并行处理", enableParallelProcessing ? "启用" : "禁用");
        return stats;
    }

    /// <summary>
    /// 清理资源
    /// </summary>
    public void Dispose()
    {
        if (isDisposed)
            return;

        // 释放ComputeBuffer
        if (contentBuffer != null)
        {
            contentBuffer.Release();
            contentBuffer = null;
        }

        if (styleBuffer != null)
        {
            styleBuffer.Release();
            styleBuffer = null;
        }

        if (outputBuffer != null)
        {
            outputBuffer.Release();
            outputBuffer = null;
        }

        // 清理缓冲池
        foreach (var buffer in bufferPool.Values)
        {
            if (buffer != null)
            {
                buffer.Release();
            }
        }
        bufferPool.Clear();

        // 清理渲染纹理池
        foreach (var texture in rtPool.Values)
        {
            if (texture != null)
            {
                texture.Release();
                UnityEngine.Object.Destroy(texture);
            }
        }
        rtPool.Clear();

        isDisposed = true;

        if (debugLogging)
        {
            Debug.Log("GPUInferenceManager: 已释放所有资源");
        }
    }

    // 在对象被垃圾回收时确保释放资源
    ~GPUInferenceManager()
    {
        if (!isDisposed)
        {
            Debug.LogWarning("GPUInferenceManager: 资源未通过Dispose方法正确释放，在终结器中强制清理");
            Dispose();
        }
    }
}