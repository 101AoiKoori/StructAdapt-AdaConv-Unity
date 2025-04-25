using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;
using UnityEngine.Rendering;

public class StyleTransferInference : MonoBehaviour
{
    private InferenceSession _session;
    private const string ModelPathInResources = "Models/SourceAdaConv"; // 不需要扩展名
    private const int InputSize = 256;
    private Tensor<float> contentInput;
    private Tensor<float> styleInput;

    void Start()
    {
        InitializeInferenceEnvironment();
        contentInput = CreateZeroTensor();
        styleInput = CreateZeroTensor();
    }

    private void InitializeInferenceEnvironment()
    {
        try
        {
            LogSystemInfo();
            LogCudaEnvironmentDetails();

            bool isGpuAvailable = DetectGpuAvailability();

            var options = new SessionOptions();

            if (isGpuAvailable)
            {
                try
                {
                    options.AppendExecutionProvider_CUDA(0);
                    Debug.Log("成功启用CUDA GPU推理。");
                }
                catch (Exception cudaEx)
                {
                    Debug.LogWarning($"CUDA初始化失败，将回退到CPU: {cudaEx.Message}");
                    options.AppendExecutionProvider_CPU();
                }
            }
            else
            {
                Debug.Log("未检测到可用GPU，将使用CPU推理。");
                options.AppendExecutionProvider_CPU();
            }

            // 获取模型文件的实际路径
            string modelPath = GetModelFilePath();
            if (!File.Exists(modelPath))
            {
                throw new Exception($"模型文件不存在: {modelPath}");
            }

            // 创建推理会话
            _session = new InferenceSession(modelPath, options);
        }
        catch (Exception ex)
        {
            Debug.LogError($"推理环境初始化失败: {ex.Message}");
            Debug.LogError($"详细异常信息: {ex.StackTrace}");
        }
    }

    private string GetModelFilePath()
    {
        // 获取 Unity 项目中 Resources 文件夹的实际路径
        string resourcesPath = Path.Combine(Application.dataPath, "Resources");
        return Path.Combine(resourcesPath, $"{ModelPathInResources}.onnx");
    }

    private bool DetectGpuAvailability()
    {
        try
        {
            // 1. 检查是否NVIDIA显卡
            if (!SystemInfo.graphicsDeviceName.Contains("NVIDIA"))
            {
                Debug.LogWarning($"非NVIDIA显卡: {SystemInfo.graphicsDeviceName}");
                return false;
            }

            // 2. 放宽图形API检测条件
            var unsupportedApis = new[]
            {
                GraphicsDeviceType.Metal,
                GraphicsDeviceType.OpenGLES2,
                GraphicsDeviceType.OpenGLES3
            };

            if (unsupportedApis.Contains(SystemInfo.graphicsDeviceType))
            {
                Debug.LogWarning($"不支持的图形API: {SystemInfo.graphicsDeviceType}");
                return false;
            }

            // 3. 检查CUDA环境
            string cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (string.IsNullOrEmpty(cudaPath))
            {
                Debug.Log("CUDA_PATH环境变量未设置");
                return false;
            }

            // 4. 验证CUDA 12.x库文件
            var requiredLibs = new Dictionary<string, string>
            {
                { "cudart64_12.dll", "CUDA 12运行时库" },
                { "cublas64_12.dll", "CUDA基础线性代数库" },
                { "cudnn_ops64_9.dll", "cuDNN推理库" }
            };

            foreach (var lib in requiredLibs)
            {
                string libPath = Path.Combine(cudaPath, "bin", lib.Key);
                if (!File.Exists(libPath))
                {
                    Debug.LogWarning($"缺失关键库: {lib.Value}\n路径: {libPath}");
                    return false;
                }
            }

            // 5. 实际加载测试
            return CheckCudaRuntimeLoadable();
        }
        catch (Exception ex)
        {
            Debug.LogError($"GPU检测异常: {ex.Message}");
            return false;
        }
    }

    // CUDA运行时加载验证
    [DllImport("kernel32", SetLastError = true)]
    private static extern IntPtr LoadLibrary(string lpFileName);

    private bool CheckCudaRuntimeLoadable()
    {
        try
        {
            // 尝试加载多个版本的 CUDA 运行时库
            string[] cudaVersions = {
                "cudart64_12.dll",
                "cudart64_11.dll",
                "cudart64_10.dll"
            };

            foreach (var cudaDll in cudaVersions)
            {
                string fullPath = Path.Combine(Environment.GetEnvironmentVariable("CUDA_PATH"), "bin", cudaDll);
                IntPtr handle = LoadLibrary(fullPath);

                if (handle != IntPtr.Zero)
                {
                    Debug.Log($"成功加载 CUDA 运行时库: {cudaDll}");
                    return true;
                }
            }

            Debug.LogError("未能加载任何 CUDA 运行时库");
            return false;
        }
        catch (Exception ex)
        {
            Debug.LogError($"检测 CUDA 运行时时发生异常: {ex.Message}");
            return false;
        }
    }

    private void LogSystemInfo()
    {
        Debug.Log($"操作系统: {SystemInfo.operatingSystem}");
        Debug.Log($"处理器: {SystemInfo.processorType}");
        Debug.Log($"显卡名称: {SystemInfo.graphicsDeviceName}");
        Debug.Log($"显卡类型: {SystemInfo.graphicsDeviceType}");
        Debug.Log($"显存大小: {SystemInfo.graphicsMemorySize} MB");
    }

    private void LogCudaEnvironmentDetails()
    {
        // 打印 CUDA 相关环境变量
        string[] cudaEnvVars = {
            "CUDA_PATH",
            "PATH",
            "LD_LIBRARY_PATH"
        };

        foreach (var envVar in cudaEnvVars)
        {
            string value = Environment.GetEnvironmentVariable(envVar);
            Debug.Log($"{envVar}: {value ?? "未设置"}");
        }

        // 检查 CUDA 库文件是否存在
        string cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        if (!string.IsNullOrEmpty(cudaPath))
        {
            string binPath = Path.Combine(cudaPath, "bin");
            string[] libsToCheck = {
                "cudart64_12.dll",
                "cublas64_12.dll",
                "cudnn_ops64_9.dll"
            };

            foreach (var lib in libsToCheck)
            {
                string fullPath = Path.Combine(binPath, lib);
                Debug.Log($"检查库文件 {lib}: {(File.Exists(fullPath) ? "存在" : "不存在")}");
            }
        }
    }

    void Update()
    {
        try
        {
            var sw = new System.Diagnostics.Stopwatch();
            sw.Start();

            var output = RunInference(contentInput, styleInput);

            sw.Stop();
            Debug.Log($"诊断推理耗时: {sw.ElapsedMilliseconds} ms");
            Debug.Log($"输出张量长度: {output.Length}");
        }
        catch (Exception ex)
        {
            Debug.LogError($"诊断推理失败: {ex.Message}");
            Debug.LogError($"详细异常信息: {ex.StackTrace}");
        }
    }

    private Tensor<float> CreateZeroTensor()
    {
        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        return tensor;
    }

    private float[] RunInference(Tensor<float> contentInput, Tensor<float> styleInput)
    {
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("content", contentInput),
            NamedOnnxValue.CreateFromTensor("style", styleInput)
        };

        using (var results = _session.Run(inputs))
        {
            return results.First().AsTensor<float>().ToArray();
        }
    }

    void OnDestroy()
    {
        _session?.Dispose();
    }
}