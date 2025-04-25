using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using System;
using System.Threading.Tasks;

public class InferenceExecutor
{
    private InferenceSession session;
    private bool isWarmupComplete = false;
    private bool debugLogging = false;
    private int warmupSize = 256;
    private int defaultBatchSize = 1;
    private int defaultChannels = 3;

    // 输入输出名称缓存，避免每次查询
    private string[] inputNames;
    private string[] outputNames;

    // 缓存上一次输入的形状，用于快速检测是否需要重新分配输出内存
    private int[] lastInputDimensions = null;
    private int[] lastOutputDimensions = null;

    // 计时统计
    private System.Diagnostics.Stopwatch performanceWatch = new System.Diagnostics.Stopwatch();
    private float averageInferenceTime = 0f;
    private int inferenceCount = 0;

    public InferenceExecutor(InferenceSession session)
    {
        this.session = session;

        // 提前缓存输入输出名称
        CacheIONames();

        // 初始化时进行预热，优化首次推理性能
        if (!isWarmupComplete)
        {
            RunWarmup();
        }
    }

    /// <summary>
    /// 缓存输入输出名称，避免每次推理都查询
    /// </summary>
    private void CacheIONames()
    {
        try
        {
            if (session != null)
            {
                inputNames = session.InputMetadata.Keys.ToArray();
                outputNames = session.OutputMetadata.Keys.ToArray();

                if (debugLogging)
                {
                    Debug.Log($"缓存了输入名称: {string.Join(", ", inputNames)}");
                    Debug.Log($"缓存了输出名称: {string.Join(", ", outputNames)}");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"缓存I/O名称时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 执行预热推理，避免首次推理延迟
    /// </summary>
    private void RunWarmup()
    {
        try
        {
            // 检查是否已有输入描述
            var metadata = session.InputMetadata;
            if (metadata == null || metadata.Count == 0)
            {
                Debug.LogWarning("无法获取模型输入元数据，使用默认值");
                return;
            }

            // 获取第一个输入的形状
            var firstInput = metadata.First();
            var inputDims = firstInput.Value.Dimensions;

            // 如果维度是动态的（包含负值），替换为预设值
            int[] warmupDims = new int[inputDims.Length];
            for (int i = 0; i < inputDims.Length; i++)
            {
                warmupDims[i] = inputDims[i] > 0 ? inputDims[i] :
                                (i == 0 ? defaultBatchSize :
                                 i == 1 ? defaultChannels :
                                 warmupSize);
            }

            Debug.Log($"使用形状 {string.Join("x", warmupDims)} 进行模型预热");

            // 创建预热输入
            var warmupContentTensor = new DenseTensor<float>(warmupDims);
            var warmupStyleTensor = new DenseTensor<float>(warmupDims);

            // 用随机值填充
            System.Random rand = new System.Random(42);
            FillTensorWithRandomData(warmupContentTensor, rand);
            FillTensorWithRandomData(warmupStyleTensor, rand);

            // 准备输入
            var inputs = new List<NamedOnnxValue>();
            if (inputNames.Length >= 2)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[0], warmupContentTensor));
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[1], warmupStyleTensor));
            }
            else if (inputNames.Length == 1)
            {
                // 某些模型可能只有一个输入
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[0], warmupContentTensor));
            }

            // 执行预热推理
            performanceWatch.Restart();
            using (var warmupResult = session.Run(inputs))
            {
                var output = warmupResult.First().AsTensor<float>();
                performanceWatch.Stop();

                if (output != null)
                {
                    Debug.Log($"预热成功: 输出形状 {string.Join("x", output.Dimensions.ToArray())}, 耗时: {performanceWatch.ElapsedMilliseconds}ms");
                    lastOutputDimensions = output.Dimensions.ToArray().Select(d => (int)d).ToArray();
                }
            }

            isWarmupComplete = true;
        }
        catch (Exception e)
        {
            Debug.LogWarning($"预热失败: {e.Message}");
            isWarmupComplete = true; // 即使失败也标记为完成，避免重复尝试
        }
    }

    // 辅助方法：用随机数填充张量
    private void FillTensorWithRandomData(Tensor<float> tensor, System.Random rand)
    {
        // 通过ToArray()和CopyTo方法处理数据
        float[] tensorData = new float[tensor.Length];
        for (int i = 0; i < tensorData.Length; i++)
        {
            tensorData[i] = (float)rand.NextDouble();
        }

        // 使用SetValue方法逐个填充tensor
        for (int i = 0; i < tensorData.Length; i++)
        {
            tensor.SetValue(i, tensorData[i]);
        }
    }

    /// <summary>
    /// 执行模型推理
    /// </summary>
    public Tensor<float> RunInference(Tensor<float> inputTensor, Tensor<float> styleTensor)
    {
        // 性能计时
        performanceWatch.Restart();

        try
        {
            // 检查输入形状是否变化
            bool inputShapeChanged = false;

            // 将输入张量的维度转换为int数组
            int[] currentDimensions = inputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();

            if (lastInputDimensions == null ||
                !currentDimensions.SequenceEqual(lastInputDimensions))
            {
                lastInputDimensions = currentDimensions;
                inputShapeChanged = true;

                if (debugLogging && inputShapeChanged)
                {
                    Debug.Log($"输入形状改变为: {string.Join("x", lastInputDimensions)}");
                }
            }

            // 准备输入
            var inputs = new List<NamedOnnxValue>();
            if (inputNames.Length >= 2)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor));
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[1], styleTensor));
            }
            else if (inputNames.Length == 1)
            {
                // 兼容只有一个输入的模型
                inputs.Add(NamedOnnxValue.CreateFromTensor(inputNames[0], inputTensor));
            }

            // 执行推理
            using (var results = session.Run(inputs))
            {
                // 获取第一个输出（通常是生成的图像）
                var outputTensor = results.First().AsTensor<float>();

                // 更新统计信息
                performanceWatch.Stop();
                float inferenceTime = performanceWatch.ElapsedMilliseconds / 1000f;

                // 使用指数移动平均更新平均推理时间
                inferenceCount++;
                if (inferenceCount == 1)
                {
                    averageInferenceTime = inferenceTime;
                }
                else
                {
                    float alpha = 0.1f;  // 平滑因子
                    averageInferenceTime = (1 - alpha) * averageInferenceTime + alpha * inferenceTime;
                }

                // 每10次推理报告一次性能
                if (debugLogging && inferenceCount % 10 == 0)
                {
                    Debug.Log($"推理性能: 平均 {averageInferenceTime * 1000:F2}ms, FPS: {1 / averageInferenceTime:F1}");
                }

                // 缓存输出形状
                lastOutputDimensions = outputTensor.Dimensions.ToArray().Select(d => (int)d).ToArray();

                return outputTensor;
            }
        }
        catch (Exception e)
        {
            performanceWatch.Stop();
            Debug.LogError($"推理错误: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 获取模型元数据
    /// </summary>
    public Dictionary<string, string> GetModelMetadata()
    {
        Dictionary<string, string> metadata = new Dictionary<string, string>();

        try
        {
            if (session != null)
            {
                // 获取输入和输出信息
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;

                metadata.Add("模型会话", session.GetType().ToString());
                metadata.Add("输入数量", inputs.Count.ToString());
                metadata.Add("输出数量", outputs.Count.ToString());
                metadata.Add("平均推理时间", $"{averageInferenceTime * 1000:F2}ms");
                metadata.Add("估计FPS", $"{1 / averageInferenceTime:F1}");

                // 添加详细的输入信息
                int inputIndex = 0;
                foreach (var input in inputs)
                {
                    var shape = string.Join("x", input.Value.Dimensions);
                    metadata.Add($"输入 {inputIndex} 名称", input.Key);
                    metadata.Add($"输入 {inputIndex} 形状", shape);
                    metadata.Add($"输入 {inputIndex} 类型", input.Value.ElementType.ToString());
                    inputIndex++;
                }

                // 添加详细的输出信息
                int outputIndex = 0;
                foreach (var output in outputs)
                {
                    var shape = string.Join("x", output.Value.Dimensions);
                    metadata.Add($"输出 {outputIndex} 名称", output.Key);
                    metadata.Add($"输出 {outputIndex} 形状", shape);
                    metadata.Add($"输出 {outputIndex} 类型", output.Value.ElementType.ToString());
                    outputIndex++;
                }
            }
            else
            {
                metadata.Add("错误", "会话未初始化");
            }
        }
        catch (Exception e)
        {
            metadata.Add("错误", e.Message);
        }

        return metadata;
    }
}