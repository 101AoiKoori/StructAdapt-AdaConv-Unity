using Microsoft.ML.OnnxRuntime;
using System.IO;
using UnityEngine;
using System;
using System.Collections.Generic;

public class OnnxModelLoader
{
    private bool debugLogging = false;
    private Dictionary<string, InferenceSession> sessionCache = new Dictionary<string, InferenceSession>();

    /// <summary>
    /// 加载ONNX模型，支持模型缓存以避免重复加载
    /// </summary>
    public InferenceSession LoadModel(string modelPath, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL)
    {
        if (string.IsNullOrEmpty(modelPath))
        {
            Debug.LogError("模型路径为空");
            return null;
        }

        // 检查缓存
        if (sessionCache.TryGetValue(modelPath, out InferenceSession cachedSession))
        {
            if (debugLogging)
            {
                Debug.Log($"使用缓存的模型会话: {modelPath}");
            }
            return cachedSession;
        }

        if (!File.Exists(modelPath))
        {
            Debug.LogError($"模型文件未找到: {modelPath}");
            return null;
        }

        // 创建会话配置选项
        var options = CreateOptimizedSessionOptions(optimizationLevel);

        try
        {
            if (debugLogging)
            {
                Debug.Log($"正在加载ONNX模型: {modelPath}");
                Debug.Log($"优化级别: {optimizationLevel}, 线程数: Inter={options.InterOpNumThreads}, Intra={options.IntraOpNumThreads}");
            }

            // 测量加载时间
            var startTime = DateTime.Now;

            // 加载模型
            var session = new InferenceSession(modelPath, options);

            var loadTime = (DateTime.Now - startTime).TotalMilliseconds;

            // 记录模型信息
            Debug.Log($"模型加载成功: {Path.GetFileName(modelPath)}, 耗时: {loadTime:F0}ms, " +
                      $"输入数量={session.InputMetadata.Count}, 输出数量={session.OutputMetadata.Count}");

            // 缓存会话以供将来使用
            sessionCache[modelPath] = session;

            return session;
        }
        catch (Exception e)
        {
            Debug.LogError($"❌ 加载模型失败: {e.Message}");
            return null;
        }
    }

    /// <summary>
    /// 创建优化的会话选项
    /// </summary>
    private SessionOptions CreateOptimizedSessionOptions(GraphOptimizationLevel optimizationLevel)
    {
        var options = new SessionOptions();

        try
        {
            // 尝试启用GPU加速
            bool cudaAvailable = false;
            try
            {
                options.AppendExecutionProvider_CUDA(0);
                cudaAvailable = true;
                Debug.Log("✅ CUDA加速已启用");
            }
            catch
            {
                // 回退到DirectML (仅限Windows)
                try
                {
#if UNITY_STANDALONE_WIN || UNITY_EDITOR_WIN
                    options.AppendExecutionProvider_DML(0);
                    Debug.Log("✅ DirectML加速已启用");
#else
                    Debug.Log("❌ DirectML仅在Windows上可用");
#endif
                }
                catch
                {
                    Debug.Log("⚠️ 无可用GPU加速方案，使用CPU执行");
                }
            }

            // 设置图优化级别
            options.GraphOptimizationLevel = optimizationLevel;

            // 使用高级优化选项
            options.EnableMemoryPattern = true;
            options.EnableCpuMemArena = true;

            // 如果使用CPU，设置更激进的优化
            if (!cudaAvailable)
            {
                options.AddSessionConfigEntry("session.intra_op.allow_spinning", "1");
                options.AddSessionConfigEntry("session.inter_op.allow_spinning", "1");

                // 启用模型优化
                options.AddSessionConfigEntry("session.graph_optimization_level", "ORT_ENABLE_ALL");
            }

            // 设置线程池大小
            int numThreads = Mathf.Max(1, SystemInfo.processorCount);
            options.InterOpNumThreads = Mathf.Max(1, numThreads / 2);
            options.IntraOpNumThreads = numThreads;

            if (debugLogging)
            {
                Debug.Log($"配置优化设置: 处理器数量={numThreads}, " +
                          $"Inter线程={options.InterOpNumThreads}, " +
                          $"Intra线程={options.IntraOpNumThreads}, " +
                          $"优化级别={optimizationLevel}");
            }
        }
        catch (Exception e)
        {
            Debug.LogWarning($"配置会话选项时出错: {e.Message}，使用默认设置");
        }

        return options;
    }

    /// <summary>
    /// 清理所有已加载的模型会话
    /// </summary>
    public void Cleanup()
    {
        foreach (var session in sessionCache.Values)
        {
            try
            {
                session.Dispose();
            }
            catch
            {
                // 忽略清理错误
            }
        }
        sessionCache.Clear();

        if (debugLogging)
        {
            Debug.Log("已清理所有模型会话");
        }
    }
}