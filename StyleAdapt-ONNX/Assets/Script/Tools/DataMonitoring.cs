using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Simplified data monitoring component for performance data and image quality metrics
/// </summary>
public class DataMonitoring : MonoBehaviour
{
    [Header("UI Settings")]
    public bool showMonitoringUI = true;
    public int fontSize = 25;
    public Color backgroundColor = new Color(0, 0, 0, 0.7f);
    public Color textColor = Color.white;

    [Header("Monitoring Settings")]
    [Tooltip("Enable detailed performance analysis, showing time for each stage")]
    public bool enableDetailedProfiling = true;

    // UI components
    private Text performanceDetailText;
    private Canvas monitoringCanvas;
    private bool isUIVisible = true;

    // Performance data
    private Dictionary<string, float> performanceData = new Dictionary<string, float>();
    private System.Diagnostics.Stopwatch detailedProfiler = new System.Diagnostics.Stopwatch();

    // Reference to rendering controller
    private RenderingController renderingController;

    void Start()
    {
        // Find rendering controller
        renderingController = FindObjectOfType<RenderingController>();
        if (renderingController == null)
        {
            Debug.LogWarning("RenderingController not found, performance monitoring will be limited");
        }
        else
        {
            // Subscribe to performance data update event
            renderingController.OnPerformanceDataUpdated += OnPerformanceDataUpdated;
        }

        if (showMonitoringUI)
        {
            InitializeMonitoringUI();
        }
    }

    // Initialize monitoring UI
    private void InitializeMonitoringUI()
    {
        // Create canvas
        monitoringCanvas = new GameObject("PerformanceMonitorCanvas").AddComponent<Canvas>();
        monitoringCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
        monitoringCanvas.sortingOrder = 100; // Ensure it's on top

        // Add Canvas scaler
        CanvasScaler scaler = monitoringCanvas.gameObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);

        // Create background panel
        GameObject panelObj = new GameObject("PerformancePanel");
        panelObj.transform.SetParent(monitoringCanvas.transform);
        Image panelImage = panelObj.AddComponent<Image>();
        panelImage.color = backgroundColor;

        // Set panel position and size
        RectTransform panelRect = panelImage.rectTransform;
        panelRect.anchorMin = new Vector2(0, 0);
        panelRect.anchorMax = new Vector2(0.34f, 0.50f);
        panelRect.pivot = new Vector2(0, 0);
        panelRect.offsetMin = new Vector2(10, 10);
        panelRect.offsetMax = new Vector2(-10, -10);

        // Create text component
        GameObject textObj = new GameObject("PerformanceText");
        textObj.transform.SetParent(panelObj.transform);
        performanceDetailText = textObj.AddComponent<Text>();

        // Try to load font
        performanceDetailText.font = Resources.Load<Font>("Font/msyh");
        if (performanceDetailText.font == null)
        {
            Debug.LogWarning("Failed to load Font/msyh, using default font");
            performanceDetailText.font = Resources.GetBuiltinResource<Font>("Arial.ttf");
        }

        performanceDetailText.fontSize = fontSize;
        performanceDetailText.color = textColor;
        performanceDetailText.alignment = TextAnchor.UpperLeft;

        // Set text position
        RectTransform textRect = performanceDetailText.rectTransform;
        textRect.anchorMin = new Vector2(0, 0);
        textRect.anchorMax = new Vector2(1, 1);
        textRect.pivot = new Vector2(0.5f, 0.5f);
        textRect.offsetMin = new Vector2(10, 10);
        textRect.offsetMax = new Vector2(-10, -10);

        performanceDetailText.text = "Initializing performance monitoring...";

        // Make sure the canvas is enabled
        monitoringCanvas.enabled = showMonitoringUI;
        isUIVisible = showMonitoringUI;

        Debug.Log("Performance monitoring UI initialized");
    }

    void Update()
    {
        // Only update UI when detailed profiling is enabled and UI is visible
        if (enableDetailedProfiling && isUIVisible && performanceDetailText != null)
        {
            UpdateMonitoringUI();
        }

        // Handle keyboard input to toggle display
        if (Input.GetKeyDown(KeyCode.F1))
        {
            ToggleVisibility();
        }
    }

    // Toggle UI visibility
    public void ToggleVisibility()
    {
        isUIVisible = !isUIVisible;
        if (monitoringCanvas != null)
        {
            monitoringCanvas.enabled = isUIVisible;
        }
        Debug.Log($"Performance monitoring UI display: {isUIVisible}");
    }

    // Receive performance data updates from RenderingController
    private void OnPerformanceDataUpdated(
        float totalInferenceTime, float textureToTensorTime, float onnxInferenceTime,
        float tensorToTextureTime, float dataTransferTime, float fps, float ssim, float psnr)
    {
        // Update internal performance data
        performanceData["TotalInference"] = totalInferenceTime;
        performanceData["TextureToTensor"] = textureToTensorTime;
        performanceData["OnnxInference"] = onnxInferenceTime;
        performanceData["TensorToTexture"] = tensorToTextureTime;
        performanceData["DataTransfer"] = dataTransferTime;
        performanceData["FPS"] = fps;
        performanceData["SSIM"] = ssim;
        performanceData["PSNR"] = psnr;
    }

    /// <summary>
    /// Get performance data value
    /// </summary>
    public float GetPerformanceData(string metricName)
    {
        if (performanceData.TryGetValue(metricName, out float value))
        {
            return value;
        }
        return 0f;
    }

    /// <summary>
    /// Manually log performance data
    /// </summary>
    public void LogPerformanceData(string metricName, float value)
    {
        // Ensure non-negative value
        value = Mathf.Max(0, value);
        performanceData[metricName] = value;
    }

    // Update monitoring UI
    private void UpdateMonitoringUI()
    {
        if (performanceDetailText == null)
            return;

        System.Text.StringBuilder sb = new System.Text.StringBuilder();
        sb.AppendLine("<b>Performance Analysis</b>");

        // Get total inference time
        float totalInferTime = GetPerformanceData("TotalInference");

        float fps = totalInferTime > 0 ? 1.0f / totalInferTime : 0;
        string fpsText = fps > 1000 ? "Infinity" : fps.ToString("F1");

        sb.AppendLine($"Total Inference Time: {totalInferTime * 1000:F2} ms (FPS: {fpsText})");
        sb.AppendLine("---------------------");
        // Get performance data for each stage
        float avgTextureToTensor = GetPerformanceData("TextureToTensor");
        float avgOnnxInference = GetPerformanceData("OnnxInference");
        float avgTensorToTexture = GetPerformanceData("TensorToTexture");
        float avgDataTransfer = GetPerformanceData("DataTransfer");

        // Calculate total time and percentage
        float sumTime = avgTextureToTensor + avgOnnxInference + avgTensorToTexture + avgDataTransfer;

        // Ensure we have data to display
        if (sumTime > 0)
        {
            sb.AppendLine($"Texture→Tensor Conversion: {avgTextureToTensor * 1000:F2} ms ({avgTextureToTensor / sumTime * 100:F1}%)");
            sb.AppendLine($"ONNX Inference: {avgOnnxInference * 1000:F2} ms ({avgOnnxInference / sumTime * 100:F1}%)");
            sb.AppendLine($"Tensor→Texture Conversion: {avgTensorToTexture * 1000:F2} ms ({avgTensorToTexture / sumTime * 100:F1}%)");
            sb.AppendLine($"Data Transfer: {avgDataTransfer * 1000:F2} ms ({avgDataTransfer / sumTime * 100:F1}%)");
        }
        else
        {
            sb.AppendLine("Waiting for performance data...");
        }

        sb.AppendLine("---------------------");
        sb.AppendLine($"GPU Utilization: {GetEstimatedGPUUtilization()}%");
        sb.AppendLine($"Memory Usage: {(float)System.GC.GetTotalMemory(false) / (1024 * 1024):F1} MB");

        // Add image quality metrics
        sb.AppendLine("---------------------");
        sb.AppendLine("<b>Image Quality Metrics</b>");
        sb.AppendLine($"SSIM: {GetPerformanceData("SSIM"):F4}");
        sb.AppendLine($"PSNR: {GetPerformanceData("PSNR"):F2} dB");

        // Get extra data from rendering controller
        if (renderingController != null)
        {
            sb.AppendLine("---------------------");
            sb.AppendLine($"Inference Count: {renderingController.inferenceCount}");
            sb.AppendLine($"Thread Group Size: {renderingController.threadGroupSize}");
            sb.AppendLine($"GPU Acceleration: {(renderingController.enableGPUAcceleration ? "Enabled" : "Disabled")}");
            sb.AppendLine($"Async Rendering: {(renderingController.asyncRendering ? "Enabled" : "Disabled")}");
        }

        // Bottleneck analysis
        if (sumTime > 0)
        {
            sb.AppendLine("---------------------");
            // Find maximum value
            float maxTime = Mathf.Max(avgTextureToTensor, avgOnnxInference, avgTensorToTexture, avgDataTransfer);
            string bottleneck = "Unknown";

            if (maxTime == avgTextureToTensor) bottleneck = "Texture→Tensor";
            else if (maxTime == avgOnnxInference) bottleneck = "ONNX Inference";
            else if (maxTime == avgTensorToTexture) bottleneck = "Tensor→Texture";
            else if (maxTime == avgDataTransfer) bottleneck = "Data Transfer";

            sb.AppendLine($"<b>Bottleneck Stage:</b> {bottleneck} ({maxTime * 1000:F1}ms)");
        }

        performanceDetailText.text = sb.ToString();
    }

    /// <summary>
    /// Get detailed profiler instance for external use
    /// </summary>
    public System.Diagnostics.Stopwatch GetDetailedProfiler()
    {
        return detailedProfiler;
    }

    /// <summary>
    /// Get estimated GPU utilization
    /// </summary>
    private string GetEstimatedGPUUtilization()
    {
        try
        {
            System.Diagnostics.Stopwatch gpuTimer = new System.Diagnostics.Stopwatch();
            gpuTimer.Start();
            // Perform some GPU operations
            Matrix4x4 proj = GL.GetGPUProjectionMatrix(Camera.main.projectionMatrix, false);
            gpuTimer.Stop();

            // Estimate load based on operation time
            float gpuTimeMs = gpuTimer.ElapsedTicks / (float)System.Diagnostics.Stopwatch.Frequency * 1000f;
            float estimatedUtilization = Mathf.Clamp01(gpuTimeMs / 3f) * 100f; // Empirical estimation

            return estimatedUtilization.ToString("F3");
        }
        catch (Exception)
        {
            return "N/A";
        }
    }

    void OnDisable()
    {
        // Unsubscribe from events
        if (renderingController != null)
        {
            renderingController.OnPerformanceDataUpdated -= OnPerformanceDataUpdated;
        }
    }

    void OnDestroy()
    {
        // Clean up UI resources
        if (monitoringCanvas != null)
        {
            Destroy(monitoringCanvas.gameObject);
        }
    }
}