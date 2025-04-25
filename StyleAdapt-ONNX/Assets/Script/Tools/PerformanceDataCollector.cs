using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Improved Performance Data Collector - Collects performance data at fixed time intervals and exports to Origin-compatible Excel file
/// </summary>
public class PerformanceDataCollector : MonoBehaviour
{
    public enum UsedModel { SourceAdaConv = 0, DynamicBatchAdaConv = 1, StaticAdaConv = 2, Model = 3 }

    [Header("Collection Settings")]
    [Tooltip("Start collecting data")]
    public bool startCollection = false;

    [Tooltip("Collection duration (seconds)")]
    public float collectionDuration = 60f;

    [Tooltip("Sample interval (seconds)")]
    public float sampleInterval = 0.5f;

    [Tooltip("Save to Assets folder instead of Application data folder")]
    public bool saveToAssetsFolder = true;

    [Tooltip("Subdirectory for saving data")]
    public string savePath = "PerformanceData";

    [Header("Model Info")]
    [Tooltip("Name of the model being tested")]
    public UsedModel modelName = UsedModel.SourceAdaConv;

    [Tooltip("Resolution of the test")]
    public string resolution = "256x256";


    [Header("References")]
    [Tooltip("Rendering controller reference")]
    public RenderingController renderingController;

    [Tooltip("Status text")]
    public Text statusText;

    // Performance data structure
    [System.Serializable]
    public class PerformanceSample
    {
        public float timestamp;                // Timestamp (s)
        public float totalInferenceTime;       // Total inference time (s)
        public float textureToTensorTime;      // Texture to tensor time (s)
        public float onnxInferenceTime;        // ONNX inference time (s)
        public float tensorToTextureTime;      // Tensor to texture time (s)
        public float dataTransferTime;         // Data transfer time (s)
        public float gpuSyncTime;              // GPU sync time (s)
        public float fps;                      // Frames per second
        public float memoryUsageMB;            // Memory usage (MB)
        public float gpuUtilization;           // GPU utilization (%)
        public float ssim;                     // Structural similarity
        public float psnr;                     // Peak signal-to-noise ratio (dB)
        public int inferenceCount;             // Cumulative inference count
    }

    // Performance samples collection
    private List<PerformanceSample> samples = new List<PerformanceSample>();

    // Collection state
    private bool isCollecting = false;
    private float collectionStartTime = 0f;
    private float nextSampleTime = 0f;
    private bool isInitialized = false;

    // Coroutine reference for precise timing
    private Coroutine collectionCoroutine = null;

    // Expected sample count
    private int expectedSampleCount = 0;

    // Collected sample count
    private int collectedSampleCount = 0;

    // Change the accessibility of the DataFormat enum to match the accessibility of the exportFormat field
    public enum DataFormat
    {
        Excel = 0,
        CSV = 1,
        OriginLab = 2
    }
    [Header("Export Settings")]
    public DataFormat exportFormat = DataFormat.OriginLab;

    void Start()
    {
        // Delayed initialization to ensure all components are loaded
        StartCoroutine(DelayedInitialization());
    }

    // Delayed initialization to ensure other components are properly initialized
    private IEnumerator DelayedInitialization()
    {
        // Wait a frame to allow other components to initialize
        yield return null;

        Debug.Log("Initializing PerformanceDataCollector...");

        // Try to use assigned RenderingController
        if (renderingController == null)
        {
            Debug.Log("Searching for RenderingController...");
            renderingController = FindObjectOfType<RenderingController>();
            if (renderingController != null)
            {
                Debug.Log($"Found RenderingController on {renderingController.gameObject.name}");

                // Subscribe to event to receive performance data updates
                renderingController.OnPerformanceDataUpdated += OnPerformanceDataUpdated;
            }
            else
            {
                Debug.LogWarning("No RenderingController found in the scene");
            }
        }
        else
        {
            // Subscribe to the assigned rendering controller's events
            renderingController.OnPerformanceDataUpdated += OnPerformanceDataUpdated;
        }

        // Ensure save directory exists
        EnsureSaveDirectoryExists();

        // Update UI
        UpdateStatusText("Ready - Press C key to start data collection");

        isInitialized = true;
        Debug.Log("PerformanceDataCollector initialization complete");
    }

    // Ensure save directory exists
    private void EnsureSaveDirectoryExists()
    {
        string fullPath = GetFullSavePath();

        if (!Directory.Exists(fullPath))
        {
            try
            {
                Directory.CreateDirectory(fullPath);
                Debug.Log($"Created data save directory: {fullPath}");
            }
            catch (Exception e)
            {
                Debug.LogError($"Failed to create directory: {e.Message}");

                // Fall back to application data directory if Assets directory creation fails
                if (saveToAssetsFolder)
                {
                    saveToAssetsFolder = false;
                    Debug.LogWarning("Cannot create directory in Assets folder, switching to application data folder");
                    EnsureSaveDirectoryExists();
                }
            }
        }
    }

    // Get full save path
    private string GetFullSavePath()
    {
        if (saveToAssetsFolder)
        {
            return Path.Combine(Application.dataPath, savePath);
        }
        else
        {
            return Path.Combine(Application.persistentDataPath, savePath);
        }
    }

    void Update()
    {
        if (!isInitialized)
            return;

        // Check for C key press to start collection
        if (Input.GetKeyDown(KeyCode.C) && !isCollecting)
        {
            StartCollection();
        }

        // Start collection via Inspector
        if (startCollection && !isCollecting)
        {
            StartCollection();
            startCollection = false;
        }

        // Display collection progress (only showing progress, actual collection happens in coroutine)
        if (isCollecting)
        {
            float currentTime = Time.realtimeSinceStartup - collectionStartTime;
            UpdateStatusText($"Collecting data: {currentTime:F1}s / {collectionDuration:F1}s, Collected {collectedSampleCount}/{expectedSampleCount} samples");
        }
    }

    // Performance data update event handler
    private void OnPerformanceDataUpdated(
        float totalInferenceTime, float textureToTensorTime, float onnxInferenceTime,
        float tensorToTextureTime, float dataTransferTime, float fps, float ssim, float psnr)
    {
        // These values will be used in CollectSample when active
        // No immediate processing needed
    }

    // Start collecting data
    public void StartCollection()
    {
        if (isCollecting) return;

        // Calculate expected sample count
        expectedSampleCount = Mathf.CeilToInt(collectionDuration / sampleInterval);

        Debug.Log($"Starting performance data collection: duration: {collectionDuration}s, interval: {sampleInterval}s, expected samples: {expectedSampleCount}");
        samples.Clear();
        isCollecting = true;
        collectionStartTime = Time.realtimeSinceStartup;
        nextSampleTime = collectionStartTime;
        collectedSampleCount = 0;

        // Double check references
        if (renderingController == null)
        {
            renderingController = FindObjectOfType<RenderingController>();
            Debug.Log(renderingController != null ?
                "RenderingController found" :
                "RenderingController not found");

            // If found, subscribe to events
            if (renderingController != null)
            {
                renderingController.OnPerformanceDataUpdated += OnPerformanceDataUpdated;
            }
        }

        UpdateStatusText("Collecting data...");

        // Start the precise timing coroutine
        collectionCoroutine = StartCoroutine(CollectionCoroutine());
    }

    // Use coroutine for precise timed collection
    private IEnumerator CollectionCoroutine()
    {
        // Start from 0 to ensure first sample is exactly at 0 time
        float elapsedTime = 0f;

        // Loop until collection time is reached
        while (elapsedTime <= collectionDuration)
        {
            // Calculate precise current timestamp
            float currentTimestamp = elapsedTime;

            // Collect sample
            CollectSample(currentTimestamp);

            // Calculate next collection time point
            elapsedTime += sampleInterval;

            // Calculate how long to wait to reach the next sampling point
            float timeToWait = collectionStartTime + elapsedTime - Time.realtimeSinceStartup;

            if (timeToWait > 0)
            {
                // Use WaitForSeconds for precise waiting
                yield return new WaitForSeconds(timeToWait);
            }
            else
            {
                // If already past the next sampling time, collect immediately but log a warning
                Debug.LogWarning($"Sampling delay: {-timeToWait * 1000:F2}ms, may cause imprecise time intervals");
                yield return null; // Wait at least one frame
            }
        }

        // Collection complete, stop and save
        StopCollection();
    }

    // Stop collecting and save data
    public void StopCollection()
    {
        if (!isCollecting) return;

        isCollecting = false;

        // If coroutine is still running, stop it
        if (collectionCoroutine != null)
        {
            StopCoroutine(collectionCoroutine);
            collectionCoroutine = null;
        }

        Debug.Log($"Data collection complete, collected {samples.Count} samples, expected {expectedSampleCount} samples");

        // Save data to file
        SaveDataToFile();

        // Reset state
        UpdateStatusText("Data collection complete - Press C key to restart");
    }

    // Collect a single sample
    private void CollectSample(float timestamp)
    {
        var sample = new PerformanceSample();
        sample.timestamp = timestamp;  // Use precise passed timestamp instead of calculating current time

        // Get performance data directly from RenderingController (using public fields instead of reflection)
        if (renderingController != null)
        {
            // Get all performance data
            sample.totalInferenceTime = renderingController.lastInferenceTime;
            sample.textureToTensorTime = renderingController.textureToTensorTime;
            sample.onnxInferenceTime = renderingController.onnxInferenceTime;
            sample.tensorToTextureTime = renderingController.tensorToTextureTime;
            sample.dataTransferTime = renderingController.dataTransferTime;
            sample.fps = renderingController.fpsAverage;
            sample.inferenceCount = renderingController.inferenceCount;
            sample.ssim = renderingController.currentSSIM;
            sample.psnr = renderingController.currentPSNR;
        }
        // Get memory usage
        sample.memoryUsageMB = GC.GetTotalMemory(false) / (1024f * 1024f);

        // Get estimated GPU utilization
        sample.gpuUtilization = GetEstimatedGPUUtilization();

        // Add to samples collection
        samples.Add(sample);
        collectedSampleCount++;

        // Log sample data
        StringBuilder sb = new StringBuilder("Collected sample #");
        sb.Append(samples.Count);
        sb.Append(": Timestamp=");
        sb.Append($"{sample.timestamp:F3}s, ");
        sb.Append($"Inference={sample.totalInferenceTime * 1000:F2}ms, ");
        sb.Append($"FPS={sample.fps:F1}, ");
        sb.Append($"SSIM={sample.ssim:F4}, ");
        sb.Append($"PSNR={sample.psnr:F2}dB, ");
        sb.Append($"TextureToTensor={sample.textureToTensorTime * 1000:F2}ms, ");
        sb.Append($"ONNX={sample.onnxInferenceTime * 1000:F2}ms, ");
        sb.Append($"TensorToTexture={sample.tensorToTextureTime * 1000:F2}ms, ");
        sb.Append($"DataTransfer={sample.dataTransferTime * 1000:F2}ms, ");
        sb.Append($"GPU Utilization={sample.gpuUtilization:F3}%");

        Debug.Log(sb.ToString());
    }

    // Get estimated GPU utilization
    private float GetEstimatedGPUUtilization()
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

            return estimatedUtilization;
        }
        catch (Exception)
        {
            return 0f;
        }
    }

    // Save data to file
    private void SaveDataToFile()
    {
        if (samples.Count == 0)
        {
            Debug.LogWarning("No data to save");
            return;
        }

        switch (exportFormat)
        {
            case DataFormat.Excel:
                SaveToExcel();
                break;
            case DataFormat.CSV:
                SaveToCSV();
                break;
            case DataFormat.OriginLab:
                SaveToOriginLabFormat();
                break;
            default:
                SaveToCSV();
                break;
        }
    }

    // Save to Excel file (.xlsx)
    private void SaveToExcel()
    {
        // Since Unity doesn't include built-in XLSX support, we create a CSV and rename it
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename = $"{modelName}_{timestamp}.xlsx";
        string fullPath = Path.Combine(GetFullSavePath(), filename);
        string tempCsvPath = Path.Combine(GetFullSavePath(), $"{modelName}_{timestamp}_temp.csv");

        try
        {
            // First create a CSV file (temporary)
            using (StreamWriter writer = new StreamWriter(tempCsvPath, false, Encoding.UTF8))
            {
                // Write column headers
                writer.WriteLine("Timestamp,Total Inference Time,Texture→Tensor Time," +
                               "ONNX Inference Time,Tensor→Texture Time,Data Transfer Time," +
                               "FPS,Memory Usage,GPU Utilization,SSIM,PSNR,Inference Count");
                writer.WriteLine("s,ms,ms," + "ms,ms,ms," + "-,MB,%,-,dB,-");
                // Write data rows
                foreach (var sample in samples)
                {
                    writer.WriteLine($"{sample.timestamp:F3},{sample.totalInferenceTime*1000:F2},{sample.textureToTensorTime*1000:F2}," +
                           $"{sample.onnxInferenceTime * 1000:F2},{sample.tensorToTextureTime *  1000:F2},{sample.dataTransferTime * 1000:F2}," +
                           $"{sample.fps:F2},{sample.memoryUsageMB:F2},{sample.gpuUtilization:F3}," +
                           $"{sample.ssim:F4},{sample.psnr:F2},{sample.inferenceCount}");
                }
            }

            // Rename to .xlsx (this is just a workaround, not a real Excel file)
            File.Copy(tempCsvPath, fullPath, true);
            File.Delete(tempCsvPath);

            Debug.Log($"Data saved to: {fullPath}");

            // If saving to Assets folder, refresh AssetDatabase
#if UNITY_EDITOR
            if (saveToAssetsFolder)
            {
                UnityEditor.AssetDatabase.Refresh();
            }
#endif

            // Show save path to user
            UpdateStatusText($"Data saved to: {fullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving file: {e.Message}");
            UpdateStatusText($"Save failed: {e.Message}");
        }
    }

    // Save to CSV file
    private void SaveToCSV()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string filename = $"{modelName}_{timestamp}.csv";
        string fullPath = Path.Combine(GetFullSavePath(), filename);

        try
        {
            using (StreamWriter writer = new StreamWriter(fullPath, false, Encoding.UTF8))
            {
                // Write column headers
                writer.WriteLine("Timestamp,Total Inference Time,Texture→Tensor Time," +
                               "ONNX Inference Time,Tensor→Texture Time,Data Transfer Time," +
                               "FPS,Memory Usage,GPU Utilization,SSIM,PSNR,Inference Count");
                writer.WriteLine("s,ms,ms"+"ms,ms,ms"+"MB,%,-,dB,-");
                // Write data rows
                foreach (var sample in samples)
                { 

                    writer.WriteLine($"{sample.timestamp:F3},{sample.totalInferenceTime * 1000:F2},{sample.textureToTensorTime * 1000:F2}," +
                           $"{sample.onnxInferenceTime * 1000:F2},{sample.tensorToTextureTime * 1000:F2},{sample.dataTransferTime * 1000:F2}," +
                           $"{sample.fps:F2},{sample.memoryUsageMB:F2},{sample.gpuUtilization:F3}," +
                           $"{sample.ssim:F4},{sample.psnr:F2},{sample.inferenceCount}");
                }
            }

            Debug.Log($"Data saved to: {fullPath}");

            // If saving to Assets folder, refresh AssetDatabase
#if UNITY_EDITOR
            if (saveToAssetsFolder)
            {
                UnityEditor.AssetDatabase.Refresh();
            }
#endif

            // Show save path to user
            UpdateStatusText($"Data saved to: {fullPath}");
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving file: {e.Message}");
            UpdateStatusText($"Save failed: {e.Message}");
        }
    }

    // Save in OriginLab compatible format - separate file for each metric
    private void SaveToOriginLabFormat()
    {
        string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
        string baseFilename = $"{modelName}_{timestamp}";
        string basePath = GetFullSavePath();

        try
        {
            // Create files for different metrics
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_FPS.txt", "FPS", sample => sample.fps);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_InferenceTime.txt", "Inference Time (ms)", sample => sample.totalInferenceTime * 1000);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_TexToTensor.txt", "Texture→Tensor Time (ms)", sample => sample.textureToTensorTime * 1000);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_ONNXInference.txt", "ONNX Inference Time (ms)", sample => sample.onnxInferenceTime * 1000);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_TensorToTex.txt", "Tensor→Texture Time (ms)", sample => sample.tensorToTextureTime * 1000);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_DataTransfer.txt", "Data Transfer Time (ms)", sample => sample.dataTransferTime * 1000);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_GPUUtilization.txt", "GPU Utilization (%)", sample => sample.gpuUtilization);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_SSIM.txt", "SSIM", sample => sample.ssim);
            SaveMetricToOriginFormat(basePath, $"{baseFilename}_PSNR.txt", "PSNR (dB)", sample => sample.psnr);

            // Create a summary file
            string summaryPath = Path.Combine(basePath, $"{baseFilename}_Summary.txt");
            using (StreamWriter writer = new StreamWriter(summaryPath, false, Encoding.UTF8))
            {
                writer.WriteLine("# Origin-compatible performance data summary");
                writer.WriteLine($"# Model: {modelName}");
                writer.WriteLine($"# Resolution: {resolution}");
                writer.WriteLine($"# Sample count: {samples.Count}");
                writer.WriteLine($"# Test time: {DateTime.Now}");
                writer.WriteLine();
                writer.WriteLine("Index\tTimestamp(s)\tFPS\tTotal Inference(ms)\tTexture→Tensor(ms)\tONNX Inference(ms)\tTensor→Texture(ms)\tData Transfer(ms)\tSSIM\tPSNR(dB)");

                for (int i = 0; i < samples.Count; i++)
                {
                    var sample = samples[i];
                    writer.WriteLine($"{i + 1}\t{sample.timestamp:F3}\t{sample.fps:F2}\t{sample.totalInferenceTime * 1000:F2}\t" +
                                   $"{sample.textureToTensorTime * 1000:F2}\t{sample.onnxInferenceTime * 1000:F2}\t" +
                                   $"{sample.tensorToTextureTime * 1000:F2}\t{sample.dataTransferTime * 1000:F2}\t" +
                                   $"{sample.ssim:F4}\t{sample.psnr:F2}");
                }
            }

            Debug.Log($"Origin format data saved to folder: {basePath}");
            UpdateStatusText($"Origin format data saved to: {basePath}");

            // If saving to Assets folder, refresh AssetDatabase
#if UNITY_EDITOR
            if (saveToAssetsFolder)
            {
                UnityEditor.AssetDatabase.Refresh();
            }
#endif
        }
        catch (Exception e)
        {
            Debug.LogError($"Error saving Origin format data: {e.Message}");
            UpdateStatusText($"Save failed: {e.Message}");
        }
    }

    // Save a single metric in Origin format
    private void SaveMetricToOriginFormat<T>(string basePath, string filename, string metricName, Func<PerformanceSample, T> valueSelector)
    {
        string fullPath = Path.Combine(basePath, filename);

        using (StreamWriter writer = new StreamWriter(fullPath, false, Encoding.UTF8))
        {
            writer.WriteLine($"# {metricName} - {modelName} - {resolution}");
            writer.WriteLine("# Timestamp(s)\tValue");

            foreach (var sample in samples)
            {
                writer.WriteLine($"{sample.timestamp:F3}\t{valueSelector(sample)}");
            }
        }
    }

    // Update status text
    private void UpdateStatusText(string message)
    {
        if (statusText != null)
        {
            statusText.text = message;
        }

        Debug.Log(message);
    }

    void OnDisable()
    {
        // If collecting, stop
        if (isCollecting)
        {
            StopCollection();
        }

        // Unsubscribe from events
        if (renderingController != null)
        {
            renderingController.OnPerformanceDataUpdated -= OnPerformanceDataUpdated;
        }
    }
}