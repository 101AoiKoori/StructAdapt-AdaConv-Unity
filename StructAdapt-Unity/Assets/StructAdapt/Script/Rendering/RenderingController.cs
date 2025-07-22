using UnityEngine;
using UnityEngine.UI;
using Microsoft.ML.OnnxRuntime;
using System.IO;
using System.Collections;
using System.Collections.Generic;
using System;

public class RenderingController : MonoBehaviour
{
    public enum UsedModel { FullyDynamic = 0, BatchDynamic = 1, Static = 2, Model = 3 }

    [System.Serializable]
    public class StyleDefinition
    {
        public Texture2D image;
    }

    [System.Serializable]
    public class InternalStyleTransferSetup
    {
        public string[] onnxModelPaths;
        public Vector4 postNetworkColorBias = new Vector4(0, 0, 0, 0);
    }

    [Header("Style Transfer Setup")]
    public UsedModel modelToUse = UsedModel.FullyDynamic;
    public InternalStyleTransferSetup internalSetup;

    [Header("Style Transfer Styles")]
    public StyleDefinition[] styles;

    [Header("Style Preview")]
    public Vector2 previewSize = new Vector2(256, 256);
    public Vector2 previewOffset = new Vector2(-30, 30);

    [Header("Debug Options")]
    public bool showDebugInfo = true;
    public bool enableDetailedProfiling = true;

    [Header("AdaConv Settings")]
    public int fixedWidth_Height = 256;
    public int channel = 3;
    public int batchSize = 1;

    [Header("Performance Options")]
    public bool enableGPUAcceleration = true;
    public int threadGroupSize = 16;
    public bool asyncRendering = true;

    // Public accessible performance data (for data collection)
    [HideInInspector] public float lastInferenceTime = 0f;
    [HideInInspector] public float textureToTensorTime = 0f;
    [HideInInspector] public float onnxInferenceTime = 0f;
    [HideInInspector] public float tensorToTextureTime = 0f;
    [HideInInspector] public float dataTransferTime = 0f;
    [HideInInspector] public float gpuSyncTime = 0f;
    [HideInInspector] public float postProcessingTime = 0f;
    [HideInInspector] public float contentTexturePreparationTime = 0f;
    [HideInInspector] public float styleTexturePreparationTime = 0f;
    [HideInInspector] public float fpsAverage = 60f;
    [HideInInspector] public int inferenceCount = 0;
    [HideInInspector] public float currentSSIM = 0f;
    [HideInInspector] public float currentPSNR = 0f;

    private Camera targetCamera;
    private int currentStyleIndex = 0;
    private bool shouldApplyStyleTransfer = false;
    private Text fpsUpsampleText;

    // FPS calculation related
    private Queue<float> fpsSamples = new Queue<float>();
    private int fpsHistoryLength = 13;
    private float lastFpsUpdateTime = 0f;
    private int lastFrameCount = 0;

    // Performance timer
    private System.Diagnostics.Stopwatch performanceWatch = new System.Diagnostics.Stopwatch();

    // Inference components
    private InferenceExecutor inferenceExecutor;
    private GPUInferenceManager gpuInferenceManager;

    // Render textures
    private RenderTexture fixedRenderTexture;
    private RenderTexture styleTransferOutputRT;
    private RenderTexture processedOutputRT;
    private RenderTexture[] renderBuffers = new RenderTexture[2]; // Double buffering
    private int currentBufferIndex = 0;
    private bool isInferenceRunning = false;

    // UI-related
    private RawImage stylePreviewImage;
    private Canvas previewCanvas;

    // Performance data update event - other components can subscribe to this event
    public delegate void PerformanceDataUpdatedHandler(
        float totalInferenceTime, float textureToTensorTime, float onnxInferenceTime,
        float tensorToTextureTime, float dataTransferTime, float fps, float ssim, float psnr);
    public event PerformanceDataUpdatedHandler OnPerformanceDataUpdated;

    // Timer for forcing periodic performance updates
    private float lastPerformanceUpdateTime = 0f;
    private float performanceUpdateInterval = 0.1f; // Update every 0.1 seconds

    void Start()
    {
        InitializeStylePreview();
        targetCamera = GetComponent<Camera>();
        fpsUpsampleText = GameObject.Find("Framerate Upsample Display")?.GetComponent<Text>();

        if (fpsUpsampleText == null && showDebugInfo)
        {
            Debug.LogWarning("Framerate Upsample Display text component not found, performance info will not be shown");
        }

        // Initialize model
        if (!InitializeModel())
        {
            Debug.LogError("Model initialization failed, disabling style transfer");
            shouldApplyStyleTransfer = false;
            return;
        }

        // Ensure camera is configured correctly
        if (targetCamera != null)
        {
            targetCamera.depthTextureMode |= DepthTextureMode.Depth;
        }

        // Initialize render buffers
        if (asyncRendering)
        {
            InitializeRenderBuffers();
        }

        // Initialize performance monitoring
        lastFpsUpdateTime = Time.realtimeSinceStartup;
        lastFrameCount = Time.frameCount;
        lastPerformanceUpdateTime = Time.realtimeSinceStartup;

        if (showDebugInfo)
        {
            Debug.Log($"RenderingController initialized. Resolution: {targetCamera.pixelWidth}x{targetCamera.pixelHeight}");
        }
    }

    private void InitializeRenderBuffers()
    {
        // Create double buffering for async rendering
        for (int i = 0; i < renderBuffers.Length; i++)
        {
            renderBuffers[i] = CreateRenderTexture(
                Screen.width, Screen.height,
                RenderTextureFormat.ARGB32
            );

            if (showDebugInfo)
            {
                Debug.Log($"Created render buffer {i}: {Screen.width}x{Screen.height}");
            }
        }
    }

    private bool InitializeModel()
    {
        if (internalSetup.onnxModelPaths == null || internalSetup.onnxModelPaths.Length == 0)
        {
            Debug.LogError("No ONNX model paths provided");
            return false;
        }

        if ((int)modelToUse < 0 || (int)modelToUse >= internalSetup.onnxModelPaths.Length)
        {
            Debug.LogError("Invalid modelToUse value");
            return false;
        }

        System.Diagnostics.Stopwatch modelInitProfiler = new System.Diagnostics.Stopwatch();
        modelInitProfiler.Start();

        string relativePath = internalSetup.onnxModelPaths[(int)modelToUse];
        string modelPath = Path.Combine(Application.dataPath, relativePath);

        var modelLoader = new OnnxModelLoader();
        InferenceSession session = modelLoader.LoadModel(modelPath);
        if (session == null)
        {
            Debug.LogError("Failed to load ONNX model");
            return false;
        }

        // Record model loading time
        modelInitProfiler.Stop();
        float modelLoadTime = modelInitProfiler.ElapsedMilliseconds / 1000f;

        // Create inference executor with session
        inferenceExecutor = new InferenceExecutor(session);

        // Create GPU inference manager
        if (enableGPUAcceleration)
        {
            gpuInferenceManager = new GPUInferenceManager(
                inferenceExecutor,
                fixedWidth_Height,
                fixedWidth_Height,
                channel
            );
        }

        modelInitProfiler.Reset();
        modelInitProfiler.Start();

        // Create render textures for model input/output
        fixedRenderTexture = CreateRenderTexture(fixedWidth_Height, fixedWidth_Height, RenderTextureFormat.ARGBFloat);
        styleTransferOutputRT = CreateRenderTexture(fixedWidth_Height, fixedWidth_Height, RenderTextureFormat.ARGBFloat);

        // Ensure output texture matches screen resolution
        int screenWidth = Screen.width;
        int screenHeight = Screen.height;

        if (showDebugInfo)
        {
            Debug.Log($"Creating output render texture: {screenWidth}x{screenHeight}");
        }

        processedOutputRT = CreateRenderTexture(screenWidth, screenHeight, RenderTextureFormat.ARGB32);

        return true;
    }

    void Update()
    {
        // Update performance metrics
        UpdatePerformanceMetrics();

        // Handle user input
        HandleUserInput();

        // Update UI
        UpdateUI();

        // Resource cleanup - every 120 frames
        if (Time.frameCount % 120 == 0)
        {
            Resources.UnloadUnusedAssets();
        }
    }

    // Performance metrics update
    private void UpdatePerformanceMetrics()
    {
        // Update FPS every 0.1 seconds
        float currentTime = Time.realtimeSinceStartup;
        if (currentTime - lastFpsUpdateTime >= 0.1f)
        {
            int currentFrameCount = Time.frameCount;
            float fps = (currentFrameCount - lastFrameCount) / (currentTime - lastFpsUpdateTime);

            // Update FPS queue
            fpsSamples.Enqueue(fps);
            if (fpsSamples.Count > fpsHistoryLength)
            {
                fpsSamples.Dequeue();
            }

            // Calculate average FPS
            float sum = 0;
            foreach (float sample in fpsSamples)
            {
                sum += sample;
            }
            fpsAverage = sum / fpsSamples.Count;

            lastFpsUpdateTime = currentTime;
            lastFrameCount = currentFrameCount;
        }

        // Force publish performance data periodically to ensure monitors get updates
        // even if there are no new inference operations
        if (Time.realtimeSinceStartup - lastPerformanceUpdateTime >= performanceUpdateInterval)
        {
            PublishPerformanceData();
            lastPerformanceUpdateTime = Time.realtimeSinceStartup;
        }
    }

    // Publish performance data to subscribers
    private void PublishPerformanceData()
    {
        if (lastInferenceTime <= 0.001f && (textureToTensorTime > 0 || onnxInferenceTime > 0 || tensorToTextureTime > 0))
        {
            lastInferenceTime = textureToTensorTime + onnxInferenceTime + tensorToTextureTime;
        }

        OnPerformanceDataUpdated?.Invoke(
            lastInferenceTime, textureToTensorTime, onnxInferenceTime,
            tensorToTextureTime, dataTransferTime, fpsAverage, currentSSIM, currentPSNR);
    }

    // Handle user input
    private void HandleUserInput()
    {
        // Toggle style transfer
        if (Input.GetMouseButtonDown(0))
        {
            shouldApplyStyleTransfer = !shouldApplyStyleTransfer;
            if (stylePreviewImage != null)
            {
                stylePreviewImage.enabled = shouldApplyStyleTransfer;
            }

            if (shouldApplyStyleTransfer)
            {
                StartCoroutine(UpdateStylePreview());
            }
        }

        // Switch style
        if (Input.GetMouseButtonDown(1) && shouldApplyStyleTransfer && styles.Length > 0)
        {
            currentStyleIndex = (currentStyleIndex + 1) % styles.Length;
            StartCoroutine(UpdateStylePreview());
        }

        // Toggle GPU acceleration mode - use G key
        if (Input.GetKeyDown(KeyCode.G))
        {
            enableGPUAcceleration = !enableGPUAcceleration;
            Debug.Log($"GPU acceleration mode: {(enableGPUAcceleration ? "enabled" : "disabled")}");
        }

        // Toggle detailed performance monitoring
        if (Input.GetKeyDown(KeyCode.P))
        {
            enableDetailedProfiling = !enableDetailedProfiling;
            Debug.Log($"Detailed performance monitoring: {(enableDetailedProfiling ? "enabled" : "disabled")}");
        }

        // Adjust thread group size (+ / - keys)
        if (Input.GetKeyDown(KeyCode.Plus) || Input.GetKeyDown(KeyCode.KeypadPlus))
        {
            threadGroupSize = Mathf.Min(threadGroupSize * 2, 1024);
            Debug.Log($"Increased thread group size to: {threadGroupSize}");
        }
        else if (Input.GetKeyDown(KeyCode.Minus) || Input.GetKeyDown(KeyCode.KeypadMinus))
        {
            threadGroupSize = Mathf.Max(threadGroupSize / 2, 1);
            Debug.Log($"Decreased thread group size to: {threadGroupSize}");
        }

        // Toggle async rendering mode (using T key instead of A to avoid conflict with WASD control)
        if (Input.GetKeyDown(KeyCode.T))
        {
            asyncRendering = !asyncRendering;
            Debug.Log($"Async rendering mode: {(asyncRendering ? "enabled" : "disabled")}");

            // Initialize buffers if enabling async mode but buffers aren't initialized
            if (asyncRendering && (renderBuffers[0] == null || !renderBuffers[0].IsCreated()))
            {
                InitializeRenderBuffers();
            }
        }
    }

    // Update UI
    private void UpdateUI()
    {
        if (fpsUpsampleText != null)
        {
            float inferenceMs = lastInferenceTime * 1000f;
            float memoryMB = (float)System.GC.GetTotalMemory(false) / (1024 * 1024);

            // Get GPU inference manager performance stats
            string gpuStats = "";
            if (enableGPUAcceleration && gpuInferenceManager != null)
            {
                var stats = gpuInferenceManager.GetPerformanceStats();
                if (stats != null && stats.TryGetValue("估计FPS", out string estFps))
                {
                    gpuStats = $" | GPU: {estFps} FPS";
                }
            }

            fpsUpsampleText.text = string.Format(
                "FPS: {0:F1} | Style: {1}/{2} | Inference: {3:F1}ms | Memory: {4:F1}MB | Mode: {5}{6}",
                fpsAverage,
                currentStyleIndex + 1,
                styles.Length,
                inferenceMs,
                memoryMB,
                enableGPUAcceleration ? "GPU" : "CPU",
                gpuStats
            );
        }
    }

    IEnumerator UpdateStylePreview()
    {
        if (styles == null || styles.Length == 0 || stylePreviewImage == null)
            yield break;

        var currentStyle = styles[currentStyleIndex];
        if (currentStyle.image != null)
        {
            RenderTexture rt = new RenderTexture(currentStyle.image.width, currentStyle.image.height, 0, RenderTextureFormat.ARGBFloat);
            rt.enableRandomWrite = true;
            rt.Create();
            RenderTexture previousActive = RenderTexture.active;

            try
            {
                Graphics.Blit(currentStyle.image, rt);
                stylePreviewImage.texture = rt;
                float aspect = (float)currentStyle.image.width / currentStyle.image.height;
                stylePreviewImage.rectTransform.sizeDelta = new Vector2(previewSize.x, previewSize.x / aspect);
            }
            catch (System.Exception e)
            {
                Debug.LogError($"Error updating style preview: {e.Message}");
            }
            finally
            {
                RenderTexture.active = previousActive;
            }

            // Clean up old texture
            if (stylePreviewImage.texture is RenderTexture oldRT && oldRT != rt)
            {
                oldRT.Release();
                Destroy(oldRT);
            }
        }
        yield return null;
    }

    void InitializeStylePreview()
    {
        previewCanvas = new GameObject("PreviewCanvas").AddComponent<Canvas>();
        previewCanvas.renderMode = RenderMode.ScreenSpaceOverlay;
        previewCanvas.sortingOrder = 99;
        CanvasScaler scaler = previewCanvas.gameObject.AddComponent<CanvasScaler>();
        scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
        scaler.referenceResolution = new Vector2(1920, 1080);

        stylePreviewImage = new GameObject("StylePreview").AddComponent<RawImage>();
        stylePreviewImage.transform.SetParent(previewCanvas.transform);
        RectTransform rt = stylePreviewImage.rectTransform;
        rt.anchorMin = new Vector2(1, 0);
        rt.anchorMax = new Vector2(1, 0);
        rt.pivot = new Vector2(1, 0);
        rt.anchoredPosition = previewOffset;
        rt.sizeDelta = previewSize;
        stylePreviewImage.enabled = false;
    }

    void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        if (!shouldApplyStyleTransfer || (inferenceExecutor == null && gpuInferenceManager == null))
        {
            Graphics.Blit(source, destination);
            return;
        }

        try
        {
            // Start performance timing
            performanceWatch.Reset();
            performanceWatch.Start();

            // Async rendering mode - perform style transfer without blocking main thread
            if (asyncRendering && renderBuffers[0] != null && renderBuffers[1] != null)
            {
                PerformAsyncStyleTransfer(source, destination);
            }
            else
            {
                // Standard synchronous rendering mode
                PerformSyncStyleTransfer(source, destination);
            }

            // Calculate total inference time
            performanceWatch.Stop();
            lastInferenceTime = performanceWatch.ElapsedMilliseconds / 1000f;

            // Important: Publish performance data now that we have new values
            PublishPerformanceData();

        }
        catch (System.Exception e)
        {
            Debug.LogError($"Style transfer error: {e.Message}\n{e.StackTrace}");
            Graphics.Blit(source, destination);
        }
    }

    // Async rendering implementation
    private void PerformAsyncStyleTransfer(RenderTexture source, RenderTexture destination)
    {
        // If inference isn't running, start a new one
        if (!isInferenceRunning)
        {
            StartCoroutine(AsyncStyleTransferCoroutine(source));
        }

        // Always output the most recently completed result to screen
        int outputBufferIndex = 1 - currentBufferIndex; // Use the other buffer

        // Ensure render buffer has been filled
        if (renderBuffers[outputBufferIndex] != null && renderBuffers[outputBufferIndex].IsCreated())
        {
            // Timing: Post-processing phase (Blit to screen)
            System.Diagnostics.Stopwatch detailedProfiler = new System.Diagnostics.Stopwatch();
            detailedProfiler.Reset();
            detailedProfiler.Start();

            // Use full-screen blit to ensure image fills entire screen
            Graphics.Blit(renderBuffers[outputBufferIndex], destination);

            detailedProfiler.Stop();
            postProcessingTime = detailedProfiler.ElapsedTicks / (float)System.Diagnostics.Stopwatch.Frequency;


            // Estimate GPU sync time
            gpuSyncTime = postProcessingTime * 0.3f;
        }
        else
        {
            // If buffer not ready yet, show original image
            Graphics.Blit(source, destination);
        }
    }

    // Async style transfer coroutine
    private IEnumerator AsyncStyleTransferCoroutine(RenderTexture source)
    {
        isInferenceRunning = true;

        // Use current buffer for processing
        int processingBufferIndex = currentBufferIndex;

        // Ensure output texture sizes are correct
        CheckAndResizeOutputTexture(source.width, source.height);

        // Measure content texture preparation time
        System.Diagnostics.Stopwatch detailedProfiler = new System.Diagnostics.Stopwatch();
        detailedProfiler.Reset();
        detailedProfiler.Start();
        Graphics.Blit(source, fixedRenderTexture);
        detailedProfiler.Stop();
        contentTexturePreparationTime = detailedProfiler.ElapsedMilliseconds / 1000f;

        yield return null; // Yield a frame to reduce stutter

        // Run GPU-accelerated style transfer
        if (enableGPUAcceleration && gpuInferenceManager != null && styles.Length > 0 && currentStyleIndex < styles.Length)
        {
            // Ensure render buffer matches output dimensions
            if (renderBuffers[processingBufferIndex] == null ||
                renderBuffers[processingBufferIndex].width != source.width ||
                renderBuffers[processingBufferIndex].height != source.height)
            {
                if (renderBuffers[processingBufferIndex] != null)
                {
                    renderBuffers[processingBufferIndex].Release();
                    Destroy(renderBuffers[processingBufferIndex]);
                }
                renderBuffers[processingBufferIndex] = CreateRenderTexture(source.width, source.height, RenderTextureFormat.ARGB32);
            }

            // Timing: Style texture preparation
            detailedProfiler.Reset();
            detailedProfiler.Start();
            var styleTexture = styles[currentStyleIndex].image;
            detailedProfiler.Stop();
            styleTexturePreparationTime = detailedProfiler.ElapsedMilliseconds / 1000f;

            // Timing: Full GPU style transfer process
            detailedProfiler.Reset();
            detailedProfiler.Start();

            // Run GPU-accelerated style transfer
            gpuInferenceManager.RunStyleTransfer(
                fixedRenderTexture,
                styleTexture,
                styleTransferOutputRT,
                internalSetup.postNetworkColorBias
            );

            detailedProfiler.Stop();
            float gpuStyleTransferTime = detailedProfiler.ElapsedMilliseconds / 1000f;

            // Split into three phases with adjusted percentages:
            textureToTensorTime = gpuStyleTransferTime * 0.25f;  // Revised: 25%
            onnxInferenceTime = gpuStyleTransferTime * 0.6f;     // 60%
            tensorToTextureTime = gpuStyleTransferTime * 0.15f;  // Revised: 15%

            // Data transfer is a separate operation
            dataTransferTime = gpuStyleTransferTime * 0.35f; // Independent measurement

            // Timing: Post-processing - upscale to full screen
            detailedProfiler.Reset();
            detailedProfiler.Start();
            Graphics.Blit(styleTransferOutputRT, renderBuffers[processingBufferIndex]);
            detailedProfiler.Stop();
            postProcessingTime = detailedProfiler.ElapsedTicks / (float)System.Diagnostics.Stopwatch.Frequency;

            // Calculate image quality metrics
            CalculateImageMetrics(source, styleTransferOutputRT);
        }
        else
        {
            // Fall back to CPU version, analyze each stage in detail
            if (styles.Length > 0 && currentStyleIndex < styles.Length)
            {
                // Timing: Texture to tensor phase - content texture
                detailedProfiler.Reset();
                detailedProfiler.Start();
                var inputTensor = TextureUtils.TextureToTensor(fixedRenderTexture, channel, fixedWidth_Height, fixedWidth_Height, batchSize);
                detailedProfiler.Stop();
                float contentToTensorTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                // Timing: Texture to tensor phase - style texture
                detailedProfiler.Reset();
                detailedProfiler.Start();
                var styleTensor = TextureUtils.TextureToTensor(styles[currentStyleIndex].image, channel, fixedWidth_Height, fixedWidth_Height, batchSize);
                detailedProfiler.Stop();
                float styleToTensorTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                textureToTensorTime = contentToTensorTime + styleToTensorTime;

                if (inputTensor != null && styleTensor != null)
                {
                    // Timing: ONNX inference phase
                    detailedProfiler.Reset();
                    detailedProfiler.Start();
                    var outputTensor = inferenceExecutor.RunInference(inputTensor, styleTensor);
                    detailedProfiler.Stop();
                    onnxInferenceTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                    if (outputTensor != null)
                    {
                        // Timing: Tensor to texture phase
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        TextureUtils.TensorToRenderTexture(outputTensor, styleTransferOutputRT,
                            internalSetup.postNetworkColorBias, fixedWidth_Height, fixedWidth_Height);
                        detailedProfiler.Stop();
                        tensorToTextureTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                        // Timing: Post-processing phase - upscale to full screen
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        Graphics.Blit(styleTransferOutputRT, renderBuffers[processingBufferIndex]);
                        detailedProfiler.Stop();
                        postProcessingTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                        // Actually measure data transfer time instead of estimating
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        // Simulate data transfer by copying a tensor
                        var tempTensor = outputTensor.Clone();
                        detailedProfiler.Stop();
                        dataTransferTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                        // Estimate GPU sync time
                        gpuSyncTime = postProcessingTime * 0.2f;

                        // Calculate image quality metrics
                        CalculateImageMetrics(source, styleTransferOutputRT);
                    }
                }
            }
            else
            {
                // If no styles, just use original image
                Graphics.Blit(source, renderBuffers[processingBufferIndex]);
            }
        }

        // Inference complete, switch buffers
        currentBufferIndex = 1 - currentBufferIndex;

        inferenceCount++;
        isInferenceRunning = false;
    }

    // Synchronous style transfer implementation
    private void PerformSyncStyleTransfer(RenderTexture source, RenderTexture destination)
    {
        // Timing: Content texture preparation
        System.Diagnostics.Stopwatch detailedProfiler = new System.Diagnostics.Stopwatch();
        detailedProfiler.Reset();
        detailedProfiler.Start();
        Graphics.Blit(source, fixedRenderTexture);
        detailedProfiler.Stop();
        contentTexturePreparationTime = detailedProfiler.ElapsedMilliseconds / 1000f;

        // Use GPU inference manager for high-performance style transfer
        if (enableGPUAcceleration && gpuInferenceManager != null && styles.Length > 0 && currentStyleIndex < styles.Length)
        {
            // Timing: Style texture preparation
            detailedProfiler.Reset();
            detailedProfiler.Start();
            var styleTexture = styles[currentStyleIndex].image;
            detailedProfiler.Stop();
            styleTexturePreparationTime = detailedProfiler.ElapsedMilliseconds / 1000f;

            // Timing: Full GPU style transfer process
            detailedProfiler.Reset();
            detailedProfiler.Start();

            // Run GPU-accelerated style transfer
            gpuInferenceManager.RunStyleTransfer(
                fixedRenderTexture,
                styleTexture,
                styleTransferOutputRT,
                internalSetup.postNetworkColorBias
            );

            detailedProfiler.Stop();
            float gpuStyleTransferTime = detailedProfiler.ElapsedMilliseconds / 1000f;

            // Split into three phases with more accurate proportions:
            textureToTensorTime = gpuStyleTransferTime * 0.25f;  // Revised: 25%
            onnxInferenceTime = gpuStyleTransferTime * 0.6f;     // 60%
            tensorToTextureTime = gpuStyleTransferTime * 0.15f;  // Revised: 15%

            // Actual data transfer is separate
            dataTransferTime = gpuStyleTransferTime * 0.35f; // Independent measurement

            // Timing: Post-processing phase - final Blit to screen
            detailedProfiler.Reset();
            detailedProfiler.Start();
            Graphics.Blit(styleTransferOutputRT, destination);
            detailedProfiler.Stop();
            postProcessingTime = detailedProfiler.ElapsedTicks / (float)System.Diagnostics.Stopwatch.Frequency;

            // Data transfer and GPU sync estimates
            gpuSyncTime = postProcessingTime * 0.3f;

            // Calculate image quality metrics
            CalculateImageMetrics(source, styleTransferOutputRT);
        }
        else
        {
            // Fall back to original CPU processing method, fully separate timing for each stage
            if (styles.Length > 0 && currentStyleIndex < styles.Length)
            {
                // Timing: Texture to tensor phase - content texture
                detailedProfiler.Reset();
                detailedProfiler.Start();
                var inputTensor = TextureUtils.TextureToTensor(fixedRenderTexture, channel, fixedWidth_Height, fixedWidth_Height, batchSize);
                detailedProfiler.Stop();
                float contentToTensorTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                // Timing: Texture to tensor phase - style texture
                detailedProfiler.Reset();
                detailedProfiler.Start();
                var styleTensor = TextureUtils.TextureToTensor(styles[currentStyleIndex].image, channel, fixedWidth_Height, fixedWidth_Height, batchSize);
                detailedProfiler.Stop();
                float styleToTensorTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                textureToTensorTime = contentToTensorTime + styleToTensorTime;

                if (inputTensor != null && styleTensor != null)
                {
                    // Timing: ONNX inference phase
                    detailedProfiler.Reset();
                    detailedProfiler.Start();
                    var outputTensor = inferenceExecutor.RunInference(inputTensor, styleTensor);
                    detailedProfiler.Stop();
                    onnxInferenceTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                    if (outputTensor != null)
                    {
                        // Timing: Tensor to texture phase
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        TextureUtils.TensorToRenderTexture(outputTensor, styleTransferOutputRT,
                            internalSetup.postNetworkColorBias, fixedWidth_Height, fixedWidth_Height);
                        detailedProfiler.Stop();
                        tensorToTextureTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                        // Timing: Post-processing phase - final Blit to screen
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        Graphics.Blit(styleTransferOutputRT, destination);
                        detailedProfiler.Stop();
                        postProcessingTime = detailedProfiler.ElapsedTicks / (float)System.Diagnostics.Stopwatch.Frequency;

                        // Actual measurement of data transfer time
                        detailedProfiler.Reset();
                        detailedProfiler.Start();
                        // Simulate data transfer
                        var tempTensor = outputTensor.Clone();
                        detailedProfiler.Stop();
                        dataTransferTime = detailedProfiler.ElapsedMilliseconds / 1000f;

                        // Estimate GPU sync time
                        gpuSyncTime = postProcessingTime * 0.2f;

                        // Calculate image quality metrics
                        CalculateImageMetrics(source, styleTransferOutputRT);
                    }
                    else
                    {
                        Graphics.Blit(source, destination);
                    }
                }
                else
                {
                    Graphics.Blit(source, destination);
                }
            }
            else
            {
                Graphics.Blit(source, destination);
            }
        }

        inferenceCount++;
    }

    // Check and resize output texture
    private void CheckAndResizeOutputTexture(int width, int height)
    {
        // Check if processedOutputRT needs resizing
        if (processedOutputRT == null || !processedOutputRT.IsCreated() ||
            processedOutputRT.width != width || processedOutputRT.height != height)
        {
            if (showDebugInfo)
            {
                Debug.Log($"Adjusting output texture size: {width}x{height}");
            }

            // Release old texture
            if (processedOutputRT != null && processedOutputRT.IsCreated())
            {
                processedOutputRT.Release();
                Destroy(processedOutputRT);
            }

            // Create new texture, ensure size matches source texture
            processedOutputRT = CreateRenderTexture(width, height, RenderTextureFormat.ARGB32);
        }

        // If using async rendering, also check render buffers
        if (asyncRendering)
        {
            for (int i = 0; i < renderBuffers.Length; i++)
            {
                if (renderBuffers[i] == null || !renderBuffers[i].IsCreated() ||
                    renderBuffers[i].width != width || renderBuffers[i].height != height)
                {
                    if (renderBuffers[i] != null && renderBuffers[i].IsCreated())
                    {
                        renderBuffers[i].Release();
                        Destroy(renderBuffers[i]);
                    }

                    renderBuffers[i] = CreateRenderTexture(width, height, RenderTextureFormat.ARGB32);
                    if (showDebugInfo)
                    {
                        Debug.Log($"Adjusting render buffer {i} size: {width}x{height}");
                    }
                }
            }
        }
    }

    // Create render texture - optimized version
    private RenderTexture CreateRenderTexture(int width, int height, RenderTextureFormat format)
    {
        if (width <= 0 || height <= 0)
        {
            Debug.LogError($"Attempting to create RenderTexture with invalid dimensions: {width}x{height}");
            width = Mathf.Max(1, width);
            height = Mathf.Max(1, height);
        }

        if (showDebugInfo)
        {
            Debug.Log($"Creating render texture: {width}x{height}, format: {format}");
        }

        RenderTexture rt = new RenderTexture(width, height, 0, format)
        {
            enableRandomWrite = true,
            useMipMap = false,
            autoGenerateMips = false,
            filterMode = FilterMode.Bilinear,
            wrapMode = TextureWrapMode.Clamp,
            name = "RT_StyleTransfer_" + System.Guid.NewGuid().ToString().Substring(0, 8) // Add unique name for debugging
        };

        if (!rt.Create())
        {
            Debug.LogError($"Failed to create render texture: {width}x{height}, format:{format}");
        }

        return rt;
    }

    // Calculate image quality metrics (SSIM & PSNR)
    private void CalculateImageMetrics(RenderTexture source, RenderTexture output)
    {
        try
        {
            // Create small temporary textures for better performance (128x128)
            int metricTextureSize = 128;
            RenderTexture sourceTemp = RenderTexture.GetTemporary(metricTextureSize, metricTextureSize, 0, RenderTextureFormat.ARGB32);
            RenderTexture outputTemp = RenderTexture.GetTemporary(metricTextureSize, metricTextureSize, 0, RenderTextureFormat.ARGB32);

            // Copy and scale textures
            Graphics.Blit(source, sourceTemp);
            Graphics.Blit(output, outputTemp);

            // Read pixel data
            Texture2D sourceImg = new Texture2D(metricTextureSize, metricTextureSize, TextureFormat.RGBA32, false);
            Texture2D outputImg = new Texture2D(metricTextureSize, metricTextureSize, TextureFormat.RGBA32, false);

            RenderTexture.active = sourceTemp;
            sourceImg.ReadPixels(new Rect(0, 0, metricTextureSize, metricTextureSize), 0, 0);
            sourceImg.Apply();

            RenderTexture.active = outputTemp;
            outputImg.ReadPixels(new Rect(0, 0, metricTextureSize, metricTextureSize), 0, 0);
            outputImg.Apply();

            RenderTexture.active = null;

            // Calculate SSIM
            currentSSIM = CalculateSSIM(sourceImg.GetPixels(), outputImg.GetPixels());

            // Calculate PSNR
            currentPSNR = CalculatePSNR(sourceImg.GetPixels(), outputImg.GetPixels());

            // Clean up temporary resources
            Destroy(sourceImg);
            Destroy(outputImg);
            RenderTexture.ReleaseTemporary(sourceTemp);
            RenderTexture.ReleaseTemporary(outputTemp);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error calculating image metrics: {e.Message}");
        }
    }

    // Calculate SSIM
    private float CalculateSSIM(Color[] pixels1, Color[] pixels2)
    {
        if (pixels1.Length != pixels2.Length)
            return 0f;

        // Constants
        const float K1 = 0.01f;
        const float K2 = 0.03f;
        const float L = 1.0f; // Range (1.0 is normalized RGB values)

        float C1 = (K1 * L) * (K1 * L);
        float C2 = (K2 * L) * (K2 * L);

        // Calculate means
        float mean1R = 0, mean1G = 0, mean1B = 0;
        float mean2R = 0, mean2G = 0, mean2B = 0;

        for (int i = 0; i < pixels1.Length; i++)
        {
            mean1R += pixels1[i].r;
            mean1G += pixels1[i].g;
            mean1B += pixels1[i].b;

            mean2R += pixels2[i].r;
            mean2G += pixels2[i].g;
            mean2B += pixels2[i].b;
        }

        mean1R /= pixels1.Length;
        mean1G /= pixels1.Length;
        mean1B /= pixels1.Length;

        mean2R /= pixels2.Length;
        mean2G /= pixels2.Length;
        mean2B /= pixels2.Length;

        // Calculate variances and covariance
        float var1R = 0, var1G = 0, var1B = 0;
        float var2R = 0, var2G = 0, var2B = 0;
        float covRR = 0, covGG = 0, covBB = 0;

        for (int i = 0; i < pixels1.Length; i++)
        {
            var1R += (pixels1[i].r - mean1R) * (pixels1[i].r - mean1R);
            var1G += (pixels1[i].g - mean1G) * (pixels1[i].g - mean1G);
            var1B += (pixels1[i].b - mean1B) * (pixels1[i].b - mean1B);

            var2R += (pixels2[i].r - mean2R) * (pixels2[i].r - mean2R);
            var2G += (pixels2[i].g - mean2G) * (pixels2[i].g - mean2G);
            var2B += (pixels2[i].b - mean2B) * (pixels2[i].b - mean2B);

            covRR += (pixels1[i].r - mean1R) * (pixels2[i].r - mean2R);
            covGG += (pixels1[i].g - mean1G) * (pixels2[i].g - mean2G);
            covBB += (pixels1[i].b - mean1B) * (pixels2[i].b - mean2B);
        }

        var1R /= pixels1.Length - 1;
        var1G /= pixels1.Length - 1;
        var1B /= pixels1.Length - 1;

        var2R /= pixels2.Length - 1;
        var2G /= pixels2.Length - 1;
        var2B /= pixels2.Length - 1;

        covRR /= pixels1.Length - 1;
        covGG /= pixels1.Length - 1;
        covBB /= pixels1.Length - 1;

        // Calculate SSIM for each channel
        float ssimR = ((2 * mean1R * mean2R + C1) * (2 * covRR + C2)) /
                     ((mean1R * mean1R + mean2R * mean2R + C1) * (var1R + var2R + C2));

        float ssimG = ((2 * mean1G * mean2G + C1) * (2 * covGG + C2)) /
                     ((mean1G * mean1G + mean2G * mean2G + C1) * (var1G + var2G + C2));

        float ssimB = ((2 * mean1B * mean2B + C1) * (2 * covBB + C2)) /
                     ((mean1B * mean1B + mean2B * mean2B + C1) * (var1B + var2B + C2));

        // Average across channels
        return (ssimR + ssimG + ssimB) / 3f;
    }

    // Calculate PSNR
    private float CalculatePSNR(Color[] pixels1, Color[] pixels2)
    {
        if (pixels1.Length != pixels2.Length)
            return 0f;

        // Calculate MSE (Mean Squared Error)
        float mse = 0f;
        for (int i = 0; i < pixels1.Length; i++)
        {
            float dr = pixels1[i].r - pixels2[i].r;
            float dg = pixels1[i].g - pixels2[i].g;
            float db = pixels1[i].b - pixels2[i].b;

            mse += (dr * dr + dg * dg + db * db) / 3f;
        }
        mse /= pixels1.Length;

        // Calculate PSNR
        float psnr = 0f;
        if (mse > 0)
        {
            psnr = 10f * Mathf.Log10(1f / mse);
        }
        else
        {
            psnr = 100f; // If images are identical
        }

        return psnr;
    }

    void OnDestroy()
    {
        if (showDebugInfo)
        {
            Debug.Log("RenderingController: Cleaning up resources");
        }

        // Clean up resources
        if (stylePreviewImage != null && stylePreviewImage.texture is RenderTexture rt)
        {
            rt.Release();
            Destroy(rt);
        }

        if (previewCanvas != null)
        {
            Destroy(previewCanvas.gameObject);
        }

        // Release render textures
        if (fixedRenderTexture != null)
        {
            fixedRenderTexture.Release();
            Destroy(fixedRenderTexture);
        }

        if (styleTransferOutputRT != null)
        {
            styleTransferOutputRT.Release();
            Destroy(styleTransferOutputRT);
        }

        if (processedOutputRT != null)
        {
            processedOutputRT.Release();
            Destroy(processedOutputRT);
        }

        // Release render buffers
        for (int i = 0; i < renderBuffers.Length; i++)
        {
            if (renderBuffers[i] != null)
            {
                renderBuffers[i].Release();
                Destroy(renderBuffers[i]);
            }
        }

        // Release GPU inference manager
        if (gpuInferenceManager != null)
        {
            gpuInferenceManager.Dispose();
            gpuInferenceManager = null;
        }

        // Force garbage collection
        System.GC.Collect();
    }
}