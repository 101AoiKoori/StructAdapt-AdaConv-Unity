using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// CUDA系统信息检测器 - 不依赖ONNX Runtime
/// 使用Unity自带方法检测CUDA、cuDNN、GPU等系统信息
/// </summary>
public class CudaSystemDetector : MonoBehaviour
{
    [Header("检测设置")]
    [SerializeField] private bool autoDetectOnStart = true;
    [SerializeField] private bool detailedLogging = true;
    [SerializeField] private bool saveToFile = true;
    [SerializeField] private string reportFileName = "SystemDetectionReport.txt";

    private Dictionary<string, object> detectionResults = new Dictionary<string, object>();
    private StringBuilder reportBuilder = new StringBuilder();

    void Start()
    {
        if (autoDetectOnStart)
        {
            DetectSystemInfo();
        }
    }

    /// <summary>
    /// 执行完整的系统检测
    /// </summary>
    [ContextMenu("检测系统信息")]
    public void DetectSystemInfo()
    {
        LogInfo("============================================");
        LogInfo("开始检测CUDA系统信息");
        LogInfo("============================================");

        detectionResults.Clear();
        reportBuilder.Clear();

        // 检测基本系统信息
        DetectBasicSystemInfo();

        // 检测GPU信息
        DetectGPUInfo();

        // 检测NVIDIA驱动
        DetectNvidiaDriver();

        // 检测CUDA安装
        DetectCudaInstallation();

        // 检测cuDNN安装
        DetectCudnnInstallation();

        // 检测Visual C++ Redistributable
        DetectVCRedistributable();

        // 检测环境变量
        DetectEnvironmentVariables();

        // 检测关键DLL文件
        DetectCriticalDLLs();

        // 检测.NET运行时
        DetectDotNetRuntimes();

        // 生成最终报告
        GenerateFinalReport();

        LogInfo("============================================");
        LogInfo("系统检测完成！");
        LogInfo("============================================");

        if (saveToFile)
        {
            SaveReportToFile();
        }
    }

    /// <summary>
    /// 检测基本系统信息
    /// </summary>
    private void DetectBasicSystemInfo()
    {
        LogInfo("\n=== 基本系统信息 ===");

        try
        {
            var systemInfo = new Dictionary<string, object>
            {
                ["操作系统"] = SystemInfo.operatingSystem,
                ["系统内存"] = $"{SystemInfo.systemMemorySize} MB",
                ["处理器类型"] = SystemInfo.processorType,
                ["处理器核心数"] = SystemInfo.processorCount,
                ["Unity版本"] = Application.unityVersion,
                ["平台"] = Application.platform.ToString(),
                ["系统语言"] = Application.systemLanguage.ToString(),
                ["64位系统"] = Environment.Is64BitOperatingSystem,
                ["64位进程"] = Environment.Is64BitProcess,
                [".NET版本"] = Environment.Version.ToString()
            };

            detectionResults["系统信息"] = systemInfo;

            foreach (var kvp in systemInfo)
            {
                LogInfo($"{kvp.Key}: {kvp.Value}");
            }
        }
        catch (Exception e)
        {
            LogError($"检测基本系统信息时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 检测GPU信息
    /// </summary>
    private void DetectGPUInfo()
    {
        LogInfo("\n=== GPU信息 ===");

        try
        {
            var gpuInfo = new Dictionary<string, object>
            {
                ["显卡名称"] = SystemInfo.graphicsDeviceName,
                ["显卡厂商"] = SystemInfo.graphicsDeviceVendor,
                ["显卡类型"] = SystemInfo.graphicsDeviceType.ToString(),
                ["显卡版本"] = SystemInfo.graphicsDeviceVersion,
                ["显存大小"] = $"{SystemInfo.graphicsMemorySize} MB",
                ["最大纹理尺寸"] = SystemInfo.maxTextureSize,
                ["支持计算着色器"] = SystemInfo.supportsComputeShaders,
                ["计算着色器最大工作组大小"] = SystemInfo.maxComputeWorkGroupSize,
                ["支持GPU实例化"] = SystemInfo.supportsInstancing,
                ["支持多渲染目标"] = SystemInfo.supportedRenderTargetCount
            };

            detectionResults["GPU信息"] = gpuInfo;

            foreach (var kvp in gpuInfo)
            {
                LogInfo($"{kvp.Key}: {kvp.Value}");
            }

            // 判断是否为NVIDIA GPU
            bool isNvidiaGPU = SystemInfo.graphicsDeviceName.ToLower().Contains("nvidia") ||
                              SystemInfo.graphicsDeviceVendor.ToLower().Contains("nvidia");

            LogInfo($"NVIDIA GPU: {(isNvidiaGPU ? "是" : "否")}");
            detectionResults["是否NVIDIA GPU"] = isNvidiaGPU;

            if (!isNvidiaGPU)
            {
                LogWarning("⚠️ 未检测到NVIDIA GPU，CUDA加速不可用");
            }
        }
        catch (Exception e)
        {
            LogError($"检测GPU信息时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 检测NVIDIA驱动版本
    /// </summary>
    private void DetectNvidiaDriver()
    {
        LogInfo("\n=== NVIDIA驱动信息 ===");

        try
        {
            string driverVersion = "未检测到";
            string cudaDriverVersion = "未检测到";

            // 尝试通过nvidia-smi检测
            try
            {
                using (var process = new System.Diagnostics.Process())
                {
                    process.StartInfo.FileName = "nvidia-smi";
                    process.StartInfo.Arguments = "--query-gpu=driver_version --format=csv,noheader,nounits";
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.RedirectStandardError = true;
                    process.StartInfo.CreateNoWindow = true;

                    process.Start();
                    process.WaitForExit(5000); // 5秒超时

                    if (process.ExitCode == 0)
                    {
                        string output = process.StandardOutput.ReadToEnd().Trim();
                        if (!string.IsNullOrEmpty(output))
                        {
                            driverVersion = output;
                            LogInfo($"✅ 通过nvidia-smi检测到驱动版本: {driverVersion}");
                        }
                    }
                    else
                    {
                        string error = process.StandardError.ReadToEnd();
                        LogWarning($"nvidia-smi执行失败: {error}");
                    }
                }
            }
            catch (Exception ex)
            {
                LogWarning($"nvidia-smi检测失败: {ex.Message}");
            }

            // 尝试检测NVIDIA驱动文件
            try
            {
                string[] driverPaths = {
                    @"C:\Windows\System32\DriverStore\FileRepository",
                    @"C:\Windows\System32\drivers"
                };

                foreach (string basePath in driverPaths)
                {
                    if (Directory.Exists(basePath))
                    {
                        try
                        {
                            var files = Directory.GetFiles(basePath, "*nv*.sys", SearchOption.AllDirectories);
                            if (files.Length > 0)
                            {
                                LogInfo($"✅ 找到NVIDIA驱动文件: {files.Length} 个");
                                break;
                            }
                        }
                        catch
                        {
                            // 忽略权限错误
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogWarning($"检测驱动文件失败: {ex.Message}");
            }

            var driverInfo = new Dictionary<string, object>
            {
                ["NVIDIA驱动版本"] = driverVersion,
                ["CUDA驱动版本"] = cudaDriverVersion
            };

            detectionResults["驱动信息"] = driverInfo;

            LogInfo($"NVIDIA驱动版本: {driverVersion}");

            if (driverVersion == "未检测到")
            {
                LogError("❌ 未检测到NVIDIA驱动，请安装最新驱动");
            }
        }
        catch (Exception ex)
        {
            LogError($"检测NVIDIA驱动时出错: {ex.Message}");
        }
    }

    /// <summary>
    /// 检测CUDA安装
    /// </summary>
    private void DetectCudaInstallation()
    {
        LogInfo("\n=== CUDA安装检测 ===");

        try
        {
            var cudaVersions = new List<string>();
            string cudaPath = "";
            bool cudaInstalled = false;

            // 检查环境变量CUDA_PATH
            string cudaPathEnv = Environment.GetEnvironmentVariable("CUDA_PATH");
            if (!string.IsNullOrEmpty(cudaPathEnv))
            {
                LogInfo($"发现CUDA_PATH环境变量: {cudaPathEnv}");
                cudaPath = cudaPathEnv;
                cudaInstalled = true;
            }

            // 检查常见CUDA安装路径
            string[] commonCudaPaths = {
                @"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
                @"C:\CUDA"
            };

            foreach (string basePath in commonCudaPaths)
            {
                if (Directory.Exists(basePath))
                {
                    LogInfo($"发现CUDA基础路径: {basePath}");
                    try
                    {
                        var subdirs = Directory.GetDirectories(basePath);
                        foreach (string subdir in subdirs)
                        {
                            string versionName = Path.GetFileName(subdir);
                            if (versionName.StartsWith("v"))
                            {
                                cudaVersions.Add(versionName);
                                LogInfo($"发现CUDA版本: {versionName} at {subdir}");

                                // 检查关键文件
                                string nvccPath = Path.Combine(subdir, "bin", "nvcc.exe");
                                string cudartPath = Path.Combine(subdir, "bin", "cudart64_12.dll");
                                string cudart11Path = Path.Combine(subdir, "bin", "cudart64_11.dll");

                                if (File.Exists(nvccPath))
                                {
                                    LogInfo($"  ✅ 找到nvcc编译器: {nvccPath}");
                                    cudaInstalled = true;
                                    if (string.IsNullOrEmpty(cudaPath))
                                        cudaPath = subdir;
                                }

                                if (File.Exists(cudartPath))
                                {
                                    LogInfo($"  ✅ 找到CUDA Runtime 12.x: {cudartPath}");
                                }
                                else if (File.Exists(cudart11Path))
                                {
                                    LogInfo($"  ✅ 找到CUDA Runtime 11.x: {cudart11Path}");
                                }
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        LogWarning($"遍历CUDA目录时出错: {e.Message}");
                    }
                }
            }

            // 尝试通过nvcc检测版本
            string nvccVersion = "未检测到";
            try
            {
                using (var process = new System.Diagnostics.Process())
                {
                    process.StartInfo.FileName = "nvcc";
                    process.StartInfo.Arguments = "--version";
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.RedirectStandardError = true;
                    process.StartInfo.CreateNoWindow = true;

                    process.Start();
                    process.WaitForExit(5000);

                    if (process.ExitCode == 0)
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        LogInfo($"nvcc版本信息:\n{output}");
                        nvccVersion = "已检测到";
                    }
                }
            }
            catch (Exception ex)
            {
                LogWarning($"nvcc检测失败: {ex.Message}");
            }

            var cudaInfo = new Dictionary<string, object>
            {
                ["CUDA已安装"] = cudaInstalled,
                ["CUDA路径"] = cudaPath,
                ["发现的版本"] = cudaVersions,
                ["nvcc可用"] = nvccVersion != "未检测到"
            };

            detectionResults["CUDA信息"] = cudaInfo;

            if (cudaInstalled)
            {
                LogInfo($"✅ CUDA已安装，路径: {cudaPath}");
                LogInfo($"发现的版本: {string.Join(", ", cudaVersions)}");
            }
            else
            {
                LogError("❌ 未检测到CUDA安装");
            }
        }
        catch (Exception e)
        {
            LogError($"检测CUDA安装时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 检测cuDNN安装
    /// </summary>
    private void DetectCudnnInstallation()
    {
        LogInfo("\n=== cuDNN安装检测 ===");

        try
        {
            var cudnnVersions = new List<string>();
            bool cudnnInstalled = false;
            string cudnnPath = "";

            // 检查环境变量CUDNN_PATH
            string cudnnPathEnv = Environment.GetEnvironmentVariable("CUDNN_PATH");
            if (!string.IsNullOrEmpty(cudnnPathEnv))
            {
                LogInfo($"发现CUDNN_PATH环境变量: {cudnnPathEnv}");
                cudnnPath = cudnnPathEnv;
            }

            // 检查常见cuDNN安装路径
            string[] commonCudnnPaths = {
                @"C:\Program Files\NVIDIA\CUDNN",
                @"C:\cudnn",
                @"C:\tools\cuda"
            };

            foreach (string basePath in commonCudnnPaths)
            {
                if (Directory.Exists(basePath))
                {
                    LogInfo($"发现cuDNN基础路径: {basePath}");
                    try
                    {
                        // 递归搜索cudnn DLL文件
                        SearchCudnnFiles(basePath, cudnnVersions, ref cudnnInstalled, ref cudnnPath);
                    }
                    catch (Exception e)
                    {
                        LogWarning($"搜索cuDNN文件时出错: {e.Message}");
                    }
                }
            }

            // 在PATH中搜索cuDNN DLL
            string pathVariable = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathVariable))
            {
                string[] paths = pathVariable.Split(';');
                foreach (string path in paths)
                {
                    if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                    {
                        // 检查是否包含cuDNN相关文件
                        string[] cudnnFiles = {
                            "cudnn64_8.dll",
                            "cudnn64_9.dll",
                            "cudnn.dll"
                        };

                        foreach (string dllName in cudnnFiles)
                        {
                            string dllPath = Path.Combine(path, dllName);
                            if (File.Exists(dllPath))
                            {
                                LogInfo($"✅ 在PATH中找到cuDNN: {dllPath}");
                                cudnnInstalled = true;
                                cudnnVersions.Add($"在PATH中: {dllName}");
                                if (string.IsNullOrEmpty(cudnnPath))
                                    cudnnPath = path;
                            }
                        }
                    }
                }
            }

            var cudnnInfo = new Dictionary<string, object>
            {
                ["cuDNN已安装"] = cudnnInstalled,
                ["cuDNN路径"] = cudnnPath,
                ["发现的版本"] = cudnnVersions
            };

            detectionResults["cuDNN信息"] = cudnnInfo;

            if (cudnnInstalled)
            {
                LogInfo($"✅ cuDNN已安装，路径: {cudnnPath}");
                LogInfo($"发现的版本: {string.Join(", ", cudnnVersions)}");
            }
            else
            {
                LogError("❌ 未检测到cuDNN安装");
            }
        }
        catch (Exception e)
        {
            LogError($"检测cuDNN安装时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 递归搜索cuDNN文件
    /// </summary>
    private void SearchCudnnFiles(string directory, List<string> versions, ref bool installed, ref string path)
    {
        try
        {
            // 检查当前目录中的cuDNN文件
            string[] cudnnFiles = {
                "cudnn64_8.dll",
                "cudnn64_9.dll",
                "cudnn.dll",
                "libcudnn.so",
                "libcudnn.so.8",
                "libcudnn.so.9"
            };

            foreach (string fileName in cudnnFiles)
            {
                string filePath = Path.Combine(directory, fileName);
                if (File.Exists(filePath))
                {
                    LogInfo($"✅ 找到cuDNN文件: {filePath}");
                    versions.Add($"{Path.GetFileName(directory)}: {fileName}");
                    installed = true;
                    if (string.IsNullOrEmpty(path))
                        path = directory;
                }
            }

            // 递归搜索子目录（限制深度避免过度搜索）
            var subdirs = Directory.GetDirectories(directory);
            foreach (string subdir in subdirs)
            {
                if (Path.GetFileName(subdir).StartsWith("v") ||
                    Path.GetFileName(subdir).Contains("cudnn") ||
                    Path.GetFileName(subdir).Contains("bin"))
                {
                    SearchCudnnFiles(subdir, versions, ref installed, ref path);
                }
            }
        }
        catch
        {
            // 忽略权限错误等
        }
    }

    /// <summary>
    /// 检测Visual C++ Redistributable
    /// </summary>
    private void DetectVCRedistributable()
    {
        LogInfo("\n=== Visual C++ Redistributable检测 ===");

        try
        {
            var vcVersions = new List<string>();

            // 检查常见的VC++ Runtime DLL文件
            string[] vcDlls = {
                "msvcp140.dll",
                "vcruntime140.dll",
                "vcruntime140_1.dll",
                "msvcp120.dll",
                "msvcr120.dll",
                "msvcp110.dll",
                "msvcr110.dll"
            };

            string systemPath = Environment.GetFolderPath(Environment.SpecialFolder.System);

            foreach (string dllName in vcDlls)
            {
                string dllPath = Path.Combine(systemPath, dllName);
                if (File.Exists(dllPath))
                {
                    try
                    {
                        var fileInfo = new FileInfo(dllPath);
                        string version = "未知版本";

                        // 根据DLL名称判断版本
                        if (dllName.Contains("140"))
                            version = "Visual C++ 2015-2022";
                        else if (dllName.Contains("120"))
                            version = "Visual C++ 2013";
                        else if (dllName.Contains("110"))
                            version = "Visual C++ 2012";

                        vcVersions.Add($"{dllName} - {version} (修改时间: {fileInfo.LastWriteTime:yyyy-MM-dd})");
                        LogInfo($"✅ 找到: {dllName} - {version}");
                    }
                    catch (Exception ex)
                    {
                        LogWarning($"检查文件 {dllName} 时出错: {ex.Message}");
                    }
                }
            }

            // 检查程序文件目录中的VC++ Redistributable
            try
            {
                string[] programPaths = {
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles),
                    Environment.GetFolderPath(Environment.SpecialFolder.ProgramFilesX86)
                };

                foreach (string programPath in programPaths)
                {
                    if (!string.IsNullOrEmpty(programPath))
                    {
                        string vcRedistPath = Path.Combine(programPath, "Microsoft Visual Studio");
                        if (Directory.Exists(vcRedistPath))
                        {
                            try
                            {
                                var subdirs = Directory.GetDirectories(vcRedistPath, "*", SearchOption.AllDirectories);
                                foreach (string subdir in subdirs)
                                {
                                    string dirName = Path.GetFileName(subdir);
                                    if (dirName.Contains("Redistributable") || dirName.Contains("VC"))
                                    {
                                        vcVersions.Add($"安装目录: {subdir}");
                                        LogInfo($"✅ 找到安装目录: {subdir}");
                                    }
                                }
                            }
                            catch
                            {
                                // 忽略权限错误
                            }
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                LogWarning($"检查程序文件目录时出错: {ex.Message}");
            }

            var vcInfo = new Dictionary<string, object>
            {
                ["已安装版本"] = vcVersions,
                ["安装数量"] = vcVersions.Count
            };

            detectionResults["VC++ Redistributable"] = vcInfo;

            if (vcVersions.Count > 0)
            {
                LogInfo($"✅ 检测到 {vcVersions.Count} 个VC++ Redistributable相关项");
            }
            else
            {
                LogError("❌ 未检测到Visual C++ Redistributable");
            }
        }
        catch (Exception ex)
        {
            LogError($"检测VC++ Redistributable时出错: {ex.Message}");
        }
    }

    /// <summary>
    /// 检测关键环境变量
    /// </summary>
    private void DetectEnvironmentVariables()
    {
        LogInfo("\n=== 环境变量检测 ===");

        try
        {
            string[] importantVars = {
                "CUDA_PATH",
                "CUDA_PATH_V12_6",
                "CUDA_PATH_V12_5",
                "CUDA_PATH_V11_8",
                "CUDNN_PATH",
                "PATH"
            };

            var envInfo = new Dictionary<string, object>();

            foreach (string varName in importantVars)
            {
                string value = Environment.GetEnvironmentVariable(varName);
                envInfo[varName] = value ?? "未设置";

                if (varName == "PATH")
                {
                    LogInfo($"{varName}: [包含 {value?.Split(';').Length ?? 0} 个路径]");

                    // 检查PATH中是否包含CUDA相关路径
                    if (!string.IsNullOrEmpty(value))
                    {
                        bool hasCudaPath = value.ToLower().Contains("cuda");
                        bool hasCudnnPath = value.ToLower().Contains("cudnn");
                        LogInfo($"  PATH包含CUDA路径: {hasCudaPath}");
                        LogInfo($"  PATH包含cuDNN路径: {hasCudnnPath}");

                        envInfo["PATH包含CUDA"] = hasCudaPath;
                        envInfo["PATH包含cuDNN"] = hasCudnnPath;
                    }
                }
                else
                {
                    LogInfo($"{varName}: {envInfo[varName]}");
                }
            }

            detectionResults["环境变量"] = envInfo;
        }
        catch (Exception e)
        {
            LogError($"检测环境变量时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 检测关键DLL文件
    /// </summary>
    private void DetectCriticalDLLs()
    {
        LogInfo("\n=== 关键DLL文件检测 ===");

        try
        {
            string[] criticalDlls = {
                "cudart64_12.dll",
                "cudart64_11.dll",
                "cublas64_12.dll",
                "cublas64_11.dll",
                "cublasLt64_12.dll",
                "cudnn64_9.dll",
                "cudnn64_8.dll",
                "nvrtc64_120_0.dll",
                "nvrtc64_112_0.dll"
            };

            var dllInfo = new Dictionary<string, object>();
            int foundCount = 0;

            foreach (string dllName in criticalDlls)
            {
                bool found = IsDllAvailable(dllName);
                dllInfo[dllName] = found ? "✅ 找到" : "❌ 未找到";

                if (found)
                {
                    foundCount++;
                    LogInfo($"✅ {dllName}: 可用");
                }
                else
                {
                    LogInfo($"❌ {dllName}: 不可用");
                }
            }

            dllInfo["找到的DLL数量"] = foundCount;
            dllInfo["总计DLL数量"] = criticalDlls.Length;

            detectionResults["关键DLL"] = dllInfo;

            LogInfo($"关键DLL检测: {foundCount}/{criticalDlls.Length} 个可用");
        }
        catch (Exception e)
        {
            LogError($"检测关键DLL时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 检查DLL是否可用
    /// </summary>
    private bool IsDllAvailable(string dllName)
    {
        try
        {
            // 在PATH中搜索
            string pathVariable = Environment.GetEnvironmentVariable("PATH");
            if (!string.IsNullOrEmpty(pathVariable))
            {
                string[] paths = pathVariable.Split(';');
                foreach (string path in paths)
                {
                    if (!string.IsNullOrEmpty(path) && Directory.Exists(path))
                    {
                        string dllPath = Path.Combine(path, dllName);
                        if (File.Exists(dllPath))
                        {
                            return true;
                        }
                    }
                }
            }

            // 在系统目录中搜索
            string[] systemPaths = {
                Environment.GetFolderPath(Environment.SpecialFolder.System),
                Environment.GetFolderPath(Environment.SpecialFolder.SystemX86),
                Environment.CurrentDirectory
            };

            foreach (string systemPath in systemPaths)
            {
                if (!string.IsNullOrEmpty(systemPath))
                {
                    string dllPath = Path.Combine(systemPath, dllName);
                    if (File.Exists(dllPath))
                    {
                        return true;
                    }
                }
            }

            return false;
        }
        catch
        {
            return false;
        }
    }

    /// <summary>
    /// 检测.NET运行时
    /// </summary>
    private void DetectDotNetRuntimes()
    {
        LogInfo("\n=== .NET运行时检测 ===");

        try
        {
            var dotnetInfo = new Dictionary<string, object>
            {
                ["当前.NET版本"] = Environment.Version.ToString(),
                ["Framework版本"] = Environment.Version.ToString(),
                ["运行时版本"] = System.Runtime.InteropServices.RuntimeInformation.FrameworkDescription
            };

            detectionResults[".NET运行时"] = dotnetInfo;

            foreach (var kvp in dotnetInfo)
            {
                LogInfo($"{kvp.Key}: {kvp.Value}");
            }

            // 尝试检测已安装的.NET Framework版本
            try
            {
                using (var process = new System.Diagnostics.Process())
                {
                    process.StartInfo.FileName = "dotnet";
                    process.StartInfo.Arguments = "--list-runtimes";
                    process.StartInfo.UseShellExecute = false;
                    process.StartInfo.RedirectStandardOutput = true;
                    process.StartInfo.RedirectStandardError = true;
                    process.StartInfo.CreateNoWindow = true;

                    process.Start();
                    process.WaitForExit(5000);

                    if (process.ExitCode == 0)
                    {
                        string output = process.StandardOutput.ReadToEnd();
                        LogInfo($"已安装的.NET运行时:\n{output}");
                    }
                }
            }
            catch (Exception ex)
            {
                LogWarning($"检测.NET运行时失败: {ex.Message}");
            }
        }
        catch (Exception e)
        {
            LogError($"检测.NET运行时时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 生成最终报告
    /// </summary>
    private void GenerateFinalReport()
    {
        LogInfo("\n=== 检测总结 ===");

        try
        {
            // 分析检测结果并给出建议
            bool isNvidiaGPU = detectionResults.ContainsKey("是否NVIDIA GPU") &&
                              (bool)detectionResults["是否NVIDIA GPU"];

            bool cudaInstalled = false;
            bool cudnnInstalled = false;

            if (detectionResults.ContainsKey("CUDA信息"))
            {
                var cudaInfo = (Dictionary<string, object>)detectionResults["CUDA信息"];
                cudaInstalled = (bool)cudaInfo["CUDA已安装"];
            }

            if (detectionResults.ContainsKey("cuDNN信息"))
            {
                var cudnnInfo = (Dictionary<string, object>)detectionResults["cuDNN信息"];
                cudnnInstalled = (bool)cudnnInfo["cuDNN已安装"];
            }

            LogInfo($"GPU兼容性: {(isNvidiaGPU ? "✅ NVIDIA GPU" : "❌ 非NVIDIA GPU")}");
            LogInfo($"CUDA状态: {(cudaInstalled ? "✅ 已安装" : "❌ 未安装")}");
            LogInfo($"cuDNN状态: {(cudnnInstalled ? "✅ 已安装" : "❌ 未安装")}");

            // 给出建议
            LogInfo("\n=== 建议 ===");

            if (!isNvidiaGPU)
            {
                LogError("❌ 系统没有NVIDIA GPU，无法使用CUDA加速");
            }
            else if (!cudaInstalled)
            {
                LogError("❌ 需要安装CUDA Toolkit");
                LogInfo("建议: 从 https://developer.nvidia.com/cuda-downloads 下载安装CUDA 12.x");
            }
            else if (!cudnnInstalled)
            {
                LogError("❌ 需要安装cuDNN");
                LogInfo("建议: 从 https://developer.nvidia.com/cudnn 下载安装cuDNN 9.x");
            }
            else
            {
                LogInfo("✅ 系统配置看起来正常，CUDA加速应该可用");
            }

            var summary = new Dictionary<string, object>
            {
                ["NVIDIA GPU"] = isNvidiaGPU,
                ["CUDA已安装"] = cudaInstalled,
                ["cuDNN已安装"] = cudnnInstalled,
                ["可用于ONNX Runtime GPU"] = isNvidiaGPU && cudaInstalled && cudnnInstalled
            };

            detectionResults["检测总结"] = summary;
        }
        catch (Exception e)
        {
            LogError($"生成最终报告时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 保存报告到文件
    /// </summary>
    private void SaveReportToFile()
    {
        try
        {
            string filePath = Path.Combine(Application.persistentDataPath, reportFileName);
            File.WriteAllText(filePath, reportBuilder.ToString(), Encoding.UTF8);
            LogInfo($"✅ 检测报告已保存到: {filePath}");
        }
        catch (Exception e)
        {
            LogError($"保存报告文件时出错: {e.Message}");
        }
    }

    /// <summary>
    /// 获取检测结果
    /// </summary>
    public Dictionary<string, object> GetDetectionResults()
    {
        return new Dictionary<string, object>(detectionResults);
    }

    /// <summary>
    /// 获取报告文本
    /// </summary>
    public string GetReportText()
    {
        return reportBuilder.ToString();
    }

    /// <summary>
    /// 日志输出 - 信息
    /// </summary>
    private void LogInfo(string message)
    {
        if (detailedLogging)
        {
            Debug.Log($"[SystemDetector] {message}");
        }
        reportBuilder.AppendLine($"[INFO] {message}");
    }

    /// <summary>
    /// 日志输出 - 警告
    /// </summary>
    private void LogWarning(string message)
    {
        Debug.LogWarning($"[SystemDetector] {message}");
        reportBuilder.AppendLine($"[WARNING] {message}");
    }

    /// <summary>
    /// 日志输出 - 错误
    /// </summary>
    private void LogError(string message)
    {
        Debug.LogError($"[SystemDetector] {message}");
        reportBuilder.AppendLine($"[ERROR] {message}");
    }

    // 编辑器辅助方法
#if UNITY_EDITOR
    [ContextMenu("打开报告文件夹")]
    private void OpenReportFolder()
    {
        string folderPath = Application.persistentDataPath;
        if (Directory.Exists(folderPath))
        {
            System.Diagnostics.Process.Start("explorer.exe", folderPath);
        }
    }

    [ContextMenu("复制报告到剪贴板")]
    private void CopyReportToClipboard()
    {
        if (reportBuilder.Length > 0)
        {
            GUIUtility.systemCopyBuffer = reportBuilder.ToString();
            Debug.Log("报告已复制到剪贴板");
        }
    }
#endif
}