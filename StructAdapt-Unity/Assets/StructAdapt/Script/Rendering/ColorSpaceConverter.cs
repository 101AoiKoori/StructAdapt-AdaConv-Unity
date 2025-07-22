using UnityEngine;

public class ColorSpaceConverter
{
    private ComputeShader converterCS;
    private int rgbToSRGBKernel;
    private int srgbToRGBKernel;

    // ״̬
    private bool isInitialized = false;
    private bool debugLogging = false;

    public ColorSpaceConverter()
    {
        try
        {
            // ���� Compute Shader
            converterCS = Resources.Load<ComputeShader>("Shaders/ColorSpaceConverter");
            if (converterCS == null)
            {
                Debug.LogError("�޷����� ColorSpaceConverter ������ɫ������ȷ����λ�� Resources/Shaders Ŀ¼��");
                return;
            }

            // �����ں�
            rgbToSRGBKernel = converterCS.FindKernel("CSRGBToSRGB");
            srgbToRGBKernel = converterCS.FindKernel("CSSRGBToRGB");

            if (rgbToSRGBKernel == -1)
            {
                Debug.LogError("�޷��ҵ�������ɫ���ں� CSRGBToSRGB");
            }
            if (srgbToRGBKernel == -1)
            {
                Debug.LogError("�޷��ҵ�������ɫ���ں� CSSRGBToRGB");
            }

            isInitialized = rgbToSRGBKernel != -1 && srgbToRGBKernel != -1;

            if (debugLogging && isInitialized)
            {
                Debug.Log("ColorSpaceConverter ��ʼ���ɹ�");
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ColorSpaceConverter ��ʼ������: {e.Message}");
            isInitialized = false;
        }
    }

    /// <summary>
    /// ��RGB���Կռ�ת��ΪsRGB�ռ�
    /// </summary>
    public void ConvertRGBToSRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        if (!isInitialized || converterCS == null)
        {
            Debug.LogError("ColorSpaceConverter ������ɫ��δ��ʼ��");
            SafeBlit(source, destination);
            return;
        }

        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("ColorSpaceConverter: Դ��Ŀ��������Ч");
            SafeBlit(source, destination);
            return;
        }

        try
        {
            // ���ü�����ɫ������
            converterCS.SetTexture(rgbToSRGBKernel, "_InputTexture", source);
            converterCS.SetTexture(rgbToSRGBKernel, "_OutputTexture", destination);
            converterCS.SetVector("_ColorBias", colorBias);

            // ȷ��Ŀ�������������д��
            if (!destination.enableRandomWrite)
            {
                destination.enableRandomWrite = true;
                destination.Create();
            }

            // ���ȼ�����ɫ��
            DispatchComputeShader(converterCS, rgbToSRGBKernel, destination.width, destination.height);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ConvertRGBToSRGB ����: {e.Message}");
            SafeBlit(source, destination);
        }
    }

    /// <summary>
    /// ��sRGB�ռ�ת��ΪRGB���Կռ�
    /// </summary>
    public void ConvertSRGBToRGB(RenderTexture source, RenderTexture destination, Vector4 colorBias)
    {
        if (!isInitialized || converterCS == null)
        {
            Debug.LogError("ColorSpaceConverter ������ɫ��δ��ʼ��");
            SafeBlit(source, destination);
            return;
        }

        if (source == null || !source.IsCreated() || destination == null || !destination.IsCreated())
        {
            Debug.LogError("ColorSpaceConverter: Դ��Ŀ��������Ч");
            SafeBlit(source, destination);
            return;
        }

        try
        {
            // ���ü�����ɫ������
            converterCS.SetTexture(srgbToRGBKernel, "_InputTexture", source);
            converterCS.SetTexture(srgbToRGBKernel, "_OutputTexture", destination);
            converterCS.SetVector("_ColorBias", colorBias);

            // �������д��
            if (!destination.enableRandomWrite)
            {
                destination.enableRandomWrite = true;
                destination.Create();
            }

            // ���ȼ�����ɫ��
            DispatchComputeShader(converterCS, srgbToRGBKernel, source.width, source.height);
        }
        catch (System.Exception e)
        {
            Debug.LogError($"ConvertSRGBToRGB ����: {e.Message}");
            SafeBlit(source, destination);
        }
    }

    /// <summary>
    /// ��ȫ��Blit����������������
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
            Debug.LogError($"Graphics.Blit ʧ��: {e.Message}");
        }
    }

    /// <summary>
    /// �������������ȼ�����ɫ��
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
            Debug.LogError($"���ȼ�����ɫ������: {e.Message}");
        }
    }
}