# StructAdapt

**StructAdapt** is the companion open-source project to the paper
[*Static Conversion and Deployment of AdaConv for Real-Time 3D Style Transfer in Unity*](https://doi.org/10.2991/978-94-6463-823-3_100).

This repository provides the engineering implementation of the method described in the paper: exporting the **AdaConv** style-transfer algorithm to the **ONNX** format and deploying it inside the **Unity Engine**.
The project addresses the structural incompatibility between AdaConv’s dynamic kernel generation and static-graph inference frameworks by introducing a **grouped-convolution vectorization** strategy.
Through this static conversion approach, three model variants are exported from the same AdaConv checkpoint:

* **Fully Dynamic Model** – preserves both spatial and batch dimension dynamics.
* **Batch-Dynamic Model** – fixes spatial dimensions while keeping batch dynamics.
* **Static Model** – uses a completely static computation graph.

All variants achieve real-time rendering of about **14.5 fps on an RTX 4060** and reach **SSIM ≈ 0.67–0.68**, demonstrating mathematical equivalence to the original dynamic formulation.
This repository therefore serves as a practical blueprint for deploying structurally dynamic neural networks in **real-time 3D style-transfer applications** that require static inference graphs.

> 📄 For detailed methodology and experimental results, please refer to the [published paper](https://doi.org/10.2991/978-94-6463-823-3_100).

![adaconv_static_core](./image/adaconv_static_core.pdf)
---
