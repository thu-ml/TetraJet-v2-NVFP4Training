<div align="center">

<h1>
TetraJet-v2: Accurate NVFP4 Training for LLMs<br>
with Oscillation Suppression and Outlier Control
</h1>

**Yuxiang Chen**, **Yifan Liu**, **Xiaoming Xu**, **Pengle Zhang**, **Michael Beyer**, **Martin Rapp**, **Jun Zhu**, **Jianfei Chen**

Tsinghua University \& Bosch AI Research

[![arXiv](https://img.shields.io/badge/arXiv-2510.27527-b31b1b.svg)](https://arxiv.org/abs/2510.27527)

</div>

---

This repository contains the official implementation of **TetraJet-v2**, a method for accurate NVFP4 training for large language models with oscillation suppression and outlier control.

## Related Information

- 📄 **Paper (TetraJet-v2)**: [![arXiv](https://img.shields.io/badge/arXiv-2510.27527-b31b1b.svg)](https://arxiv.org/abs/2510.27527)
- 📢 **Status**:
    - 🎉 **(2026/05)** This work has been accepted as **a Spotlight paper in ICML 2026**.
    - 🚀 **(2026/05)** We released an updated version of **TetraJet-v2** with kernels and the training recipe.
    - 📝 **(2025/11)** We released the first version of **TetraJet-v2** on arXiv.

- 🔗 **Previous work (TetraJet, ICML 2025)**: [![arXiv](https://img.shields.io/badge/arXiv-2502.20853-b31b1b.svg)](https://arxiv.org/abs/2502.20853) [![Code](https://img.shields.io/badge/GitHub-TetraJet--MXFP4Training-181717?logo=github)](https://github.com/thu-ml/TetraJet-MXFP4Training)
  
  🧩 This work (TetraJet-v2) extends our prior low-bit training efforts (TetraJet) from MXFP4 for ViTs to more accurate and robust **NVFP4 training** for LLMs.


## Core Contributions

1. 
2. 
3. 

## Quick View of This Repo 


## Quick Start

### Prerequisites

- NVIDIA Blackwell GPU. The TetraJet-v2 kernels were designed for RTX 5090 / RTX PRO 6000.
- CUDA >= 12.8
- FlashAttention 2.

### Install

```bash
conda create -y -n tjv2-nvfp4 python=3.12 pip
conda activate tjv2-nvfp4

# Install OLMo and training dependencies.
cd olmo2-training
pip install -e ".[train]"
cd ..

# Install TetraJet-v2 kernels.
cd kernels
pip install -e . --no-build-isolation
```

Possible Issues:
- Use `--no-build-isolation` when installing CUDA extension packages after PyTorch is installed.
- If using a prebuilt FlashAttention wheel, make sure its Python, PyTorch, CUDA, and CXX11 ABI tags match your environment.
- Limit CUDA/C++ build parallelism if needed: `export MAX_JOBS=4`.

### Data Preparation

### Training

### Benchmarking Kernels
> See `kernels/README.md`.

## Citation

