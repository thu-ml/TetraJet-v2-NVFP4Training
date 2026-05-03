# Environment Installation

## Prerequisites

- NVIDIA Blackwell GPU. The TetraJet-v2 kernels were designed for RTX 5090 / RTX PRO 6000.
- CUDA >= 12.8
- FlashAttention 2. Install it following the FlashAttention repository instructions, or install a matching prebuilt wheel.
- Transformer Engine is not required for training. It is needed for some FP4/mixed kernel benchmark baselines and tests under `kernels/`.

Possible installation issues:

- Use `--no-build-isolation` when installing CUDA extension packages after PyTorch is installed.
- If using a prebuilt FlashAttention wheel, make sure its Python, PyTorch, CUDA, and CXX11 ABI tags match your environment.
- If building Transformer Engine for benchmark baselines, you may need to expose cuDNN/NCCL headers and libraries from the PyTorch NVIDIA wheels. If importing TE fails with an undefined `cublasLtGroupedMatrixLayoutInit_internal` symbol, upgrade `nvidia-cublas` （`pip install --upgrade nvidia-cublas`）.

## Install

```bash
conda create -y -n tjv2-nvfp4 python=3.12 pip
conda activate tjv2-nvfp4

# Limit CUDA/C++ build parallelism if needed:
# export MAX_JOBS=4

# Install OLMo and training dependencies.
cd olmo2-training
pip install -e ".[train]"
cd ..

# Install TetraJet-v2 kernels.
cd kernels
pip install -e . --no-build-isolation
```