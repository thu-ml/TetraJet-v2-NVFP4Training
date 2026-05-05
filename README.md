<div align="center">

<h1>
TetraJet-v2: Accurate NVFP4 Training for LLMs<br>
with Oscillation Suppression and Outlier Control
</h1>

**Yuxiang Chen**, **Yifan Liu**, **Xiaoming Xu**, **Pengle Zhang**, **Michael Beyer**, **Martin Rapp**, **Jun Zhu**, **Jianfei Chen**

Tsinghua University \& Bosch AI Research

[![arXiv](https://img.shields.io/badge/arXiv-2510.27527-b31b1b.svg)](https://arxiv.org/abs/2510.27527)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICML%202026-8c1b13.svg)](https://openreview.net/forum?id=7ZQhm5HnOA)

</div>

---

This repository contains the official implementation of **TetraJet-v2**, a method for accurate NVFP4 training for large language models with oscillation suppression and outlier control.

## Related Information

- **Paper (TetraJet-v2)**: [arXiv](https://arxiv.org/abs/2510.27527), [OpenReview](https://openreview.net/forum?id=7ZQhm5HnOA)
- **Status**:
    - 🎉 **(2026/05)** This work has been accepted as **a Spotlight paper in ICML 2026**.
    - **(2026/05)** We released an updated version of **TetraJet-v2** with kernels and the training recipe.
    - **(2025/10)** We released the first version of **TetraJet-v2** on arXiv.

- **Previous work (TetraJet-MXFP4Training, ICML 2025)**: [arXiv](https://arxiv.org/abs/2502.20853), [code](https://github.com/thu-ml/TetraJet-MXFP4Training)
  
  This work (TetraJet-v2) extends our prior low-bit training efforts (TetraJet) from MXFP4 for ViTs to more accurate and robust **NVFP4 training** for LLMs.


## Core Contributions

1. **Practically optimal NVFP4 linear recipe**: an end-to-end FP4 training recipe with double-block quantization, aligned activations for correct gradient estimation, and the best backward RHT setting for LLM linear layers.
2. ⭐ **OsciReset (key algorithmic contribution)**: a lightweight weight-oscillation suppression algorithm that identifies unstable FP4 weights and resets their master weights to quantization-bin centers. It improves weight-optimization stability during annealing and convergence in large-data, long-horizon low-precision training, and can also transfer to Quantization-Aware Training (QAT) for producing low-precision weights.
3. **OutControl**: an outlier-control recipe that combines backward RHT and mixed FP4+MXFP8 outlier-channel retention for more accurate activation and gradient computation.

TetraJet-v2 improves FP4 pre-training on OLMo2 models up to 370M parameters and reduces the average gap to BF16 by 51.3% over prior FP4 methods, while providing end-to-end speedups over FP8 baselines.

## Quick View of This Repo 

- `olmo2-training/`: OLMo2 training code based on [allenai/OLMo](https://github.com/allenai/OLMo), with files not needed for training removed.
  - Main OLMo changes are in `olmo/config.py`, `olmo/model.py`, `olmo/train.py`, `scripts/train.py`, plus checkpoint/initialization compatibility for quantized layer buffers.
  - NVFP4 linear layers are implemented in `olmo/quantization_real/linear.py`.
  - Mixed NVFP4+MXFP8 outlier-channel training is implemented in `olmo/quantization_real/linear_mix.py` and scheduled by `olmo/quantization_real/calibrate.py`.
  - Oscillation reset algorithm is implemented in `olmo/quantization_real/oscillation_reset.py` and `olmo/quantization_real/oscillation_reset_memeff.py`.
- `kernels/`: TetraJet-v2 NVFP4 kernels.
- `scripts/`: local and SLURM launch scripts for OLMo2 training.

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

The training configs expect OLMo2 preprocessed `.npy` data. You can use the train and eval/perplexity file lists in the official OLMo config [OLMo2-7B-stage1.yaml](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml) to download the preprocessed files directly.

After downloading, replace both the training data prefix and the eval data paths in `olmo2-training/configs/*/*.yaml` with your local data directories.

### Training

Run launch scripts from `scripts/`:

```bash
cd scripts

# Local, 1 node, 8 GPUs.
./local_70m_8gpu.sh TJv2-mix_fp8-osci_reset-mem_eff

# SLURM, 1 node, 8 GPUs.
sbatch slurm_70m_1node.sh TJv2-mix_fp8-osci_reset-mem_eff

# SLURM, 2 nodes, 8 GPUs per node.
sbatch slurm_70m_2nodes.sh TJv2-mix_fp8-osci_reset-mem_eff
```

Use the corresponding `70m`, `150m`, or `370m` script for each model size. Available config names:

```text
bf16
TJv2-base
TJv2-mix_fp8
TJv2-mix_fp8-osci_reset
TJv2-mix_fp8-osci_reset-mem_eff
```

If no config is provided, scripts default to `bf16`. Pass a checkpoint path as the second argument to resume:

```bash
./local_70m_8gpu.sh TJv2-base /path/to/checkpoint
```

Outputs are saved to `olmo2-training/outputs/<model_size>/<config_name>`. W&B runs offline by default; change `WANDB_MODE` in `scripts/common.sh` to sync online.

### Benchmarking Kernels
> See `kernels/README.md`.

## License

This repository is released under the [Apache License 2.0](LICENSE).

The `olmo2-training/` directory contains code adapted from
[allenai/OLMo](https://github.com/allenai/OLMo), which is also licensed under
Apache License 2.0. We retain the upstream license notice and document the
TetraJet-v2 modifications in [NOTICE](NOTICE).

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2025tetrajet,
  title={Tetrajet-v2: Accurate nvfp4 training for large language models with oscillation suppression and outlier control},
  author={Chen, Yuxiang and Liu, Yifan and Xu, Xiaoming and Zhang, Pengle and Beyer, Michael and Rapp, Martin and Zhu, Jun and Chen, Jianfei},
  journal={arXiv preprint arXiv:2510.27527},
  year={2025}
}
```
