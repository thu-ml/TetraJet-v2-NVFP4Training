# OLMo2 Training Code

This directory contains the OLMo2 training stack used by TetraJet-v2. It is based on [allenai/OLMo](https://github.com/allenai/OLMo), with files not needed for this training release removed. For the original upstream documentation, see `README_OLMo.md`.


## Main Entry Points

- `scripts/train.py`: command-line training entry point. It loads the YAML config, sets runtime paths, and starts OLMo training.
- `olmo/train.py`: main training loop. TetraJet-v2 hooks for outlier-channel recording and OsciReset are called here.
- `olmo/model.py`: model definition. Linear layers are replaced by TetraJet-v2 NVFP4 / mixed-precision linear modules according to the config.
- `olmo/config.py`: config dataclasses, including `quant_nvfp4`, outlier-channel selection, and oscillation-reset options.
- `olmo/checkpoint.py` and `olmo/initialization.py`: checkpoint and initialization compatibility for quantized-layer buffers.

## TetraJet-v2 Quantization Code

- `olmo/quantization_real/linear.py`: core NVFP4 linear layer with real quantization.
- `olmo/quantization_real/linear_mix.py`: mixed NVFP4 + MXFP8 outlier-channel linear layer.
- `olmo/quantization_real/calibrate.py`: schedules outlier-channel recording and periodic OsciReset during training.
- `olmo/quantization_real/oscillation_reset.py`: full-state OsciReset implementations.
- `olmo/quantization_real/oscillation_reset_memeff.py`: memory-efficient OsciReset implementation that tracks only sampled high-risk weights.
- `olmo/quantization_real/utils.py` and `utils_type.py`: shared helpers for identifying quantized linear modules and handling distributed parameter access.

## Configs

Training configs are under `configs/{70m_50bt,150m_100bt,370m_200bt}/`.

Each model size includes:

- `bf16.yaml`
- `TJv2-base.yaml`
- `TJv2-mix_fp8.yaml`
- `TJv2-mix_fp8-osci_reset.yaml`
- `TJv2-mix_fp8-osci_reset-mem_eff.yaml`

Within each model size, these configs share the same model, data, optimizer, LR schedule, and batch settings. They differ only in `quant_nvfp4`.
