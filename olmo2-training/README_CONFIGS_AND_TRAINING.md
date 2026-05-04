# OLMo2 Training Configs

This directory contains OLMo2 configs for TetraJet-v2 NVFP4 realquant training. Launch scripts are in `../scripts`.

## Run Training

From the `TetraJet-v2-NVFP4Training` repository root, enter the launch script directory first:

```bash
cd scripts
```

Local single-node 8-GPU training:

```bash
./local_70m_8gpu.sh TJv2-mix_fp8-osci_reset-mem_eff
./local_150m_8gpu.sh TJv2-mix_fp8-osci_reset-mem_eff
./local_370m_8gpu.sh TJv2-mix_fp8-osci_reset-mem_eff
```

SLURM single-node 8-GPU training:

```bash
sbatch slurm_70m_1node.sh TJv2-mix_fp8-osci_reset-mem_eff
sbatch slurm_150m_1node.sh TJv2-mix_fp8-osci_reset-mem_eff
sbatch slurm_370m_1node.sh TJv2-mix_fp8-osci_reset-mem_eff
```

SLURM two-node training, with 8 GPUs per node:

```bash
sbatch slurm_70m_2nodes.sh TJv2-mix_fp8-osci_reset-mem_eff
sbatch slurm_150m_2nodes.sh TJv2-mix_fp8-osci_reset-mem_eff
sbatch slurm_370m_2nodes.sh TJv2-mix_fp8-osci_reset-mem_eff
```

The second argument can be a checkpoint path:

```bash
./local_70m_8gpu.sh TJv2-base /path/to/checkpoint
```

If no config name is provided, the scripts default to `bf16.yaml`.

The scripts automatically enter `../olmo2-training` and run `scripts/train.py`. `save_folder` is not stored in the YAML configs; the launch scripts set it to:

```bash
outputs/<model_size>/<config_name>
```

W&B is set to offline mode by default in `../scripts/common.sh`; change `WANDB_MODE` there if online sync is needed.

## Update Paths

Training data paths are stored in the YAML configs. The default data prefix is:

```bash
/gpfs-flash/shared-data/public_datasets/olmo2-processed/preprocessed
```

On a different machine, replace this prefix with your local OLMo2 preprocessed data directory. The default eval data path is:

```bash
/share/home/chenyuxiang22/hdd/llm_dataset/olmo-data-eval/perplexity/v3_small_dolma2-tokenizer
```

## Configs

Each model size keeps 5 configs:

- `bf16.yaml`: BF16 baseline without NVFP4.
- `TJv2-base.yaml`: TetraJet-v2 realquant FP4 training.
- `TJv2-mix_fp8.yaml`: FP4 with a 10% X-channel MXFP8 mixed path.
- `TJv2-mix_fp8-osci_reset.yaml`: Mixed path with oscillation reset.
- `TJv2-mix_fp8-osci_reset-mem_eff.yaml`: Memory-efficient oscillation reset, with a default 10% sampling ratio.
