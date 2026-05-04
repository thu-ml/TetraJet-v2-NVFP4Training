# TetraJet-v2 Kernels

## Exposed Kernel APIs

- FP4 quantization / transform:
  `quant_fp4_dequant_trans_rh_requant_fused`, `quant_fp4_dequant_trans_fused`, `rh_quant_fp4_trans_rh_requant_fused`, `rh_quant_fp4_stochastic_fused`
- FP4 GEMM:
  `fp4_gemm`
- FP8 path used by mixed kernels:
  `quant_fp8_dequant_trans_requant_fused`, `quant_fp8`, `dequant_fp8`, `fp8_gemm`, `fp8_gemm_accum`
- Mixed backward helper:
  `rh_quant_fp4_trans_rh_requant_fp4_fp8_fused`

## Benchmark Entry

- To run the benchmark suite in this directory, please install Transformer Engine in the benchmark environment, especially for the `base_*` FP8 baseline cases.
- Benchmark entrypoint: `kernels/test_pkg.py`
- Linear baseline example: `python test_pkg.py base_fwd --dims 1024 2048 4096 8192 16384 24576`
- End-to-end example: `python test_pkg.py {e2e,e2e_mix,base_e2e} --dims 4096 8192 16384 --bs 4 --sl 1024 -t`
- `-t` enables per-component timing. It records probe-based `Linear` forward / backward time and lets the benchmark report `fwd`, `bwd`, and derived `nonlinear` time.

## Paper Benchmark Version

- **The benchmark numbers reported in the paper use the standard FP8 baseline measured with `CUDA 12.8 + torch 2.8.0+cu128 + Transformer Engine 2.11.0`.** Our NVFP4 speedups should be interpreted against that standard FP8 baseline.
- To avoid mixing baselines, please keep the TE / CUDA stack fixed when reproducing the paper efficiency numbers.

## TE Notes

- Transformer Engine is only needed for benchmark baselines and tests under `kernels/`; it is not required for training.
- If importing TE fails with an undefined `cublasLtGroupedMatrixLayoutInit_internal` symbol, upgrading `nvidia-cublas` may help.
