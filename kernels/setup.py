import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0+PTX")

ext_modules = [
    CUDAExtension(
        name="tetrajetv2._quant",
        sources=[
            "csrc/binding_quant.cpp",
            "csrc/quant_fp4.cu",
            "csrc/quant_fp8.cu"
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++17",
                "-gencode=arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr",
            ],
        },
    ),
    CUDAExtension(
        name="tetrajetv2._gemm",
        sources=[
            "csrc/binding_gemm.cpp",
            "csrc/fp4_gemm.cu",
            "csrc/fp8_gemm.cu"
        ],
        extra_compile_args={
            "cxx": [
                "-O3",
                "-std=c++17",
            ],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-std=c++17",
                "-gencode=arch=compute_120a,code=sm_120a",
                "--expt-relaxed-constexpr",
            ],
        },
        extra_link_args=["-lcuda"]
    ),
]

setup(
    name="tetrajetv2",
    version="0.0.1",
    description="CUDA extension with pybind11 (RTX 5090 / sm_120)",
    packages=["tetrajetv2"],
    ext_modules=ext_modules,
    package_data={
        "tetrajetv2": ["*.pyi", "py.typed"]
    },
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
    install_requires=['torch']
)
