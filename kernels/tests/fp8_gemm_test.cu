#include <random>
#include <cstdio>
#include <cstdint>
#include "../csrc/fp8_gemm.cu"
#include "../csrc/quant_fp8.cu"
#include <assert.h>


__global__ void naive_fp8_mm(
  __nv_fp8_e4m3* A, __nv_fp8_e4m3* B, float* C,
  __nv_fp8_e8m0* scale_A, __nv_fp8_e8m0* scale_B, int n, int k
) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = id / n;
  int col = id % n;
  float ret = 0.0f;
  for (int i = 0; i < k; i++) {
    ret += ((float)A[row * k + i] * (float)scale_A[(row * k + i) / 32]) * ((float)B[col * k + i] * (float)scale_B[(col * k + i) / 32]);
  }
  // C[row * n + col] = __half2float(__float2half_rn(ret));
  C[row * n + col] = __bfloat162float(__float2bfloat16_rn(ret));
}
#ifndef _BUILD_PYTHON_EXT
int main(void) {
  int dim = 32768;
  int m = dim;
  int n = dim;
  int k = dim / 8;
  printf("Testing with m%dn%dk%d\n", m, n, k);
  float *host_A = (float*)malloc(sizeof(float) * m * k);
  float *host_B = (float*)malloc(sizeof(float) * n * k);
  float *host_C = (float*)malloc(sizeof(float) * m * n);
  float *host_gC = (float*)malloc(sizeof(float) * m * n);
  uint16_t *host_C_bf16 = (uint16_t*)malloc(sizeof(uint16_t)* m * n);
  uint8_t *host_qA = (uint8_t*)malloc(sizeof(uint8_t) * m * k);
  std::mt19937 gen(100);
  std::normal_distribution<float> dis(0, 1);
  for (int i = 0; i < m * k; ++i)
    host_A[i] = dis(gen);
    // host_A[i] = 1;
  for (int i = 0; i < n * k; ++i)
    host_B[i] = dis(gen);
    // host_B[i] = 1;
  for (int i = 0; i < m * n; ++i) {
    *((nv_bfloat16*)host_C_bf16 + i) = __float2bfloat16(0);
  }

  void *dev_A, *dev_B, *dev_qA, *dev_qB, *dev_SFA, *dev_SFB, *dev_C_bf16, *dev_C;
  void *dev_SFA_tile, *dev_SFB_tile;
  cudaMalloc(&dev_A, sizeof(float) * m * k);
  cudaMalloc(&dev_B, sizeof(float) * n * k);
  cudaMalloc(&dev_qA, sizeof(uint8_t) * m * k);
  cudaMalloc(&dev_qB, sizeof(uint8_t) * n * k);
  cudaMalloc(&dev_SFA, sizeof(uint8_t) * m * k / 32);
  cudaMalloc(&dev_SFB, sizeof(uint8_t) * n * k / 32);
  cudaMalloc(&dev_SFA_tile, sizeof(uint8_t) * m * k / 32);
  cudaMalloc(&dev_SFB_tile, sizeof(uint8_t) * n * k / 32);
  cudaMalloc(&dev_C_bf16, sizeof(uint16_t) * m * n);
  cudaMalloc(&dev_C, sizeof(uint32_t) * m * n);
  cudaMemcpy(dev_A, host_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_C_bf16, host_C_bf16, sizeof(uint16_t) * m * n, cudaMemcpyHostToDevice);
  
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  quant_fp8_launch((float*)dev_A, (uint8_t*)dev_qA, (uint8_t*)dev_SFA, m, k, stream);
  quant_fp8_launch((float*)dev_B, (uint8_t*)dev_qB, (uint8_t*)dev_SFB, n, k, stream);

  quant_fp8_tile_major_launch((float*)dev_A, (uint8_t*)dev_qA, (uint8_t*)dev_SFA_tile, m, k, stream);
  quant_fp8_tile_major_launch((float*)dev_B, (uint8_t*)dev_qB, (uint8_t*)dev_SFB_tile, n, k, stream);

  cudaMemcpy(host_qA, dev_qA, sizeof(uint8_t) * m * k, cudaMemcpyDeviceToHost);

  printf("Smem Size: %ld kb\n", sizeof(fp8::SmemStorage) / 1024);

  fp8::mxfp8_gemm_launch<false>(
    (uint8_t*)dev_qA, (uint8_t*)dev_qB,
    (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
    (uint16_t*)dev_C_bf16,
    m, n, k,
    stream
  );

  fp8::bf16_to_fp32<<<dim3(m * n / 128), dim3(128)>>>((uint16_t*)dev_C_bf16, (float*)dev_C, size_t(m) * n);
  // fp16_to_fp32<<<dim3(m * n / 128), dim3(128)>>>((uint16_t*)dev_C_bf16, (float*)dev_C, size_t(m) * n);

  cudaMemcpy(host_C, dev_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  int warmup = 5;
  int run_iter = 20;
  double ops = double(m) * n * k * 2;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for(int i = 0; i < warmup; i++){
    fp8::mxfp8_gemm_launch<false>(
      (uint8_t*)dev_qA, (uint8_t*)dev_qB,
      (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
      (uint16_t*)dev_C_bf16,
      m, n, k,
      stream
    );
  }
  cudaEventRecord(start);
  for(int i = 0; i < run_iter; i++){
    fp8::mxfp8_gemm_launch<false>(
      (uint8_t*)dev_qA, (uint8_t*)dev_qB,
      (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
      (uint16_t*)dev_C_bf16,
      m, n, k,
      stream
    );
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0.0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double avg_time = milliseconds / 1000.0 / run_iter;

  double tflops = ops * 1e-12 / avg_time;

  printf("run iter: %d\n", run_iter);
  printf("ops: %lf\n", ops);
  printf("mill_second: %f\n", milliseconds);
  printf("avg time(s): %.12lf\n", avg_time);
  printf("avg TFLOPS: %lf\n", tflops);

  float *dev_gC;
  cudaMalloc(&dev_gC, sizeof(float) * m * n);
  // bf16_to_fp32<<<dim3(m * n / 128), dim3(128)>>>((uint16_t*)dev_C_bf16, (float*)dev_gC, size_t(m) * n);
  // cudaMemcpy(host_gC, dev_gC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  // cudaDeviceSynchronize();
  
  naive_fp8_mm<<<m * n / 512, 512>>>
    ((__nv_fp8_e4m3*)dev_qA, (__nv_fp8_e4m3*)dev_qB, dev_gC, (__nv_fp8_e8m0*)dev_SFA, (__nv_fp8_e8m0*)dev_SFB, n, k);
  cudaMemcpy(host_gC, dev_gC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float max_diff = 0.0f;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      max_diff = max(max_diff, fabs(host_C[i * n + j] - host_gC[i * n + j]));
      // printf("%4.4f ", host_C[i * n + j] - host_gC[i * n + j]);
    }
    // printf("\n");
  }
  printf("Max diff: %.4f\n", max_diff);
  // printf("\n");
  // for (int i = 128; i < 128 + 16; ++i) {
  //   for (int j = 0; j < 16; ++j) {
  //     printf("%.3f ", host_gC[i * n + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  // for (int i = 128; i < 128 + 16; ++i) {
  //   for (int j = 0; j < 16; ++j) {
  //     printf("%.3f ", host_C[i * n + j]);
  //   }
  //   printf("\n");
  // }
  float atol = 1, rtol = 1e-2;
  bool all_close = true;
  int cnt = 0;
  for (int i = 0; i < m * n; i++) {
    if (fabs(host_C[i] - host_gC[i]) > atol && fabs(host_C[i] - host_gC[i]) > rtol * max(host_C[i], host_gC[i])) {
      if (all_close) {
        all_close = false;
        printf("First Error at: %d\n", i);
        printf("gemm: %f\n", host_C[i]);
        printf("baseline: %f\n", host_gC[i]);
        printf("diff:\n");
        for (int j = i; j < i + 16; j++) {
          printf("%.3f ", host_C[j] - host_gC[j]);
        }
        printf("\n");
      }
      cnt++;
      // printf("%d: (%d, %d)\n", i, i / n, i % n);
    }
  }
  if (all_close) printf("Passed!\n");
  else printf("Failed! %d different!\n", cnt);
  return 0;
}
#endif // _BUILD_PYTHON_EXT