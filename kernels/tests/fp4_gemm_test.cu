#include <random>
#include <cstdint>
#include <cstdio>
#include "../csrc/fp4_gemm.cu"
#include "../csrc/quant_fp4.cu"



__global__ void naive_fp4_mm(
  uint8_t* A, uint8_t* B, float* C,
  float* outer_scale_A, float* outer_scale_B,
  __nv_fp8_e4m3* inner_scale_A, __nv_fp8_e4m3* inner_scale_B,
  int n, int k
) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int row = id / n;
  int col = id % n;
  float ret = 0.0f;
  for (int i = 0; i < k / 2; i++) {
    uint8_t pack_A = A[row * k / 2 + i];
    float val_A[2];
    val_A[0] = fp4_to_float(pack_A & 0x0f);
    val_A[1] = fp4_to_float((pack_A >> 4) & 0x0f);
    uint8_t pack_B = B[col * k / 2 + i];
    float val_B[2];
    val_B[0] = fp4_to_float(pack_B & 0x0f);
    val_B[1] = fp4_to_float((pack_B >> 4) & 0x0f);
    float scale_product_A = (float)inner_scale_A[(row * k + i * 2) / 16] * outer_scale_A[(row * k + i * 2) / 128];
    float scale_product_B = (float)inner_scale_B[(col * k + i * 2) / 16] * outer_scale_B[(col * k + i * 2) / 128];
    #pragma unroll
    for (int j = 0; j < 2; j++) {
      ret += val_A[j] * scale_product_A * val_B[j] * scale_product_B;
    }
  }
  C[row * n + col] = __bfloat162float(__float2bfloat16_rn(ret));
}

int main(void) {
  int dim = 32768;
  int m = dim;
  int n = dim;
  int k = dim;
  printf("Testing FP4 GEMM with m%dn%dk%d\n", m, n, k);
  float *host_A = (float*)malloc(sizeof(float) * m * k);
  float *host_B = (float*)malloc(sizeof(float) * n * k);
  float *host_C = (float*)malloc(sizeof(float) * m * n);
  float *host_gC = (float*)malloc(sizeof(float) * m * n);
  uint8_t *host_qA = (uint8_t*)malloc(sizeof(uint8_t) * m * k / 2);
  std::mt19937 gen(100);
  std::normal_distribution<float> dis(0, 1);
  for (int i = 0; i < m * k; ++i)
    host_A[i] = dis(gen);
    // host_A[i] = 1;
  for (int i = 0; i < n * k; ++i)
    host_B[i] = dis(gen);
    // host_B[i] = 1;


  void *dev_A, *dev_B, *dev_qA, *dev_qB, *dev_SFA, *dev_SFB, *dev_C_bf16, *dev_C, *dev_OuterSFA, *dev_OuterSFB;
  void *dev_SFA_tile, *dev_SFB_tile, *dev_OuterSFA_tile, *dev_OuterSFB_tile;
  cudaMalloc(&dev_A, sizeof(float) * m * k);
  cudaMalloc(&dev_B, sizeof(float) * n * k);
  cudaMalloc(&dev_qA, sizeof(uint8_t) * m * k / 2);
  cudaMalloc(&dev_qB, sizeof(uint8_t) * n * k / 2);
  cudaMalloc(&dev_SFA, sizeof(uint8_t) * m * k / 16);
  cudaMalloc(&dev_SFB, sizeof(uint8_t) * n * k / 16);
  cudaMalloc(&dev_OuterSFA, sizeof(float) * m * k / 128);
  cudaMalloc(&dev_OuterSFB, sizeof(float) * n * k / 128);
  cudaMalloc(&dev_SFA_tile, sizeof(uint8_t) * m * k / 16);
  cudaMalloc(&dev_SFB_tile, sizeof(uint8_t) * n * k / 16);
  cudaMalloc(&dev_OuterSFA_tile, sizeof(float) * m * k / 128);
  cudaMalloc(&dev_OuterSFB_tile, sizeof(float) * n * k / 128);
  cudaMalloc(&dev_C_bf16, sizeof(uint16_t) * m * n);
  cudaMalloc(&dev_C, sizeof(uint32_t) * m * n);
  cudaMemcpy(dev_A, host_A, sizeof(float) * m * k, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_B, host_B, sizeof(float) * n * k, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  
  nvfp4_double_quant_launch<false>
    ((float*)dev_A, (uint8_t*)dev_qA,(float*)dev_OuterSFA, (uint8_t*)dev_SFA, m * k, 0, stream);
  cudaMemcpy(host_qA, dev_qA, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);
  nvfp4_double_quant_launch<false>
    ((float*)dev_B, (uint8_t*)dev_qB,(float*)dev_OuterSFB, (uint8_t*)dev_SFB, n * k, 0, stream);
  nvfp4_double_quant_tile_major_launch<false>
    ((float*)dev_A, (uint8_t*)dev_qA, (float*)dev_OuterSFA_tile, (uint8_t*)dev_SFA_tile, m, k, 0, stream);
  nvfp4_double_quant_tile_major_launch<false>
    ((float*)dev_B, (uint8_t*)dev_qB, (float*)dev_OuterSFB_tile, (uint8_t*)dev_SFB_tile, n, k, 0, stream);

  uint8_t *host_qA_tile = (uint8_t*)malloc(sizeof(uint8_t) * m * k / 2);
  cudaMemcpy(host_qA_tile, dev_qA, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);

  for (int i = 0; i < m * k / 2; i++) assert(host_qA[i] == host_qA_tile[i]);

  // debug only
  // __nv_fp8_e4m3* host_SFA = (__nv_fp8_e4m3*)malloc(sizeof(uint8_t) * m * k / 16);
  // __nv_fp8_e4m3* host_SFA_tile = (__nv_fp8_e4m3*)malloc(sizeof(uint8_t) * m * k / 16);
  // cudaMemcpy(host_SFA, dev_SFA, sizeof(uint8_t) * m * k / 16, cudaMemcpyDeviceToHost);
  // cudaMemcpy(host_SFA_tile, dev_SFA_tile, sizeof(uint8_t) * m * k / 16, cudaMemcpyDeviceToHost);

  // for (int i = 0; i < 128; i++) {
  //   printf("%f\n", (float)host_SFA_tile[i]);
  // }
  // for (int i = 0; i < m; i++) {
  //   for (int j = 0; j < k / 16; j++) {
  //     int offset = i / 128 * 128 * k / 16 + j / 8 * 8 * 128 * 8 + i - i / 128 * 128 + j - j / 8 * 8;
  //     assert((float)host_SFA[i * k / 16 + j] == (float)host_SFA_tile[offset]);
  //   }
  // }
  // for (int i = 0; i < m * k / 16; i++) {
  //   if ((float)host_SFA_tile[i] != 448) {
  //     printf("%d: %f\n", i, (float)host_SFA_tile[i]);
  //   }
  //   assert((float)host_SFA[i] == 448);
  // }
  // for (int i = 0; i < n * k / 16; i++) {
  //   host_SFB[i] = (__nv_fp8_e4m3)1;
  // }
  // float* host_OuterSFA = (float*)malloc(sizeof(float) * m * k / 128);
  // float* host_OuterSFB = (float*)malloc(sizeof(float) * n * k / 128);
  // for (int i = 0; i < m * k / 128; i++) {
  //   host_OuterSFA[i] = 1;
  // }
  // for (int i = 0; i < n * k / 128; i++) {
  //   host_OuterSFB[i] = 1;
  // }
  // cudaMemcpy(dev_SFA, host_SFA, sizeof(uint8_t*) * m * k / 16, cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_SFB, host_SFB, sizeof(uint8_t*) * m * k / 16, cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_OuterSFA, host_OuterSFA, sizeof(float) * m * k / 128, cudaMemcpyHostToDevice);
  // cudaMemcpy(dev_OuterSFB, host_OuterSFB, sizeof(float) * n * k / 128, cudaMemcpyHostToDevice);
  
  // printf("gA:\n");
  // for (int i = 0; i < 2; i++) {
  //   for (int j = 0; j < 64; j++) {
  //     uint8_t val = host_qA[i * 64 + j];
  //     printf("%.1f %.1f ", lut_fp4(val & 0xf), lut_fp4((val >> 4) & 0xf));
  //   }
  //   printf("\n");
  // }
  // printf("\n");

  constexpr int CtaM = 128;
  constexpr int CtaN = 128;
  constexpr int CtaK = 128;
  constexpr int EpiM = 128;
  constexpr int EpiN = 64;

  constexpr int AtomRepK = CtaK / fp4::AtomK;
  using SmemStorage = fp4::SmemStorageT<CtaM, CtaN, CtaK, EpiM, EpiN, AtomRepK>;
  printf("Smem Size: %lu kb\n", sizeof(SmemStorage) / 1024);

  fp4::nvfp4_gemm_launch<CtaM, CtaN, CtaK, EpiM, EpiN>((uint8_t*)dev_qA, (uint8_t*)dev_qB,
                                                  (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
                                                  (float*)dev_OuterSFA_tile, (float*)dev_OuterSFB_tile,
                                                  (uint16_t*)dev_C_bf16,
                                                  m, n, k, stream);

  fp4::bf16_to_fp32<<<dim3(m * n / 128), dim3(128)>>>((uint16_t*)dev_C_bf16, (float*)dev_C, size_t(m) * n);
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
    fp4::nvfp4_gemm_launch<CtaM, CtaN, CtaK, EpiM, EpiN>((uint8_t*)dev_qA, (uint8_t*)dev_qB,
                                                    (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
                                                    (float*)dev_OuterSFA_tile, (float*)dev_OuterSFB_tile,
                                                    (uint16_t*)dev_C_bf16,
                                                    m, n, k, stream);
  }
  cudaDeviceSynchronize();

  cudaEventRecord(start);
  for(int i = 0; i < run_iter; i++){
    fp4::nvfp4_gemm_launch<CtaM, CtaN, CtaK, EpiM, EpiN>((uint8_t*)dev_qA, (uint8_t*)dev_qB,
                                                    (uint8_t*)dev_SFA_tile, (uint8_t*)dev_SFB_tile,
                                                    (float*)dev_OuterSFA_tile, (float*)dev_OuterSFB_tile,
                                                    (uint16_t*)dev_C_bf16,
                                                    m, n, k, stream);
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
  naive_fp4_mm<<<m * n / 512, 512>>>(
    (uint8_t*)dev_qA, (uint8_t*)dev_qB, (float*)dev_gC, 
    (float*)dev_OuterSFA, (float*)dev_OuterSFB, 
    (__nv_fp8_e4m3*)dev_SFA, (__nv_fp8_e4m3*)dev_SFB, 
    n, k
  );
  cudaMemcpy(host_gC, dev_gC, sizeof(float) * m * n, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  float max_diff = 0.0f;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      max_diff = max(max_diff, fabs(host_C[i * n + j] - host_gC[i * n + j]));
      // printf("%.3f ", host_C[i * n + j] - host_gC[i * n + j]);
    }
    // printf("\n");
  }
  printf("Max diff: %.4f\n", max_diff);
  // printf("\n");
  // for (int i = 0; i < 16; ++i) {
  //   for (int j = 0; j < 16; ++j) {
  //     printf("%.3f ", host_gC[i * n + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n");
  // for (int i = 0; i < 16; ++i) {
  //   for (int j = 0; j < 16; ++j) {
  //     printf("%.3f ", host_C[i * n + j]);
  //   }
  //   printf("\n");
  // }
  float atol = 1.0, rtol = 1e-2;
  bool all_close = true;
  int cnt = 0;
  for (int i = 0; i < m * n; i++) {
    if ((fabs(host_C[i] - host_gC[i]) > atol && fabs(host_C[i] - host_gC[i]) > rtol * max(host_C[i], host_gC[i])) || isnanf(host_C[i])) {
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