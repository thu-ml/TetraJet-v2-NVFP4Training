// test for quant & dequant in tile major

#include "../csrc/quant_fp4.cu"
#include <random>
#include <assert.h>

int main() {
    int m = 4096;
    int k = 4096;
    float *host_in = (float*)malloc(sizeof(float) * m * k);
    float *host_A = (float*)malloc(sizeof(float) * m * k);
    float *host_A_tile = (float*)malloc(sizeof(float) * m * k);
    float *host_A_fused_dq = (float*)malloc(sizeof(float) * m * k);
    uint8_t *host_qA = (uint8_t*)malloc(sizeof(uint8_t) * m * k / 2);
    uint8_t *host_qA_tile = (uint8_t*)malloc(sizeof(uint8_t) * m * k / 2);
    uint8_t *host_qA_fused = (uint8_t*)malloc(sizeof(uint8_t) * m * k / 2);

    std::mt19937 gen(100);
    std::normal_distribution<float> dis(0, 1);
    for (int i = 0; i < m * k; ++i)
        // host_in[i] = dis(gen);
        host_in[i] = i % 128;

    void *dev_A, *dev_qA, *dev_SFA, *dev_OuterSFA;
    void *dev_A_tile, *dev_qA_tile, *dev_SFA_tile, *dev_OuterSFA_tile;
    void *dev_A_fused_dq, *dev_qA_fused;
    void *dev_qA_t, *dev_SFA_t, *dev_OuterSFA_t;
    cudaMalloc(&dev_A_fused_dq, sizeof(float) * m * k);
    cudaMalloc(&dev_qA_fused, sizeof(uint8_t) * m * k / 2);
    cudaMalloc(&dev_A, sizeof(float) * m * k);
    cudaMalloc(&dev_qA, sizeof(uint8_t) * m * k / 2);
    cudaMalloc(&dev_A_tile, sizeof(float) * m * k);
    cudaMalloc(&dev_qA_tile, sizeof(uint8_t) * m * k / 2);
    cudaMalloc(&dev_SFA, sizeof(uint8_t) * m * k / 16);
    cudaMalloc(&dev_OuterSFA, sizeof(float) * m * k / 128);
    cudaMalloc(&dev_SFA_tile, sizeof(uint8_t) * m * k / 16);
    cudaMalloc(&dev_OuterSFA_tile, sizeof(float) * m * k / 128);
    
    cudaMalloc(&dev_qA_t, sizeof(uint8_t) * m * k / 2);
    cudaMalloc(&dev_SFA_t, sizeof(uint8_t) * m * k / 16);
    cudaMalloc(&dev_OuterSFA_t, sizeof(float) * m * k / 128);

    cudaMemcpy(dev_A, host_in, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_A_tile, host_in, sizeof(float) * m * k, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nvfp4_double_quant_launch<false>
        ((float*)dev_A, (uint8_t*)dev_qA, (float*)dev_OuterSFA, (uint8_t*)dev_SFA, m * k, 0, stream);
    nvfp4_double_quant_tile_major_launch<false>
        ((float*)dev_A_tile, (uint8_t*)dev_qA_tile, (float*)dev_OuterSFA_tile, (uint8_t*)dev_SFA_tile, m, k, 0, stream);

    cudaMemcpy(host_qA, dev_qA, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_qA_tile, dev_qA_tile, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);

    // quant
    for (int i = 0; i < m * k / 2; i++) {
        assert(host_qA[i] == host_qA_tile[i]);
    }
        
    nvfp4_dequant_launch((uint8_t*)dev_qA, (float*)dev_A, (float*)dev_OuterSFA, (uint8_t*)dev_SFA, m * k, stream);
    nvfp4_dequant_tile_major_launch
        ((uint8_t*)dev_qA_tile, (float*)dev_A_tile, (float*)dev_OuterSFA_tile, (uint8_t*)dev_SFA_tile, m, k, stream);
    
    cudaMemcpy(host_A, dev_A, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_A_tile, dev_A_tile, sizeof(float) * m * k, cudaMemcpyDeviceToHost);

    // dequant
    for (int i = 0; i < m * k; i++) {
        assert(host_A[i] == host_A_tile[i]);
    }

    // quant & dequant & transpose fused
    cudaMemcpy(dev_A, host_in, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    nvfp4_double_quant_tile_major_dequant_trans_fused_launch
        ((float*)dev_A, (uint8_t*)dev_qA_fused, (float*)dev_OuterSFA, (uint8_t*)dev_SFA, (float*)dev_A_fused_dq, m, k, stream);
    cudaMemcpy(host_A_fused_dq, dev_A_fused_dq, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_qA_fused, dev_qA_fused, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < m * k / 2; i++) {
        assert(host_qA[i] == host_qA_fused[i]);
    }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            float val_a = host_A[i * k + j];
            float val_b = host_A_fused_dq[j * m + i];
            float diff = fabs(val_a - val_b);
            float mx = fmax(val_a, val_b);
            
            float relative_diff = (mx > 1e-10f) ? (diff / mx) : diff;

            if (relative_diff >= 1e-5) {
                fprintf(stderr, "Error at i=%d, j=%d (index=%d)\n", i, j, i * k + j);
                fprintf(stderr, "Baseline: %g\n", val_a);
                fprintf(stderr, "Fused:    %g\n", val_b);
                fprintf(stderr, "Rel Diff: %g\n", relative_diff);
                assert(false && "Accuracy check failed!");
            }
        }
    }

    // quant dequant trans random hadamard requant fused
    cudaMemcpy(dev_A, host_in, sizeof(float) * m * k, cudaMemcpyHostToDevice);
    nvfp4_double_quant_tile_major_dequant_trans_rh_requant_fused_launch
        (
            (float*)dev_A, (uint8_t*)dev_qA_fused, (float*)dev_OuterSFA, (uint8_t*)dev_SFA,
            (uint8_t*)dev_qA_t, (float*)dev_OuterSFA_t, (uint8_t*)dev_SFA_t, m, k, 0, stream
        );
    cudaMemcpy(host_qA_fused, dev_qA_fused, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < m * k / 2; i++) {
        assert(host_qA[i] == host_qA_fused[i]);
    }

    cudaMemcpy(host_qA_fused, dev_qA_t, sizeof(uint8_t) * m * k / 2, cudaMemcpyDeviceToHost);

    nvfp4_dequant_tile_major_launch
        ((uint8_t*)dev_qA_fused, (float*)dev_A, (float*)dev_OuterSFA, (uint8_t*)dev_SFA, m, k, stream);
    cudaMemcpy(host_A_fused_dq, dev_A, sizeof(float) * m * k, cudaMemcpyDeviceToHost);
    for (int i = 0; i < m * k; i++) {
        assert(host_A_fused_dq[i] == host_A[i]);
    }
    printf("Passed!\n");
}