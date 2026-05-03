#include "../csrc/quant_fp4.cu"

int main() {
    int x[128 * 128];
    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
            x[get_swizzled_idx<128>(i, j)] = j;
        }
    }
    for (int i = 0; i < 32; i++) {
        for (int j = 0; j < 32; j++) {
            printf("%2d ", x[i * 4 * 128 + j]);
        }
        printf("\n");
    }
}