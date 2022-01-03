#include <iostream>
#include <cuda.h>

using std::cout;
using std::endl;

__global__ void ld_global(){

}

int main() {
    int m = 256;
    size_t width = m * m;
    size_t bytes = 4 * width;

    dim3 blockDim(128);
    dim3 gridDim(2);

    float *A, *B, *C;
    float *h_A, *h_B, *h_C;

    cudaMallocManaged(&A, bytes);
    cudaMallocManaged(&B, bytes);
    cudaMallocManaged(&C, bytes);

    h_A = malloc(bytes);
    h_B = malloc(bytes);
    h_C = malloc(bytes);
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // for (int i = 0; i < width; ++i) {
    //     A[i] = i;
    // }
    printf("\n");
    float       totalElapsed;
    cudaEvent_t start_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);

    cudaEventRecord(start_t, 0);

    sgemm_128x128_nt_cuda<<<gridDim, blockDim>>>(d_A, d_B, d_C, m, m, m);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);

    printf("\nTime Elapsed %f ms", totalElapsed);
    return 0;
}
