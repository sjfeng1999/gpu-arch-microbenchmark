// 
//
//
//

#include "cuda.h"
#include "utils.cuh"

constexpr int kGridDimX   = 64;
constexpr int kBlockDimX  = 128;
constexpr int kLoopSize   = 32 * 1024;
constexpr size_t kGlobalSize = 512 * 1024 * 1024;
constexpr size_t kSharedSize = 1024 * 1024;
constexpr float kCopySize = (float)kLoopSize * kGridDimX * kBlockDimX * sizeof(float);

__global__
void copyContinuous32bKernel(float* input, float* output) {
    const int kLine = kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX + tid;
    float* thread_input = input + offset;
    float* thread_output = output + offset;

    for (int i = 0; i < kLoopSize; ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


__global__
void copyStride32bKernel(float* input, float* output) {
    const int kLine = 2 * kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX * 2 + tid * 2;
    float* thread_input = input + offset;
    float* thread_output = output + offset;

    for (int i = 0; i < kLoopSize; ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


__global__
void copyContinuous64bKernel(float2* input, float2* output) {
    const int kLine = kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX + tid;
    float2* thread_input = input + offset;
    float2* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 2); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


__global__
void copyStride64bKernel(float2* input, float2* output) {
    const int kLine = 2 * kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX * 2 + tid * 2;
    float2* thread_input = input + offset;
    float2* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 2); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}

__global__
void copyContinuous128bKernel(float4* input, float4* output) {
    const int kLine = kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX + tid;
    float4* thread_input = input + offset;
    float4* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 4); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


__global__
void copyStride128bKernel(float4* input, float4* output) {
    const int kLine = 2 * kGridDimX * kBlockDimX;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int offset = ctaid * kBlockDimX * 2 + tid * 2;
    float4* thread_input = input + offset;
    float4* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 4); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


int main() {
    float* input_d;
    float* output_d;

    cudaMalloc(&input_d,  sizeof(float) * kGlobalSize);
    cudaMalloc(&output_d,  sizeof(float) * kGlobalSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float elapsed = 0;
    float bandwidth = 0;

    dim3 gDim(kGridDimX);
    dim3 bDim(kBlockDimX);

    cudaEventRecord(start);
    copyContinuous32bKernel<<<gDim, bDim>>>(input_d, output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.32                   \t%.2f GB/s\n", bandwidth); 


    cudaEventRecord(start);
    copyStride32bKernel<<<gDim, bDim>>>(input_d, output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.32 Stride 32b        \t%.2f GB/s\n", bandwidth); 


    cudaEventRecord(start);
    copyContinuous64bKernel<<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.64                   \t%.2f GB/s\n", bandwidth); 

    cudaEventRecord(start);
    copyStride64bKernel<<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.64 Stride 64b        \t%.2f GB/s\n", bandwidth); 

    cudaEventRecord(start);
    copyContinuous128bKernel<<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.128                   \t%.2f GB/s\n", bandwidth); 

    
    cudaEventRecord(start);
    copyContinuous128bKernel<<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    bandwidth = kCopySize / elapsed / 1024 / 1024;
    printf("LDG.128  Stride 128b       \t%.2f GB/s\n", bandwidth); 

    cudaFree(input_d);
    cudaFree(output_d);
}