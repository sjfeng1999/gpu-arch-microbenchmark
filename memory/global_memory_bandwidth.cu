// 
//
//
//

#include "cuda.h"
#include "utils.cuh"

constexpr int kGridDimX      = 64;
constexpr int kBlockDimX     = 256;
constexpr int kWarpCount     = kBlockDimX / kWarpSize;
constexpr int kLoopSize      = 4 * 1024;
constexpr size_t kGlobalSize = 256 * 1024 * 1024;
constexpr float kCopySize    = (float)kLoopSize * kGridDimX * kBlockDimX * sizeof(float);

template<int GroupSize, int StrideSize>
__global__
void copyGroup32bKernel(float* input, float* output) {
    const int kWarpWorkload = 32 + kWarpSize / GroupSize * StrideSize;
    const int kBlockWorkload = kWarpCount * kWarpWorkload;
    const int kLine = kGridDimX * kBlockWorkload;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int groupid = laneid / GroupSize;
    int offset = ctaid * kBlockWorkload + warpid * kWarpWorkload + laneid + groupid * StrideSize;

    float* thread_input = input + offset;
    float* thread_output = output + offset;

    for (int i = 0; i < kLoopSize; ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


template<int GroupSize, int StrideSize>
__global__
void copyGroup64bKernel(float2* input, float2* output) {
    const int kWarpWorkload = 32 + kWarpSize / GroupSize * StrideSize;
    const int kBlockWorkload = kWarpCount * kWarpWorkload;
    const int kLine = kGridDimX * kBlockWorkload;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int groupid = laneid / GroupSize;
    int offset = ctaid * kBlockWorkload + warpid * kWarpWorkload + laneid + groupid * StrideSize;

    float2* thread_input = input + offset;
    float2* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 2); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}


template<int GroupSize, int StrideSize>
__global__
void copyGroup128bKernel(float4* input, float4* output) {
    const int kWarpWorkload = 32 + kWarpSize / GroupSize * StrideSize;
    const int kBlockWorkload = kWarpCount * kWarpWorkload;
    const int kLine = kGridDimX * kBlockWorkload;

    int ctaid = blockIdx.x;
    int tid = threadIdx.x;
    int warpid = tid / 32;
    int laneid = tid % 32;
    int groupid = laneid / GroupSize;
    int offset = ctaid * kBlockWorkload + warpid * kWarpWorkload + laneid + groupid * StrideSize;

    float4* thread_input = input + offset;
    float4* thread_output = output + offset;

    for (int i = 0; i < (kLoopSize / 4); ++i) {
        *thread_output = *thread_input;
        thread_input += kLine;
        thread_output += kLine;
    }
}

template<typename Func>
float getElapsed(Func fn, cudaEvent_t start, cudaEvent_t stop) {
    float elapsed = 0;
    cudaEventRecord(start);
    fn();
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed, start, stop);
    return kCopySize / elapsed / 1024 / 1024;
}

int main() {
    float* input_d;
    float* output_d;

    cudaMalloc(&input_d,  sizeof(float) * kGlobalSize);
    cudaMalloc(&output_d,  sizeof(float) * kGlobalSize);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 gDim(kGridDimX);
    dim3 bDim(kBlockDimX);

    printf(" Different access pattern on Global Memory\n");

    auto fn1 = [=]() { copyGroup32bKernel<1, 0><<<gDim, bDim>>>(input_d, output_d);};
    printf("    LDG.32                   \t%.2f GB/s\n", getElapsed(fn1, start, stop)); 

    auto fn2 = [=]() { copyGroup32bKernel<1, 1><<<gDim, bDim>>>(input_d, output_d);};
    printf("    LDG.32 g1s1              \t%.2f GB/s\n", getElapsed(fn2, start, stop)); 

    auto fn9 = [=]() { copyGroup32bKernel<2, 2><<<gDim, bDim>>>(input_d, output_d);};
    printf("    LDG.32 g2s2              \t%.2f GB/s\n", getElapsed(fn9, start, stop)); 

    auto fn10 = [=]() { copyGroup32bKernel<4, 4><<<gDim, bDim>>>(input_d, output_d);};
    printf("    LDG.32 g4s4              \t%.2f GB/s\n", getElapsed(fn10, start, stop)); 

    auto fn8 = [=]() { copyGroup32bKernel<8, 8><<<gDim, bDim>>>(input_d, output_d);};
    printf("    LDG.32 g8s8              \t%.2f GB/s\n", getElapsed(fn8, start, stop)); 

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    auto fn3 = [=]() { copyGroup64bKernel<1, 0><<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);};
    printf("    LDG.64                   \t%.2f GB/s\n", getElapsed(fn3, start, stop)); 

    auto fn4 = [=]() { copyGroup64bKernel<1, 1><<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);};
    printf("    LDG.64 g1s1              \t%.2f GB/s\n", getElapsed(fn4, start, stop)); 

    auto fn11 = [=]() { copyGroup64bKernel<2, 2><<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);};
    printf("    LDG.64 g2s2              \t%.2f GB/s\n", getElapsed(fn11, start, stop)); 

    auto fn12 = [=]() { copyGroup64bKernel<4, 4><<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);};
    printf("    LDG.64 g4s4              \t%.2f GB/s\n", getElapsed(fn12, start, stop)); 

    auto fn13 = [=]() { copyGroup64bKernel<8, 8><<<gDim, bDim>>>((float2*)input_d, (float2*)output_d);};
    printf("    LDG.64 g8s8              \t%.2f GB/s\n", getElapsed(fn13, start, stop)); 

    ///////////////////////////////////////////////////////////////////////////////////////////////////////

    auto fn5 = [=]() { copyGroup128bKernel<1, 0><<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);};
    printf("    LDG.128                  \t%.2f GB/s\n", getElapsed(fn5, start, stop)); 

    auto fn6 = [=]() { copyGroup128bKernel<1, 1><<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);};
    printf("    LDG.128 g1s1             \t%.2f GB/s\n", getElapsed(fn6, start, stop)); 

    auto fn7 = [=]() { copyGroup128bKernel<2, 2><<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);};
    printf("    LDG.128 g2s2              \t%.2f GB/s\n", getElapsed(fn7, start, stop)); 

    auto fn14 = [=]() { copyGroup128bKernel<4, 4><<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);};
    printf("    LDG.128 g4s4              \t%.2f GB/s\n", getElapsed(fn14, start, stop)); 

    auto fn15 = [=]() { copyGroup128bKernel<8, 8><<<gDim, bDim>>>((float4*)input_d, (float4*)output_d);};
    printf("    LDG.128 g8s8              \t%.2f GB/s\n", getElapsed(fn15, start, stop)); 


    cudaFree(input_d);
    cudaFree(output_d);
}