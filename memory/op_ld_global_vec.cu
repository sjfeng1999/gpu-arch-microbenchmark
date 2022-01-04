//
// C++
// Created by sjfeng.
//
//
#include <iostream>
#include <cuda.h>

__forceinline__ __device__ uint32_t get_clock(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ void bar_sync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}

template <typename T>
void format_array(T *array, size_t size);


template <>
void format_array<float>(float *array, size_t size){
    for (size_t i = 0; i < size; ++i){
        printf("%.3f, ", array[i]);
        if (i % 10 == 9){
            printf("\n");
        }
        if (i % 100 == 99){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}

template <>
void format_array<uint32_t>(uint32_t *array, size_t size){
    for (size_t i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % 10 == 9){
            printf("\n");
        }
        if (i % 100 == 99){
            printf("=====================\n");
        }
    }
    printf("\n\t");
}


__global__ void ld_global(float *A, uint32_t *cost){
    int tid = threadIdx.x;
    uint32_t start, stop;
    uint32_t clockElapsed;

    float dummy = 0;
    float vA[8], vB[8], vC[8], vD[8];
    float *ptr;
    ptrdiff_t offset = 0;

    bar_sync();
    start = get_clock();
    #pragma unroll
    for (int i = 0; i < 32; ++i){
        offset = i * 4;
        ptr = A + offset;

        asm volatile(
            "ld.global.ca.f32   %0,     [%4];       \n\t"
            "ld.global.ca.f32   %1,     [%4+4];     \n\t"
            "ld.global.ca.f32   %2,     [%4+8];     \n\t"
            "ld.global.ca.f32   %3,     [%4+12];    \n\t"
            :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0])
            :"l"(ptr):"memory"
        );
        dummy += vA[0];
        dummy += vB[0];
        dummy += vC[0];
        dummy += vD[0];
    }
    bar_sync();
    stop = get_clock();
    cost[tid] = stop - start;
    A[tid] += dummy;
}

__global__ void ld_global_v4(float *A, uint32_t *cost){
    int tid = threadIdx.x;
    uint32_t start, stop;
    uint32_t clockElapsed;

    float dummy = 0;
    float vA[8], vB[8], vC[8], vD[8];
    float *ptr;
    ptrdiff_t offset = 0;

    bar_sync();
    start = get_clock();
    #pragma unroll
    for (int i = 0; i < 32; ++i){
        offset = i * 4;
        ptr = A + offset;

        asm volatile(
            "ld.global.ca.v4.f32   {%0, %1, %2, %3},     [%4]; \n\t"
            :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0]):"l"(ptr):"memory"
        );
        dummy += vA[0];
        dummy += vB[0];
        dummy += vC[0];
        dummy += vD[0];
    }
    bar_sync();
    stop = get_clock();
    cost[tid] = stop - start;
    A[tid] += dummy;
}


__global__ void ld_global_mass(float *A, uint32_t *cost, size_t size){
    int tid = threadIdx.x;

    float dummy = 0;
    float vA[8], vB[8], vC[8], vD[8];
    float *ptr;
    ptrdiff_t offset = 0;

    #pragma unroll(8)
    for (int i = 0; i < size; ++i){
        offset = i * 4;
        ptr = A + offset;

        asm volatile(
            "ld.global.ca.f32   %0,     [%8]; \n\t"
            "ld.global.ca.f32   %1,     [%8+4]; \n\t"
            "ld.global.ca.f32   %2,     [%8+8]; \n\t"
            "ld.global.ca.f32   %3,     [%8+12]; \n\t"
            "ld.global.ca.f32   %4,     [%8+16]; \n\t"
            "ld.global.ca.f32   %5,     [%8+20]; \n\t"
            "ld.global.ca.f32   %6,     [%8+24]; \n\t"
            "ld.global.ca.f32   %7,     [%8+28]; \n\t"
            :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0]),
             "=f"(vA[1]),"=f"(vB[1]),"=f"(vC[1]),"=f"(vD[1])
            :"l"(ptr):"memory"
        );
        dummy += vA[0];
        dummy += vB[0];
        dummy += vC[0];
        dummy += vD[0];
        dummy += vA[1];
        dummy += vB[1];
        dummy += vC[1];
        dummy += vD[1];
    }
    A[tid] += dummy;
}

__global__ void ld_global_v4_mass(float *A, uint32_t *cost, size_t size){
    int tid = threadIdx.x;

    float dummy = 0;
    float vA[8], vB[8], vC[8], vD[8];
    float *ptr;
    ptrdiff_t offset = 0;

    #pragma unroll(8)
    for (int i = 0; i < size; ++i){
        offset = i * 8;
        ptr = A + offset;

        asm volatile(
            "ld.global.ca.v4.f32   {%0, %1, %2, %3},     [%8];      \n\t"
            "ld.global.ca.v4.f32   {%4, %5, %6, %7},     [%8+16];   \n\t"
            :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0]),
             "=f"(vA[1]),"=f"(vB[1]),"=f"(vC[1]),"=f"(vD[1])
            :"l"(ptr):"memory"
        );
        dummy += vA[0];
        dummy += vB[0];
        dummy += vC[0];
        dummy += vD[0];
        dummy += vA[1];
        dummy += vB[1];
        dummy += vC[1];
        dummy += vD[1];
    }
    A[tid] += dummy;
}

int main() {
    size_t width = 8192 * 4;
    size_t bytes = 4 * width;

    dim3 bDim(32);
    dim3 gDim(1);

    float *h_A, *d_A;
    uint32_t *h_cost, *d_cost;

    h_A = static_cast<float*>(malloc(bytes));
    h_cost = static_cast<uint32_t*>(malloc(bytes));
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_cost, bytes);

    for (int i = 0; i < width; ++i) {
        h_A[i] = i;
    }

    float       totalElapsed;
    cudaEvent_t start_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);

    printf("\n========================================================================================================");
    cudaEventRecord(start_t, 0);

    ld_global<<<gDim, bDim>>>(d_A, d_cost);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    cudaMemcpy(h_cost, d_cost, bytes, cudaMemcpyDeviceToHost);

    printf("\nld_global:");
    printf("\nArray Cost: \n");
    format_array(h_cost, 32);
    printf("\nHost Time Elapsed %f ms", totalElapsed);
    printf("\nThroughput  %.3fGFlops", static_cast<float>(width) / totalElapsed / 1024 / 1024);
    printf("\n========================================================================================================");
    cudaEventRecord(start_t, 0);

    ld_global_v4<<<gDim, bDim>>>(d_A, d_cost);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    cudaMemcpy(h_cost, d_cost, bytes, cudaMemcpyDeviceToHost);

    printf("\nld_global_v4:");
    printf("\nArray Cost: \n");
    format_array(h_cost, 32);
    printf("\nHost Time Elapsed %f ms", totalElapsed);
    printf("\nThroughput  %.3fGFlops", static_cast<float>(width) / totalElapsed / 1024 / 1024);
    printf("\n========================================================================================================");
    cudaEventRecord(start_t, 0);

    ld_global_mass<<<gDim, bDim>>>(d_A, d_cost, width / 4);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nld_global_mass:");
    printf("\nHost Time Elapsed %f ms \t Throughput  %.3fGFlops", totalElapsed, static_cast<float>(width) / totalElapsed / 1024 / 1024);
    printf("\n========================================================================================================");
    cudaEventRecord(start_t, 0);

    ld_global_v4_mass<<<gDim, bDim>>>(d_A, d_cost, width / 4);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nld_global_v4_mass:");
    printf("\nHost Time Elapsed %f ms \t Throughput  %.3fGFlops", totalElapsed, static_cast<float>(width) / totalElapsed / 1024 / 1024);
    printf("\n========================================================================================================");
    return 0;
}
