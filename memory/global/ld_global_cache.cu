//
// C++
// Created by sjfeng.
//
//
#include <iostream>
#include <cuda.h>


void format_array(float *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%.2f, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


void format_array(uint32_t *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


void format_array(int32_t *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%3d, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


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

__forceinline__ __device__ int32_t ld_global_ca_s32(int32_t* ptr){
    int32_t ret;
    asm volatile(
        "ld.global.ca.s32   %0,     [%1]; \n\t"
        :"=r"(ret):"l"(ptr):"memory"
    );
    return ret;
}

__global__ void ld_global_ca(int32_t *A, int32_t *cost){
    uint32_t s0, s1, s2;
    int32_t vA, vB, vC;
    s0 = get_clock();
    vA = ld_global_ca_s32(A);
    bar_sync();
    s1 = get_clock();
    vB = ld_global_ca_s32(A + 1);
    vC = ld_global_ca_s32(A + 2);
    bar_sync();
    s2 = get_clock();
    vA = vA + vB + vC;
    A[0] = vA;
    cost[0] = s1 - s0;
    cost[1] = s2 - s1;
}


int main() {
    int width = 512;
    int bytes = 4 * width;

    dim3 bDim(32);
    dim3 gDim(1);

    int32_t *h_A, *d_A;
    int32_t *h_cost, *d_cost;

    h_A = static_cast<int32_t*>(malloc(bytes));
    h_cost = static_cast<int32_t*>(malloc(bytes));
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
    cudaEventRecord(start_t, 0);

    ld_global_ca<<<gDim, bDim>>>(d_A, d_cost);

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);

    printf(cudaGetErrorString(cudaGetLastError()));
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    cudaMemcpy(h_cost, d_cost, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);

    printf("Array A: \n");
    format_array(h_A, 32, 10);
    printf("Array Cost: \n");
    format_array(h_cost, 32, 10);
    printf("d_A : %p  h_A : %p\n", d_A, h_A);
    printf("Host Time Elapsed %f ms", totalElapsed);
    return 0;
}
