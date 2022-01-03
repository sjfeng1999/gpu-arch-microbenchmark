#include <iostream>
#include <cuda.h>

using std::cout;
using std::endl;

__forceinline__ __device__ uint32_t get_clock(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ uint32_t get_clock64(){
    uint64_t clock64;
    asm volatile(
        "mov.u64    %0,     %%clock; \n\t"
        :"=l"(clock64)::"memory"
    );
    return clock64;
}

__forceinline__ __device__ uint32_t get_warpid(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ uint32_t get_laneid(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%laneid; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ void bar_sync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}

__global__ void ld_global_ca(float *A, float *B, uint32_t *cost, size_t size){
    int tid = threadIdx.x;
    uint32_t start, stop;

    float sink = 0;
    float *ptr;
    ptrdiff_t offset = 0;
    for (int i = 0; i < size; ++i){
        offset = i;
        ptr = A + offset;
        start = get_clock();
        asm volatile(
            "ld.global.ca.f32   %0,     [%1]; \n\t"
            :"=f"(sink):"l"(ptr):"memory"
        );

        bar_sync();
        stop = get_clock();
        cost[i] = stop - start;
        sink += 1.0;
    }
    __syncthreads();
    B[tid] = sink;
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


int main() {
    size_t width = 512;
    size_t bytes = 4 * width;

    dim3 bDim(32);
    dim3 gDim(1);

    float *h_A, *h_B;
    float *d_A, *d_B;
    uint32_t *h_cost, *d_cost;

    h_A = static_cast<float*>(malloc(bytes));
    h_B = static_cast<float*>(malloc(bytes));
    h_cost = static_cast<uint32_t*>(malloc(bytes));
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
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

    ld_global_ca<<<gDim, bDim>>>(d_A, d_B, d_cost, width);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cost, d_cost, bytes, cudaMemcpyDeviceToHost);

    printf("\nArray A: \n");
    format_array(h_A, width);
    printf("\nArray B: \n");
    format_array(h_B, width);
    printf("\nArray Cost: \n");
    format_array(h_cost, width);
    printf("\nd_A : %p  h_A : %p\n", d_A, h_A);
    printf("\nHost Time Elapsed %f ms", totalElapsed);
    return 0;
}
