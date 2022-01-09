// warp exec position

#include <iostream>
#include <cuda.h>

__forceinline__ __device__ uint32_t get_warpid(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}
template <int M, int N>
__global__ void warp_workload(float *A, float *B){
    int tid = threadIdx.x;
    int warpid = get_warpid();
    if (warpid == M or warpid == N){
        float dummy = 0;
        float vA[4], vB[4], vC[4], vD[4];
        float *ptr;
        ptrdiff_t offset = 0;

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
        B[tid] = dummy;
    }
}


int main() {
    size_t width = 512;
    size_t bytes = 4 * width;

    dim3 bDim(256);
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

    warp_workload<0, 1><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);


    cudaEventRecord(start_t, 0);

    warp_workload<0, 2><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);





    cudaEventRecord(start_t, 0);

    warp_workload<0, 3><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);




    cudaEventRecord(start_t, 0);
    warp_workload<0, 4><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);





    cudaEventRecord(start_t, 0);
    warp_workload<0, 5><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);






    cudaEventRecord(start_t, 0);
    warp_workload<0, 6><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);





    cudaEventRecord(start_t, 0);
    warp_workload<0, 7><<<gDim, bDim>>>(d_A, d_B);
    printf(cudaGetErrorString(cudaGetLastError()));

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost);

    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    printf("\nHost Time Elapsed %f ms", totalElapsed);





    return 0;
}
