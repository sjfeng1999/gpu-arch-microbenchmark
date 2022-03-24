// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__global__ void linesizeDetectKernel(float* input, float* output, uint* clock){

    uint c[48];
    float val = 0;
    float acc = 0;

    c[0] = getClock();

    #pragma unroll
    for (int i = 0; i < 32; ++i){
        asm volatile(
            "ld.global.ca.f32    %0,    [%1];  \n\t"
            :"=f"(val):"l"(input):"memory"
        );
        c[i+1] = getClock();
        acc += val;
        input++;
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i){
        clock[i] = c[i+1] - c[i];
    }
    output[0] = acc;
}


int detectCacheLinesize(uint* clock, int size){
    int l1_linesize = 0;
    uint last_cycle = clock[0];
    for (int i = 1; i < size; ++i){
        // printf("clock %d   latency %u\n", i, clock[i]);
        if (clock[i] > last_cycle and clock[i] - last_cycle > 10) {
            l1_linesize = i * 4;
            break;
        } 
        last_cycle = clock[i];
    }
    return l1_linesize;
}


int main(){
    float* input_h; 
    float* input_d;
    float* output_h;
    float* output_d;
    uint* clock_h;
    uint* clock_d;

    int size = 4096;

    input_h     = static_cast<float*>(malloc(sizeof(float) * size));
    output_h    = static_cast<float*>(malloc(sizeof(float) * size));
    clock_h     = static_cast<uint*>(malloc(sizeof(uint) * size));

    cudaMalloc(&input_d,  sizeof(float) * size);
    cudaMalloc(&output_d, sizeof(float) * size);
    cudaMalloc(&clock_d,  sizeof(uint) * size);

    cudaMemcpy(input_d, input_h, sizeof(float) * size, cudaMemcpyHostToDevice);

    dim3 gDim(1, 1, 1);
    dim3 bDim(1, 1, 1);

    void* kernel_args[3] = {&input_d, &output_d, &clock_d};
    const char* cubin_name = "../sass_cubin/cache_linesize_75.cubin";
    const char* kernel_name = "linesizeDetect";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf(">>> SASS-Level Detect Latency Result\n");
    // printf("        L2 Linesize \t= %3d B\n", clock_h[1]);
    printf("        L1 LineSize \t= %3u B\n", detectCacheLinesize(clock_h, size));



    linesizeDetectKernel<<<gDim, bDim>>>(input_d, output_d, clock_d);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf("\n");
    printf(">>> CUDA C-Level Detect Latency Result\n");
    // printf("        L2 Linesize \t= %3d B\n", clock_h[1]);
    printf("        L1 LineSize \t= %3u B\n", detectCacheLinesize(clock_h, size));
    return 0;
}