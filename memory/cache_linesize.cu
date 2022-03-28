// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__constant__ float cinput[1024];

__global__ void linesizeDetectKernel(float* input, float* output, uint* clock, float* cinput){

    uint c[256];
    float val = 0;

    float acc = 0;
    c[0] = getClock();
    #pragma unroll
    for (int i = 0; i < 256; ++i){
        asm volatile(
            "ld.global.cg.b32    %0,    [%1];  \n\t"
            :"=f"(val):"l"(input):"memory"
        );
        c[i+1] = getClock();
        acc += val;
        input += 2;
    }
    #pragma unroll
    for (int i = 0; i < 256; ++i){
        clock[i] = c[i+1] - c[i];
    }
    output[0] = acc;

    /////////////////////////////////////////////////////////////////////////

    input += 1024;
    clock += 512;
    acc = 0;
    c[0] = getClock();
    #pragma unroll
    for (int i = 0; i < 256; ++i){
        asm volatile(
            "ld.global.ca.f32    %0,    [%1];  \n\t"
            :"=f"(val):"l"(input):"memory"
        );
        c[i+1] = getClock();
        acc += val;
        input++;
    }
    #pragma unroll
    for (int i = 0; i < 256; ++i){
        clock[i] = c[i+1] - c[i];
    }
    output[1] = acc;
}


int detectCacheLinesize(uint* clock, int size, uint gap){
    int linesize = 0;
    uint last_cycle = clock[0];

    int first = 0;
    int second = 0;

    // formatArray(clock, 256, 16);
    for (int i = 1; i < size; ++i){
        if (clock[i] > last_cycle and clock[i] - last_cycle > gap) {
            if (first == 0){
                first = i;
            } else {
                second = i;
                break;
            }
        } 
        last_cycle = clock[i];
    }
    return (second - first) * 4;
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

    void* kernel_args[] = {&input_d, &output_d, &clock_d, &cinput};
    const char* cubin_name = "../sass_cubin/cache_linesize.cubin";
    const char* kernel_name = "cacheLinesize";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, 0, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf(">>> SASS Level Cache Linesize Result\n");
    printf("        Global   L2 LineSize \t= %3u B\n", detectCacheLinesize(clock_h, 512, 40));
    printf("        Global   L1 LineSize \t= %3u B\n", detectCacheLinesize(clock_h + 512,  512, 10));
    printf("        Constant L2 LineSize \t= %3u B\n", detectCacheLinesize(clock_h + 1024, 512, 100));
    printf("        Constant L1 LineSize \t= %3u B\n", detectCacheLinesize(clock_h + 1536, 512, 10));



    linesizeDetectKernel<<<gDim, bDim>>>(input_d, output_d, clock_d, cinput);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("\n");
    printf(">>> CUDA-C Level Cache Linesize Result\n");
    printf("        Global   L2 LineSize \t= %3u B\n", detectCacheLinesize(clock_h, 512, 40));
    printf("        Global   L1 LineSize \t= %3u B\n", detectCacheLinesize(clock_h + 512, 512, 10));
    return 0;
}