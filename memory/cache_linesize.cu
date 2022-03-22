// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__global__ void linesizeDetectKernel(float* input, float* output, uint32_t* clock){

    uint32_t c[48];
    float val = 0;
    float acc = 0;

    c[0] = get_clock();

    #pragma unroll
    for (int i = 0; i < 32; ++i){
        asm volatile(
            "ld.global.ca.f32    %0,    [%1];  \n\t"
            :"=f"(val):"l"(input):"memory"
        );
        c[i+1] = get_clock();
        acc += val;
        input++;
    }

    #pragma unroll
    for (int i = 0; i < 32; ++i){
        clock[i] = c[i+1] - c[i];
    }
    output[0] = acc;
}


int main(){
    float* input_h; 
    float* input_d;
    float* output_h;
    float* output_d;
    uint32_t* clock_h;
    uint32_t* clock_d;

    int size = 4096;

    input_h     = static_cast<float*>(malloc(sizeof(float) * size));
    output_h    = static_cast<float*>(malloc(sizeof(float) * size));
    clock_h     = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));

    cudaMalloc(&input_d,  sizeof(float) * size);
    cudaMalloc(&output_d, sizeof(float) * size);
    cudaMalloc(&clock_d,  sizeof(uint32_t) * size);

    cudaMemcpy(input_d, input_h, sizeof(float) * size, cudaMemcpyHostToDevice);

    dim3 gDim(1, 1, 1);
    dim3 bDim(1, 1, 1);

    void* kernel_args[3] = {&input_d, &output_d, &clock_d};
    const char* cubin_name = "./sass_cubin/cache_linesize.cubin";
    const char* kernel_name = "linesizeDetect";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    int l1_linesize;
    uint32_t last_cycle;

    last_cycle = clock_h[0];


    // printf("clock %d   latency %u\n", 0, clock_h[0]);
    for (int i = 1; i < 48; ++i){
        // printf("clock %d   latency %u\n", i, clock_h[i]);
        if (clock_h[i] > last_cycle and clock_h[i] - last_cycle > 10) {
            l1_linesize = i * 4;
            break;
        } 
        last_cycle = clock_h[i];
    }
    printf(">>> SASS-Level Detect Latency Result\n");
    // printf("        L2 Linesize \t= %3d B\n", clock_h[1]);
    printf("        L1 LineSize \t= %3u B\n", l1_linesize);


    linesizeDetectKernel<<<gDim, bDim>>>(input_d, output_d, clock_d);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);


    last_cycle = clock_h[0];
    // printf("clock %d   latency %u\n", 0, clock_h[0]);
    for (int i = 1; i < 48; ++i){
        // printf("clock %d   latency %u\n", i, clock_h[i]);
        if (clock_h[i] > last_cycle and clock_h[i] - last_cycle > 10) {
            l1_linesize = i * 4;
            break;
        } 
        last_cycle = clock_h[i];
    }
    printf("\n");
    printf(">>> CUDA C-Level Detect Latency Result\n");
    // printf("        L2 Linesize \t= %3d B\n", clock_h[1]);
    printf("        L1 LineSize \t= %3u B\n", l1_linesize);
    return 0;
}