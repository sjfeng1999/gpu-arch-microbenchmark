// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__global__ void latencyDetectKernel(float* input, float* output, uint32_t* clock){
    // int ctaid = blockIdx.x;
    // input += ctaid;
    // clock += ctaid;

    // __shared__ float shArray[1024];
    
    // float val[16];

    asm volatile (
        ".reg.f32   val1, val2;                         \n\t"
        ".reg.u32   c_1, c_2, c_3;                      \n\t"
        ".reg.u32   e_1, e_2;                           \n\t"

        "mov.u32    c_1, %%clock;                       \n\t"
        "ld.global.ca.f32    val1,    [%0];             \n\t"
        "mov.u32    c_2,   %%clock;                     \n\t"
        "ld.global.ca.f32    val2,    [%0 + 4];         \n\t"

        "mov.u32    c_3,   %%clock;                     \n\t"
        "sub.u32    e_1, c_2, c_1;                      \n\t"
        "st.global.u32    [%2], e_1;                    \n\t"
        "sub.u32    e_2, c_3, c_2;                      \n\t"
        "st.global.u32    [%2 + 4], e_2;                \n\t"

        "st.global.f32    [%1], val1;                   \n\t"
        "st.global.f32    [%1 + 4], val2;               \n\t"
        ::"l"(input),"l"(output),"l"(clock):"memory"
    );

    // ld.constant.ca.f32 //


}


int main(){
    float* input_h; 
    float* input_d;
    float* output_h;
    float* output_d;
    uint32_t* clock_h;
    uint32_t* clock_d;

    int size = 1024;

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
    const char* cubin_name = "./sass_cubin/memory_latency.cubin";
    const char* kernel_name = "latencyDetect";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);


    printf(">>> SASS-Level Detect Latency Result\n");
    printf("        Global Memory Latency \t= %3u cycle\n", clock_h[0]);
    // printf("Shared Memory   Latency =\t %3d cycle\n", clock_h[0]);
    // printf("Constant Memory Latency =\t %3d cycle\n", clock_h[1]);
    // printf("Constant L1 Cache  Latency =\t %3d cycle\n", clock_h[1]);
    // printf("Constant L2 Cache  Latency =\t %3d cycle\n", clock_h[1]);
    printf("        L1 Cache Latency \t= %3u cycle\n", clock_h[1]);
    // printf("L2 Cache Latency =\t %3d cycle\n", clock_h[3]);
    

    latencyDetectKernel<<<gDim, bDim>>>(input_d, output_d, clock_d);
    cudaMemcpy(clock_h, clock_d, sizeof(float) * size, cudaMemcpyDeviceToHost);

    printf("\n");
    printf(">>> CUDA C-Level Detect Latency Result\n");
    printf("        Global Memory Latency \t= %3u cycle\n", clock_h[0]);
    // printf("Shared Memory   Latency =\t %3d cycle\n", clock_h[0]);
    // printf("Constant Memory Latency =\t %3d cycle\n", clock_h[1]);
    // printf("Constant L1 Cache  Latency =\t %3d cycle\n", clock_h[1]);
    // printf("Constant L2 Cache  Latency =\t %3d cycle\n", clock_h[1]);
    printf("        L1 Cache Latency \t= %3u cycle\n", clock_h[1]);
    // printf("L2 Cache Latency =\t %3d cycle\n", clock_h[3]);
    return 0;
}