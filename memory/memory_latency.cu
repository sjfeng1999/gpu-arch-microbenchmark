// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__constant__ float cinput[1024];

__global__ void latencyDetectKernel(float* input, float* output, uint32_t* clock, float* cinput){

    input += 1024 * 1024 * 1024 / sizeof(float) / 2;
    cinput += 512;

    asm volatile (
        ".reg.f32   val1, val2, val3;                   \n\t"
        ".reg.u32   c_1, c_2, c_3, c_4;                 \n\t"
        ".reg.u32   e_1, e_2, e_3;                      \n\t"
        ".reg.u32   e_4, e_5, e_6;                      \n\t"
        ".shared.b8 smem[32];                           \n\t"
        

        "mov.u32    c_1,   %%clock;                     \n\t"
        "ld.global.cg.f32    val1,    [%0];             \n\t"
        "mov.u32    c_2,   %%clock;                     \n\t"
        "ld.global.ca.f32    val2,    [%0 + 0x4];       \n\t"
        "mov.u32    c_3,   %%clock;                     \n\t"
        "ld.global.ca.f32    val3,    [%0 + 0x8];       \n\t"
        "mov.u32    c_4,   %%clock;                     \n\t"

        "sub.u32    e_1, c_2, c_1;                      \n\t"
        "sub.u32    e_2, c_3, c_2;                      \n\t"
        "sub.u32    e_3, c_4, c_3;                      \n\t"
        
        "add.f32    val1, val1, val2;                   \n\t"
        "add.f32    val1, val1, val3;                   \n\t"

        "st.global.u32    [%2],       e_1;              \n\t"
        "st.global.u32    [%2 + 0x4], e_2;              \n\t"
        "st.global.u32    [%2 + 0x8], e_3;              \n\t"

        "st.global.f32    [%1],       val1;             \n\t"
        "st.global.f32    [%1 + 0x4], val2;             \n\t"
        "st.global.f32    [%1 + 0x8], val3;             \n\t"

        ///////////////////////////////////////////////////////////////////

        "bar.sync   0;                                  \n\t"

        ///////////////////////////////////////////////////////////////////
        
        "mov.u32    c_1,   %%clock;                     \n\t"
        "ld.const.cg.f32    val1,    [%3];              \n\t"
        "mov.u32    c_2,   %%clock;                     \n\t"
        "ld.const.ca.f32    val2,    [%3 + 0x4];        \n\t"
        "mov.u32    c_3,   %%clock;                     \n\t"
        "ld.const.ca.f32    val3,    [%3 + 0x8];        \n\t"
        "mov.u32    c_4,   %%clock;                     \n\t"

        "sub.u32    e_4, c_2, c_1;                      \n\t"
        "sub.u32    e_5, c_3, c_2;                      \n\t"
        "sub.u32    e_6, c_4, c_3;                      \n\t"
        
        "add.f32    val1, val1, val2;                   \n\t"
        "add.f32    val1, val1, val3;                   \n\t"

        "st.global.u32    [%2 + 0xc],  e_4;             \n\t"
        "st.global.u32    [%2 + 0x10], e_5;             \n\t"
        "st.global.u32    [%2 + 0x14], e_6;             \n\t"

        "st.global.f32    [%1 + 0xc],  val1;            \n\t"
        "st.global.f32    [%1 + 0x10], val2;            \n\t"
        "st.global.f32    [%1 + 0x14], val3;            \n\t"

        /////////////////////////////////////////////////////////////////////////

        "bar.sync   0;                                  \n\t"

        ///////////////////////////////////////////////////////////////////

        "mov.u32    c_1,   %%clock;                     \n\t"
        "ld.shared.f32    val1,    [smem];              \n\t"
        "mov.u32    c_2,   %%clock;                     \n\t"

        "sub.u32    e_4, c_2, c_1;                      \n\t"
        "st.global.u32    [%2 + 0x18], e_4;             \n\t"
        "st.global.f32    [%1 + 0x18], val1;            \n\t"

        ::"l"(input),"l"(output),"l"(clock),"l"(cinput):"memory"
    );

}


int main(){
    float* input_d;
    float* output_d;
    uint32_t* clock_h;
    uint32_t* clock_d;

    int size = 1024;
    int large_size = 1500 * 1024 * 1024;

    clock_h = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * size));

    cudaMalloc(&input_d,  large_size);
    cudaMalloc(&output_d, sizeof(float) * size);
    cudaMalloc(&clock_d,  sizeof(uint32_t) * size);

    for (int i = 0; i < 128; ++i){
        cinput[i] = i;
    }

    dim3 gDim(1, 1, 1);
    dim3 bDim(1, 1, 1);

    void* kernel_args[] = {&input_d, &output_d, &clock_d, &cinput};
    const char* cubin_name = "../sass_cubin/memory_latency.cubin";
    const char* kernel_name = "memoryLatency";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, size, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf(">>> SASS Level Memory Latency Result\n");
    printf("        Global    Memory    Latency \t= %4u cycle\n", clock_h[0]);
    printf("        Global    TLB       Latency \t= %4u cycle\n", clock_h[7]);
    printf("        Global    L2-Cache  Latency \t= %4u cycle\n", clock_h[1]);
    printf("        Global    L1-Cache  Latency \t= %4u cycle\n", clock_h[2]);
    printf("        Shared    Memory    Latency \t= %4u cycle\n", clock_h[6]);
    printf("        Constant  Memory    Latency \t= %4u cycle\n", clock_h[3]);
    printf("        Constant  L2-Cache  Latency \t= %4u cycle\n", clock_h[4]);
    printf("        Constant  L1-Cache  Latency \t= %4u cycle\n", clock_h[5]);
    

    
    latencyDetectKernel<<<gDim, bDim>>>(input_d, output_d, clock_d, cinput);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("\n");
    printf(">>> CUDA-C Level Memory Latency Result\n");
    printf("        Global    Memory    Latency \t= %4u cycle\n", clock_h[0]);
    printf("        Global    L2-Cache  Latency \t= %4u cycle\n", clock_h[1]);
    printf("        Global    L1-Cache  Latency \t= %4u cycle\n", clock_h[2]);
    printf("        Shared    Memory    Latency \t= %4u cycle\n", clock_h[6]);
    printf("        Constant  Memory    Latency \t= %4u cycle\n", clock_h[3]);
    printf("        Constant  L2-Cache  Latency \t= %4u cycle\n", clock_h[4]);
    printf("        Constant  L1-Cache  Latency \t= %4u cycle\n", clock_h[5]);
    return 0;
}