// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


__global__ void sharedBankconflictKernel(float* input, float* output, uint32_t* clock){
    
    asm volatile (
        ".reg.f32   val1, val2, val3, val4;                   \n\t"
        ".reg.u32   c_1, c_2;                                 \n\t"
        ".reg.u32   e_1;                                        \n\t"
        ".shared.b32 smem[1024];                                \n\t"

        "mov.u32    c_1,   %%clock;                         \n\t"
        "ld.shared.f32    val1,    [smem + 0x100];          \n\t"
        "mov.u32    c_2,   %%clock;                         \n\t"
        "sub.u32    e_1, c_2, c_1;                          \n\t"
        "st.global.u32    [%2], e_1;                        \n\t"
        "st.global.f32    [%1] , val1;                      \n\t"

        //////////////////////////////////////////////////////////////////

        "mov.u32    c_1,   %%clock;                         \n\t"
        "ld.shared.f32    val1,    [smem];                  \n\t"
        "ld.shared.f32    val2,    [smem + 0x80];           \n\t"
        "ld.shared.f32    val3,    [smem + 0x100];          \n\t"
        "ld.shared.f32    val4,    [smem + 0x180];          \n\t"
        "mov.u32    c_2,   %%clock;                         \n\t"

        "sub.u32    e_1, c_2, c_1;                          \n\t"
        "st.global.u32    [%2 + 0x4], e_1;                  \n\t"
        "st.global.f32    [%1 + 0x10], val1;                \n\t"
        "st.global.f32    [%1 + 0x20], val2;                \n\t"
        "st.global.f32    [%1 + 0x30], val3;                \n\t"
        "st.global.f32    [%1 + 0x40], val4;                \n\t"

        //////////////////////////////////////////////////////////////////

        "mov.u32    c_1,   %%clock;                         \n\t"
        "ld.shared.f32    val1,    [smem];                  \n\t"
        "ld.shared.f32    val2,    [smem + 0x84];           \n\t"
        "ld.shared.f32    val3,    [smem + 0x108];          \n\t"
        "ld.shared.f32    val4,    [smem + 0x18c];          \n\t"
        "mov.u32    c_2,   %%clock;                         \n\t"

        "sub.u32    e_1, c_2, c_1;                          \n\t"
        "st.global.u32    [%2 + 0x8], e_1;                  \n\t"
        "st.global.f32    [%1 + 0x44], val1;                \n\t"
        "st.global.f32    [%1 + 0x14], val2;                \n\t"
        "st.global.f32    [%1 + 0x24], val3;                \n\t"
        "st.global.f32    [%1 + 0x34], val4;                \n\t"

        //////////////////////////////////////////////////////////////////

        "mov.u32    c_1,   %%clock;                         \n\t"
        "ld.shared.v2.f32 {val1, val2},  [smem];            \n\t"
        "ld.shared.v2.f32 {val3, val4},  [smem + 0x8];      \n\t"
        "mov.u32    c_2,   %%clock;                         \n\t"

        "sub.u32    e_1, c_2, c_1;                          \n\t"
        "st.global.u32    [%2 + 0xc], e_1;                  \n\t"
        "st.global.f32    [%1 + 0x48] , val1;               \n\t"
        "st.global.f32    [%1 + 0x18], val2;                \n\t"
        "st.global.f32    [%1 + 0x28], val3;                \n\t"
        "st.global.f32    [%1 + 0x38], val4;                \n\t"

        //////////////////////////////////////////////////////////////////
        ::"l"(input),"l"(output),"l"(clock):"memory"
    );
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

    void* kernel_args[] = {&input_d, &output_d, &clock_d};


    const char* cubin_name = "../sass_cubin/shared_bankconflict.cubin";
    const char* kernel_name = "sharedBankconflict";
    launchSassKernel(cubin_name, kernel_name, gDim, bDim, size * sizeof(float), kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf(">>> SASS Level Shared Load BankConflict Result\n");
    printf("        Single           Load [0x100]                   Elapsed \t%3u cycle\n", clock_h[0]);
    printf("        Vector           Load [0x0, 0x4 , 0x8  , 0xc  ] Elapsed \t%3u cycle\n", clock_h[1]);
    printf("        WithConflict     Load [0x0, 0x84, 0x108, 0x18c] Elapsed \t%3u cycle\n", clock_h[2]);
    printf("        WithoutConflict  Load [0x0, 0x80, 0x100, 0x180] Elapsed \t%3u cycle\n", clock_h[3]);
    

    sharedBankconflictKernel<<<gDim, bDim>>>(input_d, output_d, clock_d);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("\n");
    printf(">>> CUDA-C Level Shared Load BankConflict Result\n");
    printf("        Single           Load [0x100]                   Elapsed \t%3u cycle\n", clock_h[0]);
    printf("        Vector           Load [0x0, 0x4 , 0x8  , 0xc  ] Elapsed \t%3u cycle\n", clock_h[3]);
    printf("        WithConflict     Load [0x0, 0x84, 0x108, 0x18c] Elapsed \t%3u cycle\n", clock_h[1]);
    printf("        WithoutConflict  Load [0x0, 0x80, 0x100, 0x180] Elapsed \t%3u cycle\n", clock_h[2]);

    return 0;
}