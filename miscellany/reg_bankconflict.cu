// 
//
//
//

#include "cuda.h"
#include "utils.cuh"


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


    const char* cubin_name1 = "../sass_cubin/reg_with_bankconflict.cubin";
    const char* kernel_name1 = "regWithBankconflict";
    launchSassKernel(cubin_name1, kernel_name1, gDim, bDim, 0, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf(">>> SASS Level Reg With    BankConflict IPC Result\n");
    printf("        FFMA per \t%.3f cycle\n", static_cast<float>(clock_h[0]) / 128);
    



    const char* cubin_name2 = "../sass_cubin/reg_without_bankconflict.cubin";
    const char* kernel_name2 = "regWithoutBankconflict";
    launchSassKernel(cubin_name2, kernel_name2, gDim, bDim, 0, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("\n");
    printf(">>> SASS Level Reg Without BankConflict IPC Result\n");
    printf("        FFMA per \t%.3f cycle\n", static_cast<float>(clock_h[0]) / 128);

    return 0;
}