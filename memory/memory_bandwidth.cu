// 
//
//
//

#include "cuda.h"
#include "utils.cuh"

const double kMemoryFrequency_MHz = 5000.0f;          // 5000MHz

double calculateBandWidth(uint elapsed_cycle, const int data_bytes) {
    double second_x_1024_x_1024 = static_cast<double>(elapsed_cycle) / kMemoryFrequency_MHz;
    double data_KBytes = static_cast<double>(data_bytes) / 1024;
    return data_KBytes / second_x_1024_x_1024;
}

int main(){
    float* input_d;
    uint32_t* clock_h;
    uint32_t* clock_d;

    int global_size = 4 * 1024 * 1024;
    int shared_size = 256 * 128;

    clock_h = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * global_size));

    cudaMalloc(&input_d,  sizeof(float) * global_size);
    cudaMalloc(&clock_d,  sizeof(uint32_t) * global_size);

    dim3 gDim(1, 1, 1);
    dim3 bDim(1, 1, 1);


    void* kernel_args[] = {&input_d, &clock_d};
    const char* cubin_name = "../sass_cubin/memory_bandwidth_single.cubin";
    const char* kernel_name = "memoryBandwidthSingle";

    launchSassKernel(cubin_name, kernel_name, gDim, bDim, shared_size, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * global_size, cudaMemcpyDeviceToHost);

    int global_load_bytes = 1024 * 16 * 4;
    int shared_load_bytes = 256 * 16 * 4;
    
    printf(">>> SASS Level Single Thread Memory BandWidth Result\n");

    printf("        Global Memory Load %9d Bytes\n", global_load_bytes);
    printf("            LDG.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[0], calculateBandWidth(clock_h[0], global_load_bytes));
    printf("            LDG.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[2], calculateBandWidth(clock_h[2], global_load_bytes));
    printf("            LDG.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[4], calculateBandWidth(clock_h[4], global_load_bytes));

    printf("        Shared Memory Load %9d Bytes\n", shared_load_bytes);
    printf("            LDS.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[1], calculateBandWidth(clock_h[1], shared_load_bytes));
    printf("            LDS.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[3], calculateBandWidth(clock_h[3], shared_load_bytes));
    printf("            LDS.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %6.2f GB/s\n", 
        clock_h[5], calculateBandWidth(clock_h[5], shared_load_bytes));

    return 0;
}