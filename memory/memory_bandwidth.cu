// 
//
//
//

#include "cuda.h"
#include "utils.cuh"

const float kMemoryFrequency_MHz = 5000.0f;          // 5000MHz

float calculateBandWidth(uint elapsed_cycle, const int data_bytes) {
    float second_x_1024_x_1024 = static_cast<float>(elapsed_cycle) / kMemoryFrequency_MHz;
    float data_KBytes = static_cast<float>(data_bytes) / 1024;
    return data_KBytes / second_x_1024_x_1024;
}

template<typename T>
uint getAvgElapsedCycle(int thread_group_size, int stride_size, T* data) {
    T acc = 0;
    for (int i = 0; i < thread_group_size; ++i) {
        acc += data[i * stride_size];
    }
    return static_cast<uint>(acc) / thread_group_size;
}


int main(){
    float* input_d;
    uint32_t* clock_h;
    uint32_t* clock_d;

    int global_size = 4 * 1024 * 1024;
    int shared_size = 32 * 1024;

    clock_h = static_cast<uint32_t*>(malloc(sizeof(uint32_t) * global_size));

    cudaMalloc(&input_d,  sizeof(float) * global_size);
    cudaMalloc(&clock_d,  sizeof(uint32_t) * global_size);

    void* kernel_args[] = {&input_d, &clock_d};


    dim3 gDim1(1, 1, 1);
    dim3 bDim1(1, 1, 1);
    int global_load_bytes = 512 * 1024;
    int shared_load_bytes = 32 * 1024;

    const char* cubin_name1 = "../sass_cubin/memory_bandwidth_thread.cubin";
    const char* kernel_name1 = "memoryBandwidthThread";
    launchSassKernel(cubin_name1, kernel_name1, gDim1, bDim1, shared_size, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * global_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf(">>> SASS Level Memory BandWidth Result\n");
    printf("    Global Memory Load %9d Bytes\n", global_load_bytes);
    printf("    Shared Memory Load %9d Bytes\n", shared_load_bytes);
    printf("        Within Thread Result\n");
    printf("            LDG.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 0), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 0), global_load_bytes));
    printf("            LDG.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 2), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 2), global_load_bytes));
    printf("            LDG.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 4), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 4), global_load_bytes));
    printf("            LDS.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 1), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 1), shared_load_bytes));
    printf("            LDS.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 3), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 3), shared_load_bytes));
    printf("            LDS.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(1, 6, clock_h + 5), calculateBandWidth(getAvgElapsedCycle(1, 6, clock_h + 5), shared_load_bytes));
    printf("\n");


    dim3 gDim2(1, 1, 1);
    dim3 bDim2(256, 1, 1);
    const char* cubin_name2 = "../sass_cubin/memory_bandwidth_block.cubin";
    const char* kernel_name2 = "memoryBandwidthBlock";
    launchSassKernel(cubin_name2, kernel_name2, gDim2, bDim2, shared_size, kernel_args);
    cudaMemcpy(clock_h, clock_d, sizeof(uint) * global_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("        Thread Average Result Within Block\n");
    printf("            LDG.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 0), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 0), global_load_bytes));
    printf("            LDG.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 2), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 2), global_load_bytes));
    printf("            LDG.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 4), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 4), global_load_bytes));
    printf("            LDS.128  Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 1), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 1), shared_load_bytes));
    printf("            LDS.64   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 3), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 3), shared_load_bytes));
    printf("            LDS.32   Elaped Cycle \t=%8u cycle   Bandwidth =\t %5.2f GB/s\n", 
        getAvgElapsedCycle(256, 6, clock_h + 5), calculateBandWidth(getAvgElapsedCycle(256, 6, clock_h + 5), shared_load_bytes));
    printf("\n");
    return 0;
}