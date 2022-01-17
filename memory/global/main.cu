
#include <iostream>
#include <cuda.h>


void format_array(float *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%.2f, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


void format_array(uint32_t *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%3u, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


void format_array(int32_t *array, int size, int split_line){
    printf("\n");
    for (int i = 0; i < size; ++i){
        printf("%3d, ", array[i]);
        if (i % split_line == split_line - 1){
            printf("\n");
        }
    }
    printf("\n");
}


int main() {
    size_t width = 1024;
    size_t bytes = 4 * width;
    const char* cubin = "ld_global_cache.cubin";

    int32_t *h_A, *d_A;
    int32_t *h_cost, *d_cost;

    h_A = static_cast<int32_t*>(malloc(bytes));
    h_cost = static_cast<int32_t*>(malloc(bytes));
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_cost, bytes);

    for (int i = 0; i < width; ++i) {
        h_A[i] = i;
    }

    float       totalElapsed;
    cudaEvent_t start_t, stop_t;

    CUmodule module;
    CUfunction kernel;

    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);

    cuModuleLoad(&module, cubin);
    cuModuleGetFunction(&kernel, module, "kern");

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(start_t, 0);

    void* args[2] = {&d_A, &d_cost};
    cuLaunchKernel(kernel, 1, 1, 1,
                   1, 1, 1,
                   0, 0, args, 0);

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    printf(cudaGetErrorString(cudaGetLastError()));
    cudaEventElapsedTime(&totalElapsed, start_t, stop_t);
    cudaMemcpy(h_cost, d_cost, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost);

    printf("ld_global:");
    printf("Array A: \n");
    format_array(h_A, 32, 10);
    printf("Array Cost: \n");
    format_array(h_cost, 32, 10);
    printf("Host Time Elapsed %f ms\n", totalElapsed);
    printf("Throughput  %.3fGFlops", static_cast<float>(width) / totalElapsed / 1024 / 1024);
    return 0;
}
