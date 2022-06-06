//
//
//
//

#pragma once 

#include "cuda.h"
#include "cuda_runtime.h"

cudaError_t launchSassKernel(const char* cubin_name, const char* kernel_name, const dim3& gDim, const dim3& bDim, const int shared_bytes, void** args){
    CUmodule module;
    CUfunction kernel;

    cuModuleLoad(&module, cubin_name);
    cuModuleGetFunction(&kernel, module, kernel_name);

    cuLaunchKernel(kernel, 
                   gDim.x, gDim.y, gDim.z,
                   bDim.x, bDim.y, bDim.z,
                   shared_bytes, // SharedMem Bytes
                   0, // Stream
                   args, 0);

    return cudaPeekAtLastError();
}
