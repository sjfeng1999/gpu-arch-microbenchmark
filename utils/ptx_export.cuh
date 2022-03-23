//
//
//
//

#pragma once 

#include "cuda.h"
#include "cuda_runtime.h"
#include "./macro.cuh"

__forceinline__ __device__ uint32_t getClock(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ uint32_t getSmid(){
    uint32_t smid;
    asm volatile(
        "mov.u32    %0,     %%smid; \n\t"
        :"=r"(smid)::"memory"
    );
    return smid;
}

__forceinline__ __device__ uint32_t getWarpid(){
    uint32_t warpid;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(warpid)::"memory"
    );
    return warpid;
}

__forceinline__ __device__ uint32_t getLaneid(){
    uint32_t laneid;
    asm volatile(
        "mov.u32    %0,     %%laneid; \n\t"
        :"=r"(laneid)::"memory"
    );
    return laneid;
}


__forceinline__ __device__ void barSync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}
