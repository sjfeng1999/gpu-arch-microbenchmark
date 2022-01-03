#include "cuda.h"

__forceinline__ __device__ uint32_t get_clock(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%clock; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ uint32_t get_clock64(){
    uint64_t clock64;
    asm volatile(
        "mov.u64    %0,     %%clock; \n\t"
        :"=l"(clock64)::"memory"
    );
    return clock64;
}

__forceinline__ __device__ uint32_t get_warpid(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%warpid; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ uint32_t get_laneid(){
    uint32_t clock;
    asm volatile(
        "mov.u32    %0,     %%laneid; \n\t"
        :"=r"(clock)::"memory"
    );
    return clock;
}

__forceinline__ __device__ void bar_sync(){
    asm volatile(
        "bar.sync   0; \n\t"
    );
}
