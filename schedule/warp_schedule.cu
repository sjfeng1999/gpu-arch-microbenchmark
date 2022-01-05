// warp exec position
template <int m, int n, int k=-1>
__global__ void warp_workload(float *A, float *B){
    int tid = threadIdx.x;
    int warpid = get_warpid();
    if (warpid == m or warpid == n or warpid == k){
        float dummy = 0;
        float vA[4], vB[4], vC[4], vD[4];
        float *ptr;
        ptrdiff_t offset = 0;

        #pragma unroll
        for (int i = 0; i < 32; ++i){
            offset = i * 4;
            ptr = A + offset;

            asm volatile(
                "ld.global.ca.f32   %0,     [%4];       \n\t"
                "ld.global.ca.f32   %1,     [%4+4];     \n\t"
                "ld.global.ca.f32   %2,     [%4+8];     \n\t"
                "ld.global.ca.f32   %3,     [%4+12];    \n\t"
                :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0])
                :"l"(ptr):"memory"
            );
            dummy += vA[0];
            dummy += vB[0];
            dummy += vC[0];
            dummy += vD[0];
        }
        B[tid] = dummy;
    }
}

__global__ void block_workload(float *A, float *B){
    int tid = threadIdx.x;

    float dummy = 0;
    float vA[4], vB[4], vC[4], vD[4];
    float *ptr;
    ptrdiff_t offset = 0;

    #pragma unroll
    for (int i = 0; i < 32; ++i){
        offset = i * 4;
        ptr = A + offset;

        asm volatile(
            "ld.global.ca.f32   %0,     [%4];       \n\t"
            "ld.global.ca.f32   %1,     [%4+4];     \n\t"
            "ld.global.ca.f32   %2,     [%4+8];     \n\t"
            "ld.global.ca.f32   %3,     [%4+12];    \n\t"
            :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0])
            :"l"(ptr):"memory"
        );
        dummy += vA[0];
        dummy += vB[0];
        dummy += vC[0];
        dummy += vD[0];
    }
    B[tid] = dummy;
}
