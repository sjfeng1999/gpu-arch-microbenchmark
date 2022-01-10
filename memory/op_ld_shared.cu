__global__ void ld_shared_without_conflict(float *A, float *B, uint32_t *cost){
    int tid = threadIdx.x;
    uint32_t laneid = get_laneid();
    uint32_t start, stop;
    extern __shared__ shArray[];

    float dummy = 0;
    float vA, vB, vC, vD;
    ptrdiff_t offset = laneid;
    float *ptr = shArray + offset;

    start = get_clock();
    asm volatile(
        "ld.shared.ca.f32   %0,     [%4];       \n\t"
        "ld.shared.ca.f32   %1,     [%4+128];   \n\t"
        "ld.shared.ca.f32   %2,     [%4+256];   \n\t"
        "ld.shared.ca.f32   %3,     [%4+384];   \n\t"
        :"=f"(vA),"=f"(vB),"=f"(vC),"=f"(vD)
        :"l"(ptr):"memory"
    );
    bar_sync();
    stop = get_clock();

    cost[tid] = stop - start;
    dummy += vA;
    dummy += vB;
    dummy += vC;
    dummy += vD;
    B[tid] = dummy;
}


__global__ void ld_shared_vec_with_conflict(float *A, float *B, uint32_t *cost){
    int tid = threadIdx.x;
    uint32_t laneid = get_laneid();
    uint32_t start, stop;
    extern __shared__ shArray[];

    float dummy = 0;
    float vA, vB, vC, vD;
    ptrdiff_t offset = 4 * laneid;
    float *ptr = shArray + offset;

    start = get_clock();
    asm volatile(
        "ld.shared.ca.v4.f32   {%0, %1, %2, %3},     [%4]; \n\t"
        :"=f"(vA),"=f"(vB),"=f"(vC),"=f"(vD)
        :"l"(ptr):"memory"
    );
    bar_sync();
    stop = get_clock();

    cost[tid] = stop - start;
    dummy += vA;
    dummy += vB;
    dummy += vC;
    dummy += vD;
    B[tid] = dummy;
}
