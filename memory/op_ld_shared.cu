// kernel compare with bank-conflict and without


__global__ void ld_shared_without_conflict(float *A, float *B, uint32_t *cost){
    int tid = threadIdx.x;

    extern __shared__ shArray[];

    float dummy = 0;
    float vA, vB, vC, vD;
    float *ptr;
    ptrdiff_t offset = 0;

    asm volatile(
        "ld.shared.ca.f32   %0,     [%4];       \n\t"
        "ld.shared.ca.f32   %1,     [%4+4];     \n\t"
        "ld.shared.ca.f32   %2,     [%4+8];     \n\t"
        "ld.shared.ca.f32   %3,     [%4+12];    \n\t"
        :"=f"(vA[0]),"=f"(vB[0]),"=f"(vC[0]),"=f"(vD[0])
        :"l"(ptr):"memory"
    );
    dummy += vA[0];
    dummy += vB[0];
    dummy += vC[0];
    dummy += vD[0];

    B[tid] = dummy;
}
