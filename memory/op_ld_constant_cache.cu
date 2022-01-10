__global__ void ld_constant_cg(float *A, uint32_t *cost, size_t size){
    int tid = threadIdx.x;
    uint32_t start, stop;

    float value = 0;
    float dummy = 0;

    float *ptr;
    ptrdiff_t offset = 0;

    for (int i = 0; i < size; ++i){
        offset = i;
        ptr = A + offset;
        start = get_clock();
        asm volatile(
            "ld.constant.cg.f32   %0,     [%1]; \n\t"
            :"=f"(value):"l"(ptr):"memory"
        );
        bar_sync();
        stop = get_clock();
        cost[i] += stop - start;
        dummy += value;
    }
    __syncthreads();
    A[tid] = dummy;
}


__global__ void ld_constant_ca(float *A, uint32_t *cost, size_t size){
    int tid = threadIdx.x;
    uint32_t start, stop;

    float value = 0;
    float dummy = 0;

    float *ptr;
    ptrdiff_t offset = 0;

    for (int i = 0; i < size; ++i){
        offset = i;
        ptr = A + offset;
        start = get_clock();
        asm volatile(
            "ld.constant.ca.f32   %0,     [%1]; \n\t"
            :"=f"(value):"l"(ptr):"memory"
        );
        bar_sync();
        stop = get_clock();
        cost[i] += stop - start;
        dummy += value;
    }
    __syncthreads();
    A[tid] = dummy;
}
