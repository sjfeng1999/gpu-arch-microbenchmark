template <int N>
__global__ void reg_detect(float *A, float *B){
    int tid = threadIdx.x;
    float vA;
    float dummy = 0;
    ptrdiff_t offset = 0;
    #pragma unroll(N)
    for (int i = 0; i < 64; ++i){
        offset = tid + i * 32;
        dummy += A[offset];
    }
    B[tid] = dummy;
}

template <int N>
__global__ void ureg_detect(float *A, float *B){
    int tid = threadIdx.x;
    float vA;
    float dummy = 0;
    ptrdiff_t offset = 0;
    #pragma unroll(N)
    for (int i = 0; i < 64; ++i){
        offset = i * 32;
        dummy += A[offset];
    }
    B[tid] = dummy;
}
