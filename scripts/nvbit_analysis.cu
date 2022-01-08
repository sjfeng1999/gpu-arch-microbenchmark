
#include <cuda.h>
#include <stdio.h>
#include <nvbit.h>

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char* event_name, void* params,
                         CUresult* pStatus){
    printf(event_name);
}
