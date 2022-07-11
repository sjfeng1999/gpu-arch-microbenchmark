cmake_minimum_required(VERSION 3.14)

project(gpu-arch-microbenchmark
        LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin/)

set(TARGET_ARCH "-gencode arch=compute_80,code=sm_80 \
                 -gencode arch=compute_75,code=sm_75 \
                 -gencode arch=compute_70,code=sm_70")

set(CMAKE_CUDA_FLAGS "${CMAKE_NVCC_FLAGS} ${TARGET_ARCH}")

set(MICROBENCHMARK_SRC memory/memory_latency.cu
                       memory/memory_bandwidth.cu
                       memory/cache_linesize.cu
                       memory/global_memory_bandwidth.cu
                       miscellany/reg_bankconflict.cu
                       miscellany/shared_bankconflict.cu
                       schedule/warp_schedule.cu)


message(STATUS ">>> GPU Microbenchmark")

foreach(benchmark ${MICROBENCHMARK_SRC})
    get_filename_component(benchmark_exec ${benchmark} NAME_WE)
    message(STATUS "Benchmark: ${benchmark_exec}")
    add_executable(${benchmark_exec} ${benchmark})
    target_include_directories(${benchmark_exec} PUBLIC ${PROJECT_SOURCE_DIR}/utils)
    target_link_libraries(${benchmark_exec} cuda)
endforeach()

message(STATUS "<<<")
