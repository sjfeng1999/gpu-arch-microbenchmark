cmake_minimum_required(VERSION 3.14)

project(gpu-microbenchmark
        LANGUAGES CXX CUDA)

enable_language(CUDA)
set(MICROBENCHMARK_SRC memory/op_ld_global_vec.cu)

message(STATUS ">>> GPU Microbenchmark")

foreach(benchmark ${MICROBENCHMARK_SRC})
    get_filename_component(benchmark_exec ${benchmark} NAME_WE)
    message(STATUS "Benchmark: ${benchmark_exec}")
    add_executable(${benchmark_exec} ${benchmark})
    target_include_directories(${benchmark_exec} PUBLIC ${PROJECT_SOURCE_DIR}/utils)
endforeach()

message(STATUS "<<<")
