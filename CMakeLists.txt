cmake_minimum_required(VERSION 3.14)

project(gpu-microbenchmark
        LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(MICROBENCHMARK_SRC memory/memory_latency.cu)


message(STATUS ">>> GPU Microbenchmark")

foreach(benchmark ${MICROBENCHMARK_SRC})
    get_filename_component(benchmark_exec ${benchmark} NAME_WE)
    message(STATUS "Benchmark: ${benchmark_exec}")
    add_executable(${benchmark_exec} ${benchmark})
    target_include_directories(${benchmark_exec} PUBLIC ${PROJECT_SOURCE_DIR}/utils)
    target_link_libraries(${benchmark_exec} cuda)
endforeach()

message(STATUS "<<<")
