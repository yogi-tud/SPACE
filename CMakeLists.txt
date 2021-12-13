# convenient cuda clang support was added in 3.18
cmake_minimum_required(VERSION 3.18)

set(CMAKE_CUDA_COMPILER "clang++")
set(CMAKE_CXX_COMPILER "clang++")

project(gpu_compressstore2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_EXTENSIONS OFF)

# force c++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(INCLUDES
    src
)

set(SOURCES
    src/main.cu

    src/cuda_time.cuh
    src/cuda_try.cuh
    src/data_generator.cuh
    src/fast_prng.cuh

    src/kernels/data_generator.cuh
    src/kernels/kernel_pattern.cuh
)

add_executable(gpu_compressstore2 ${SOURCES})

option(AVXPOWER "use avx512-ish" OFF)
if(AVXPOWER)
    target_compile_definitions(gpu_compressstore2 PUBLIC AVXPOWER)
    set(CMAKE_CUDA_FLAGS "-Xcompiler -march=native")
endif()

set_property(TARGET gpu_compressstore2 PROPERTY CUDA_ARCHITECTURES 75 61)

target_include_directories(gpu_compressstore2 PRIVATE ${INCLUDES})
