#include <bitset>
#include <cstdlib>
#include <cub/cub.cuh>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

#include "cuda_time.cuh"
#include "cuda_try.cuh"
#include "data_generator.cuh"

#include "fast_prng.cuh"
#include "kernels/data_generator.cuh"

#include "csv_loader.hpp"


template <typename T>
void cpu_buffer_print(T* h_buffer, uint32_t offset, uint32_t length)
{
    for (int i = offset; i < offset+length; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
}

template <typename T>
void gpu_buffer_print(T* d_buffer, uint32_t offset, uint32_t length)
{
    T* h_buffer = static_cast<T*>(malloc(length*sizeof(T)));
    CUDA_TRY(cudaMemcpy(h_buffer, d_buffer+offset, length*sizeof(T), cudaMemcpyDeviceToHost));
    for (int i = 0; i < length; i++) {
        std::bitset<sizeof(T)*8> bits(h_buffer[i]);
        std::cout << bits << " - " << unsigned(h_buffer[i]) << "\n";
    }
    free(h_buffer);
}

int gen_dummy_data(){
    int cuda_dev_id = 0;
    CUDA_TRY(cudaSetDevice(cuda_dev_id));

    uint64_t cnt_b = 1<<30; // mask size in bytes
    uint64_t cnt_e = cnt_b*8; // element count

    uint8_t* d_mask;
    CUDA_TRY(cudaMalloc(&d_mask, sizeof(uint8_t)*cnt_b));
    printf("GPU generation...");
    kernel_generate_mask_uniform<<<64, 32>>>(d_mask, cnt_b, 0.5);
    CUDA_TRY(cudaDeviceSynchronize());
    printf("done\n");

    uint8_t* d_mask2;
    CUDA_TRY(cudaMalloc(&d_mask2, sizeof(uint8_t)*cnt_b));
    CUDA_TRY(cudaMemcpy(d_mask2, d_mask, sizeof(uint8_t)*cnt_b, cudaMemcpyDeviceToDevice));

    uint64_t* d_failure_count;
    CUDA_TRY(cudaMalloc(&d_failure_count, sizeof(uint64_t)));
    CUDA_TRY(cudaMemset(d_failure_count, 0x00, sizeof(uint64_t)));

    // introduce some artificial errors
    fast_prng rng(17);
    for (int i = 0; i < 20; i++) {
        uint64_t rand_idx = rng.rand() % cnt_b;
        CUDA_TRY(cudaMemset(d_mask2+rand_idx, 0x00, sizeof(uint8_t)));
    }

    printf("GPU check validation...");
    kernel_check_validation<<<64, 32>>>(d_mask, d_mask2, cnt_b, d_failure_count);
    CUDA_TRY(cudaDeviceSynchronize());
    printf("done\n");

    uint64_t h_failure_count = 0;
    CUDA_TRY(cudaMemcpy(&h_failure_count, d_failure_count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    printf("validation failures = %lu\n", h_failure_count);

    CUDA_TRY(cudaFree(d_failure_count));
    CUDA_TRY(cudaFree(d_mask2));
    CUDA_TRY(cudaFree(d_mask));

    printf("done\n");
    return 0;
}

template<typename T>
std::vector<uint8_t> gen_predicate(const std::vector<T>& col, bool (*predicate)(T value)){
    std::vector<uint8_t> predicate_bitmask{};
    predicate_bitmask.reserve(col.size() / 8);
    auto it = col.begin();
    for (int i = 0; i < col.size() / 8; i++) {
        uint8_t acc = 0;
        for (int j = 7; j >= 0; j--) {
            if (it == col.end()) break;
            if (predicate(*it++)) {
                acc |= (1<<j);
            }
        }
        predicate_bitmask.push_back(acc);
    }
    return predicate_bitmask;
}

int main()
{
    std::vector<float> col;
    load_csv("../res/Arade_1.csv", {3}, col);
    auto pred = gen_predicate(col, +[](float f){return f > 200;});
    cpu_buffer_print(&pred[0], 0, pred.size());
 

    return 0;
   
}
