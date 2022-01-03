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
#include "utils.hpp"

#include <src/cub_wraps.cuh>



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

int main(int argc, char** argv)
{
    //load data
    std::vector<float> col;
    load_csv("../res/Arade_1.csv", {3}, col);
    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size());
    // gen predicate mask
    auto pred = gen_predicate(col, +[](float f){return f > 200;});
    uint8_t* d_mask = vector_to_gpu(pred);
    uint32_t* d_selected_out = alloc_gpu<uint32_t>(1);
    uint64_t* d_failure_count = alloc_gpu<uint64_t>(1);
    // run cub
    cudaEvent_t start, end;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&end));
    launch_cub_flagged_biterator(start, end, d_input, d_output, d_mask, d_selected_out, col.size()); 
    //gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);
    
    // cross check validation
    kernel_check_validation<<<64, 32>>>(d_validation, d_output, out_length, d_failure_count);
    auto vec = gpu_to_vector(d_failure_count, 1);
    std::cout << vec[0] << std::endl;
    return 0;
   
}
