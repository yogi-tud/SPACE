#include <bitset>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <stdio.h>
#include <thread>
#include <vector>

#define DISABLE_CUDA_TIME
#include "cuda_time.cuh"
#include "cuda_try.cuh"

#include "csv_loader.hpp"
#include "utils.cuh"
#include "data_generator.cuh"
#include "benchmarks.cuh"

// clang-format off


// skipping: wenn 32*32=1024 elemente nur nullen enthält wird er für den writeout geskipped
// global/shared memory: compute prefix sum in shared memory
// optimized chunks: whole warp does a sequential read in
// two phase / partial prefix sum / cub: 
    // two phase: upsweep -> write; downsweep -> write; writeout reads downsweep result 
    // cub: = two phase 
    // partial: upsweep -> write; writeout reads upsweep result and computes downsweep on the fly
// streams: use multiple cuda streams splitting up the work

// clang-format on

int main(int argc, char** argv)
{
    if (argc > 1) {
        int device = atoi(argv[1]);
        printf("setting device numer to %i\n", device);
        CUDA_TRY(cudaSetDevice(device));
    }

    // load data
    std::vector<float> col;
    load_csv("../res/Arade_1.csv", {3}, col);
    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size());

    // gen predicate mask
    size_t one_count;
    auto pred = gen_predicate(
        col, +[](float f) { return f > 2000; }, &one_count);
    uint8_t* d_mask = vector_to_gpu(pred);

    printf(
        "line count: %zu, one count: %zu, percentage: %f\n", col.size(),
        one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length =
        generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    // run benchmark
    intermediate_data id{col.size(), 1024, 8};
    const int iterations = 100;

    float time_3pass = 0;
    float time_cub = 0;
    for (int i = 0; i < iterations; i++) {

        time_3pass += bench1_base_variant(
            &id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
        validate(&id, d_validation, d_output, out_length);

        time_cub +=
            bench7_cub_flagged(&id, d_input, d_mask, d_output, col.size());
        validate(&id, d_validation, d_output, out_length);
    }
    time_3pass /= iterations;
    time_cub /= iterations;
    std::cout << "3pass avg time (ms): " << time_3pass << "\n"
              << "cub avg time (ms): " << time_cub << std::endl;

    return 0;
}
