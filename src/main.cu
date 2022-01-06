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
    std::vector<float> col{10000, 2, 3, 4, 5, 6, 7, 8};
    // load_csv("../res/Arade_1.csv", {3}, col);

    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size() + 1);
    float* canary = d_output + col.size();
    val_to_gpu(canary, 17);

    // gen predicate mask
    size_t one_count;
    auto pred = gen_predicate(
        col, +[](float f) { return f > 2000; }, &one_count);
    pred.push_back(255);
    uint8_t* d_mask = vector_to_gpu(pred);
    pred.pop_back();

    printf("line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    // run benchmark
    intermediate_data id{col.size(), 1024, 8};
    const int iterations = 100;

    float time_3pass = 0;
    float time_cub = 0;

    std::vector<std::pair<std::string, std::function<float()>>> benchs;
    /*
    benchs.emplace_back("bench1_base_variant", [&]() { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
   benchs.emplace_back("bench2_base_variant_shared_mem", [&]() {
        return bench2_base_variant_shared_mem(
            &id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });*/
    benchs.emplace_back(
        "bench3_3pass_streaming", [&]() { return bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024); });
    /*
    benchs.emplace_back("bench4_3pass_optimized_read_skipping_partial_pss", [&]() {
        return bench4_3pass_optimized_read_skipping_partial_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench5_3pass_optimized_read_skipping_two_phase_pss", [&]() {
        return bench5_3pass_optimized_read_skipping_two_phase_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench6_3pass_optimized_read_skipping_cub_pss", [&]() {
        return bench6_3pass_optimized_read_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), 1024, 256, 1024);
    });
    benchs.emplace_back("bench7_cub_flagged", [&]() { return bench7_cub_flagged(&id, d_input, d_mask, d_output, col.size()); });
    ;*/
    std::vector<float> timings(benchs.size(), 0.0f);
    for (int it = 0; it < iterations; it++) {
        for (int i = 0; i < benchs.size(); i++) {
            timings[i] += benchs[i].second();
            size_t failure_count;
            if (!validate(&id, d_validation, d_output, out_length, &failure_count)) {
                fprintf(stderr, "validation failure in bench %s, run %i: %zu failures\n", benchs[i].first.c_str(), it, failure_count);
                // exit(EXIT_FAILURE);
            }
        }
    }
    for (int i = 0; i < benchs.size(); i++) {
        std::cout << "benchmark " << benchs[i].first << " time (ms): " << (double)timings[i] / iterations << std::endl;
    }
    if (gpu_to_val(canary) != 17) error("who ate the canary ? /(*.*)\\");
    return 0;
}
