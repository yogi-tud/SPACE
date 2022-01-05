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
#include "data_generator.cuh"

#include "fast_prng.cuh"
#include "kernels/data_generator.cuh"

#include "csv_loader.hpp"
#include "utils.hpp"

#include <src/cub_wraps.cuh>
#include <src/kernels/kernel_3pass.cuh>
#include <src/kernels/kernel_streaming_add.cuh>

int gen_dummy_data()
{
    int cuda_dev_id = 0;
    CUDA_TRY(cudaSetDevice(cuda_dev_id));

    uint64_t cnt_b = 1 << 30; // mask size in bytes
    uint64_t cnt_e = cnt_b * 8; // element count

    uint8_t* d_mask;
    CUDA_TRY(cudaMalloc(&d_mask, sizeof(uint8_t) * cnt_b));
    printf("GPU generation...");
    kernel_generate_mask_uniform<<<64, 32>>>(d_mask, cnt_b, 0.5);
    CUDA_TRY(cudaDeviceSynchronize());
    printf("done\n");

    uint8_t* d_mask2;
    CUDA_TRY(cudaMalloc(&d_mask2, sizeof(uint8_t) * cnt_b));
    CUDA_TRY(cudaMemcpy(
        d_mask2, d_mask, sizeof(uint8_t) * cnt_b, cudaMemcpyDeviceToDevice));

    uint64_t* d_failure_count;
    CUDA_TRY(cudaMalloc(&d_failure_count, sizeof(uint64_t)));
    CUDA_TRY(cudaMemset(d_failure_count, 0x00, sizeof(uint64_t)));

    // introduce some artificial errors
    fast_prng rng(17);
    for (int i = 0; i < 20; i++) {
        uint64_t rand_idx = rng.rand() % cnt_b;
        CUDA_TRY(cudaMemset(d_mask2 + rand_idx, 0x00, sizeof(uint8_t)));
    }

    printf("GPU check validation...");
    kernel_check_validation<<<64, 32>>>(
        d_mask, d_mask2, cnt_b, d_failure_count);
    CUDA_TRY(cudaDeviceSynchronize());
    printf("done\n");

    uint64_t h_failure_count = 0;
    CUDA_TRY(cudaMemcpy(
        &h_failure_count, d_failure_count, sizeof(uint64_t),
        cudaMemcpyDeviceToHost));
    printf("validation failures = %lu\n", h_failure_count);

    CUDA_TRY(cudaFree(d_failure_count));
    CUDA_TRY(cudaFree(d_mask2));
    CUDA_TRY(cudaFree(d_mask));

    printf("done\n");
    return 0;
}

template <typename T>
float launch_3pass(
    cudaEvent_t start, cudaEvent_t end, T* d_input, T* d_output,
    uint8_t* d_mask, uint32_t* d_pss_total, size_t length)
{
    uint32_t chunk_length = 1024;
    uint32_t chunk_count = length / chunk_length;
    uint32_t* d_pss;
    CUDA_TRY(cudaMalloc(&d_pss, chunk_count * sizeof(uint32_t)));
    uint32_t* d_popc;
    CUDA_TRY(cudaMalloc(&d_popc, chunk_count * sizeof(uint32_t)));
    CUDA_TRY(cudaMemset(d_pss_total, 0x00, sizeof(uint32_t)));
    float time = 0;
    time += launch_3pass_popc_none(
        start, end, 0, 32, d_mask, d_pss, chunk_length, chunk_count);
    time += launch_3pass_popc_none(
        start, end, 0, 32, d_mask, d_popc, chunk_length, chunk_count);
    time += launch_cub_pss(0, start, end, d_pss, d_pss_total, chunk_count);
    time += launch_3pass_proc_true(
        start, end, 18432, 1024, d_input, d_output, d_mask, d_pss, true, d_popc,
        chunk_length, chunk_count);

    CUDA_TRY(cudaFree(d_pss));
    CUDA_TRY(cudaFree(d_popc));
    return time;
}
void validate(
    float* d_validation, float* d_output, uint64_t out_length,
    uint64_t* d_failure_count)
{
    kernel_check_validation<<<64, 32>>>(
        d_validation, d_output, out_length, d_failure_count);
    auto vec = gpu_to_vector(d_failure_count, 1);
    if (vec[0] != 0) {
        puts("validation failure!");
        exit(-1);
    }
}

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
void error(const char* error)
{
    fputs(error, stderr);
    assert(false);
    exit(EXIT_FAILURE);
}
void alloc_failure()
{
    error("memory allocation failed");
}
struct intermediate_data {
    uint32_t* d_pss;
    uint32_t* d_pss2;
    uint32_t* d_popc;
    uint32_t* d_out_count;
    uint8_t* d_cub_intermediate;
    size_t chunk_length;
    size_t element_count;
    size_t chunk_count;
    size_t max_stream_count;
    cudaEvent_t dummy_event_1;
    cudaEvent_t dummy_event_2;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaStream_t* streams;
    cudaEvent_t* stream_events;
    intermediate_data(
        size_t element_count, int chunk_length, int max_stream_count)
    {
        this->chunk_length = chunk_length;
        this->element_count = element_count;
        this->chunk_count = element_count / chunk_length;
        this->max_stream_count = max_stream_count;
        uint8_t* null = (uint8_t*)NULL;
        size_t intermediate_size_3pass = chunk_count * sizeof(uint32_t);
        if (chunk_length > 32) {
            // for the streaming kernel
            intermediate_size_3pass = element_count / 32 * sizeof(uint32_t);
        }
        size_t temp_storage_bytes_pss;
        CUDA_TRY(cub::DeviceScan::ExclusiveSum(
            null, temp_storage_bytes_pss, null, null, chunk_count));
        size_t temp_storage_bytes_flagged;
        cub::DeviceSelect::Flagged(
            null, temp_storage_bytes_flagged, null, null, null, null,
            element_count);
        size_t intermediate_size =
            std::max(temp_storage_bytes_pss, temp_storage_bytes_flagged);
        CUDA_TRY(cudaMalloc(&d_cub_intermediate, intermediate_size));
        CUDA_TRY(cudaMalloc(&d_pss, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(&d_pss2, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(&d_popc, intermediate_size_3pass));
        CUDA_TRY(cudaMalloc(
            &d_out_count, sizeof(uint32_t) * (max_stream_count + 1)));
        CUDA_TRY(cudaEventCreate(&dummy_event_1));
        CUDA_TRY(cudaEventCreate(&dummy_event_2));
        CUDA_TRY(cudaEventCreate(&start));
        CUDA_TRY(cudaEventCreate(&stop));
        streams =
            (cudaStream_t*)malloc(max_stream_count * sizeof(cudaStream_t));
        if (!streams) alloc_failure();
        stream_events =
            (cudaEvent_t*)malloc(max_stream_count * sizeof(cudaEvent_t));
        if (!stream_events) alloc_failure();
        for (int i = 0; i < max_stream_count; i++) {
            CUDA_TRY(cudaStreamCreate(&(streams[i])));
            CUDA_TRY(cudaEventCreate(&(stream_events[i])));
        }
    }
    template <typename T>
    void prepare_buffers(size_t element_count, int chunk_length, T* d_output)
    {
        CUDA_TRY(cudaMemset(d_out_count, 0, sizeof(*d_out_count)));
        CUDA_TRY(cudaMemset(d_output, 0, element_count * sizeof(T)));
        if (this->element_count > element_count ||
            this->chunk_length > chunk_length)
            return;
        error("sizes in intermediate data are smaller than the ones "
              "submitted "
              "to the algorithm");
    }
    ~intermediate_data()
    {
        CUDA_TRY(cudaFree(d_pss));
        CUDA_TRY(cudaFree(d_pss2));
        CUDA_TRY(cudaFree(d_popc));
        CUDA_TRY(cudaFree(d_cub_intermediate));
        CUDA_TRY(cudaFree(d_out_count));
        CUDA_TRY(cudaEventDestroy(dummy_event_1));
        CUDA_TRY(cudaEventDestroy(dummy_event_2));
        CUDA_TRY(cudaEventDestroy(start));
        CUDA_TRY(cudaEventDestroy(stop));
        for (int i = 0; i < max_stream_count; i++) {
            CUDA_TRY(cudaStreamDestroy(streams[i]));
            CUDA_TRY(cudaEventDestroy(stream_events[i]));
        }
        free(streams);
        free(stream_events);
    }
};

template <class T>
float bench1_base_variant(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
    float time = 0;
    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &time, {
        launch_3pass_popc_none(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size, d_mask,
            id->d_pss, chunk_length, id->chunk_count);
        launch_3pass_pss_gmem(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size,
            id->d_pss, id->chunk_count, id->d_out_count);
        launch_3pass_pss2_gmem(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size,
            id->d_pss, id->d_pss2, id->chunk_count);
        launch_3pass_proc_none(
            id->dummy_event_1, id->dummy_event_2, grid_size, block_size,
            d_input, d_output, d_mask, id->d_pss2, true, chunk_length,
            id->chunk_count);
    });
    return time;
}

template <class T>
float bench2_base_variant_shared_mem(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
    float time = 0;
    CUDA_TIME_FORCE_ENABLED(
        id->start, id->stop, 0, &time,
        {
            // TODO
        });
    return time;
}

template <class T>
float bench3_3pass_streaming(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size,
    int stream_count = 2)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
    float time = 0;
    if (stream_count < 1) {
        error("stream_count must be >= 1");
    }
    int p2_sc = 1;
    while (p2_sc < stream_count) {
        p2_sc *= 2;
    }
    if (stream_count != p2_sc) error("stream_count must be a power of 2");

    CUDA_TIME_FORCE_ENABLED(id->start, id->stop, 0, &time, {
        uint32_t max_chunk_count = element_count / 32;

        CUDA_TRY(cudaMemset(
            id->d_out_count, 0x00, sizeof(uint32_t) * (stream_count + 1)));

        uint64_t elems_per_stream = element_count / stream_count;
        uint64_t bytes_per_stream = elems_per_stream / 8;
        uint64_t chunks_per_stream = elems_per_stream / chunk_length;
        uint64_t chunks1024_per_stream = elems_per_stream / 1024;

        uint32_t chunk_count_p2 = 1;
        while (chunk_count_p2 < chunks_per_stream) {
            chunk_count_p2 *= 2;
        }

        uint32_t chunk_length32 = chunk_length / 32;
        int popc1_threadcount = 1024;
        int popc1_blockcount = (id->chunk_count / popc1_threadcount) + 1;
        int popc2_threadcount = 1024;
        int popc2_blockcount = (id->chunk_count / popc2_threadcount) + 1;
        const int proc_threadcount = 1024;
        int proc_blockcount = id->chunk_count / 1024;
        if (proc_blockcount < 1) {
            proc_blockcount = 1;
        }

        for (int i = 0; i < stream_count; i++) {
            // launch popc for i
            kernel_3pass_popc_none_monolithic<<<
                popc1_blockcount, popc1_threadcount, 0, id->streams[i]>>>(
                d_mask + bytes_per_stream * i,
                id->d_pss + chunks_per_stream * i, chunk_length32,
                chunks_per_stream);
            // launch pss for i
            launch_cub_pss(
                id->streams[i], 0, 0, id->d_pss + chunks_per_stream * i,
                id->d_out_count + i + 1, chunks_per_stream);
            // if i > 0: launch extra: add previous d->out_count to own output
            if (i > 0) {
                launch_streaming_add_pss_totals(
                    id->streams[i], id->d_out_count + i - 1,
                    id->d_out_count + i);
            }
            // record event i
            CUDA_TRY(cudaEventRecord(id->stream_events[i], id->streams[i]));
            // launch optimization popc 1024 for i
            kernel_3pass_popc_none_monolithic<<<
                popc2_blockcount, popc2_threadcount, 0, id->streams[i]>>>(
                d_mask + bytes_per_stream * i,
                id->d_popc + chunks1024_per_stream * i, 1024 / 32,
                chunks1024_per_stream);
            // if i > 0: wait for event i-1
            if (i > 0) {
                CUDA_TRY(cudaStreamWaitEvent(
                    id->streams[i], id->stream_events[i - 1]));
            }
            // launch optimized writeout proc for i using  d->out_count at i as
            // offset from output
            kernel_3pass_proc_true_striding<proc_threadcount, T, true>
                <<<proc_blockcount, proc_threadcount, 0, id->streams[i]>>>(
                    d_input + elems_per_stream * i, d_output,
                    d_mask + bytes_per_stream * i,
                    id->d_pss + chunks_per_stream * i,
                    id->d_popc + chunks1024_per_stream * i, chunk_length,
                    chunks_per_stream, chunk_count_p2, id->d_out_count + i);
        }

        CUDA_TRY(cudaDeviceSynchronize());
    });
    return time;
}

template <class T>
float bench4_3pass_optimized_read_skipping_partial_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
}

template <class T>
float bench5_3pass_optimized_read_skipping_two_phase_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
}

template <class T>
float bench6_3pass_optimized_read_skipping_cub_pss(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length, int block_size, int grid_size)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
}

template <class T>
float bench7_cub_flagged(
    intermediate_data* id, T* d_input, uint8_t* d_mask, T* d_output,
    size_t element_count, size_t chunk_length)
{
    id->prepare_buffers(element_count, chunk_length, d_output);
}

int main(int argc, char** argv)
{
    CUDA_TRY(cudaSetDevice(1));
    // load data
    std::vector<float> col;
    load_csv("../res/Arade_1.csv", {3}, col);
    float* d_input = vector_to_gpu(col);
    float* d_output = alloc_gpu<float>(col.size());
    // gen predicate mask
    auto pred = gen_predicate(
        col, +[](float f) { return f > 200; });
    uint8_t* d_mask = vector_to_gpu(pred);
    uint32_t* d_selected_out = alloc_gpu<uint32_t>(1);
    uint64_t* d_failure_count = alloc_gpu<uint64_t>(1);

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length =
        generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    // run benchmark
    cudaEvent_t start, end;
    CUDA_TRY(cudaEventCreate(&start));
    CUDA_TRY(cudaEventCreate(&end));
    const int iterations = 100;
    float time_3pass = 0;
    float time_cub = 0;
    for (int i = 0; i < iterations; i++) {
        time_3pass += launch_3pass(
            start, end, d_input, d_output, d_mask, d_selected_out, col.size());
        validate(d_validation, d_output, out_length, d_failure_count);

        time_cub += launch_cub_flagged_biterator(
            start, end, d_input, d_output, d_mask, d_selected_out, col.size());
        validate(d_validation, d_output, out_length, d_failure_count);
    }
    time_3pass /= iterations;
    time_cub /= iterations;
    std::cout << "3pass avg time (ms): " << time_3pass << "\n"
              << "cub avg time (ms): " << time_cub << std::endl;

    return 0;
}
