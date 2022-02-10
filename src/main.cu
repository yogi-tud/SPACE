#include <bitset>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <filesystem>
#include <stdio.h>
#include <thread>
#include <vector>
#include <bitset>
#include <string>
#define MEBIBYTE (1<<20)
#define DISABLE_CUDA_TIME
#define DELIM ";"
#include "cuda_time.cuh"
#include "cuda_try.cuh"

#include "csv_loader.hpp"
#include "utils.cuh"
#include "data_generator.cuh"
#include "benchmarks.cuh"

/***
 *
 *
 * @param total_elements number of bits in mask
 * @param selectivity percentage of elements that are set to 1 in bitmask
 * @param cluster_count on how many clusters should the elements distributed
 * @return a bitmask as uint8_t which is used for compressstore operations
 *
 *
 */

std::vector<uint8_t> create_bitmask(float selectivity, size_t cluster_count, size_t total_elements)
{
        std::vector<bool> bitset;
        bitset.resize(total_elements);
    size_t total_set_one = selectivity * total_elements;
    size_t cluster_size = total_set_one / cluster_count;
    size_t slice = bitset.size() / cluster_count;

        //start by setting all to zero
        for(int i=0; i<bitset.size();i++)
        {
            bitset[i]=0;
        }

    for (int i = 0 ; i< cluster_count;i++)
    {
        for(int k =0;k< cluster_size;k++)
        {
            size_t cluster_offset = i*slice;
            bitset[k+cluster_offset]=1;
        }
    }


    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements/8);

    for(int i=0; i<total_elements/8;i++)
    {
        final_bitmask_cpu[i]=0;
    }

    for(int i =0;i< bitset.size();i++)
    {
        //set bit of uint8
        if(bitset[i])
        {
            uint8_t current =final_bitmask_cpu[i/8];
            int location = i % 8;
            current = 1 << (7-location);
            uint8_t add_res = final_bitmask_cpu[i/8];
            add_res = add_res | current;
            final_bitmask_cpu[i/8] = add_res;
        }

    }


    return final_bitmask_cpu;
}

template <typename T> void benchmark (int argc, char** argv, string datatype)
{
    int dataset_pick=0;

    size_t datasize_MIB = 1024;
    size_t cluster_count =0;
    float sel = 0.025;
    int iterations = 2;
    bool report_failures = true;
    string dataset = "";
    string device = "";
    //dataset pick (0 uniform, 1 1cluster, 2 multiclsuter)
    if (argc > 1) {
        dataset_pick = atoi(argv[1]);


    }
    //selectivity
    if(argc > 2)
    {
        sel = atof(argv[2]);

    }
    //datasize
    if(argc > 3)
    {
        datasize_MIB  = atoi(argv[3]);

    }
    if(argc > 4)
    {
        cluster_count  = atoi(argv[4]);

    }
    if(argc > 6)
    {
        stringstream ss;
        ss <<argv[6];
        ss >> device;


    }



    std::vector<uint8_t> mask1= create_bitmask(1,2,64);




    std::vector<T> col;
    size_t ele_count = MEBIBYTE* datasize_MIB / sizeof (T);


    col.resize(ele_count);


    size_t one_count=sel*ele_count;
    size_t mask_bytes = (col.size()/8)+1 ;
    uint8_t* pred = (uint8_t*) malloc(mask_bytes); //mask on host

    std::vector<uint8_t> im;

    //dataset pick (0 uniform, 1 1cluster, 2 multiclsuter)
    switch(dataset_pick) {
        case 0:
            cluster_count =1;
            col=genRandomInts<T>(ele_count, 45000);
            generate_mask_uniform(pred, 0, mask_bytes, sel);
            dataset="uniform";
            break;

        case 1: //generate 1 big cluster
            cluster_count =1;
            im = create_bitmask(sel,1,ele_count);
            pred= im.data();
            // generate_mask_zipf(pred,one_count,0,mask_bytes);
           col= genRandomInts<T>(ele_count, 45000);

            dataset="single_cluster";
            break;

        case 2:

            im = create_bitmask(sel,cluster_count,ele_count);
            pred= im.data();
            col=genRandomInts<T>(ele_count, 45000);

            dataset="multi_cluster";



    }


    CUDA_TRY(cudaSetDevice(0));

    //set up GPU pointers
    T * d_input = vector_to_gpu(col);
    T * d_output = alloc_gpu<T>(col.size() + 1);
    uint8_t* d_mask;



    CUDA_TRY(cudaMalloc(&d_mask, mask_bytes));
    CUDA_TRY(cudaMemcpy(d_mask, &pred[0], mask_bytes, cudaMemcpyHostToDevice));



    //uint8_t* d_mask = vector_to_gpu(pred);


   // printf("line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<T> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    T* d_validation = vector_to_gpu(validation);

    // prepare candidates for benchmark
    intermediate_data id{col.size(), 1024, 8, (T*)NULL}; // setup shared intermediate data



    //building config string for csv output
    std::string gridblock ="";


    std::vector<std::pair<std::string, std::function<timings(int, int, int)>>> benchs;


    benchs.emplace_back(
        "bench1_base_variant", [&](int cs, int bs, int gs) { return bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs); });
    benchs.emplace_back("bench2_base_variant_skipping", [&](int cs, int bs, int gs) {
      return bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench3_3pass_streaming", [&](int cs, int bs, int gs) {
      return bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, bs, gs);
    });



    benchs.emplace_back("bench4_3pass_optimized_read_non_skipping_cub_pss", [&](int cs, int bs, int gs) {
      return bench4_3pass_optimized_read_non_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench5_3pass_optimized_read_skipping_partial_pss", [&](int cs, int bs, int gs) {
      return bench5_3pass_optimized_read_skipping_partial_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench6_3pass_optimized_read_skipping_two_phase_pss", [&](int cs, int bs, int gs) {
      return bench6_3pass_optimized_read_skipping_two_phase_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench7_3pass_optimized_read_skipping_cub_pss", [&](int cs, int bs, int gs) {
      return bench7_3pass_optimized_read_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });
    benchs.emplace_back("bench8_cub_flagged", [&](int cs, int bs, int gs) { return bench8_cub_flagged(&id, d_input, d_mask, d_output, col.size()); });

    benchs.emplace_back("bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss", [&](int cs, int bs, int gs) {
      return bench10_3pass_optimized_read_skipping_optimized_writeout_cub_pss(&id, d_input, d_mask, d_output, col.size(), cs, bs, gs);
    });


    // run benchmark
    /**
    int grid_size_min =256;
    int grid_size_max =4096;
    int block_size_min =128;
    int block_size_max =1024;
    int chunk_length_min =256;
    int chunk_length_max =2048;
     **/

    int grid_size_min =4096;
    int grid_size_max =4096;
    int block_size_min =1024;
    int block_size_max =1024;
    int chunk_length_min =512;
    int chunk_length_max =4096;
    size_t MBSIZE= col.size() * sizeof(T) / MEBIBYTE;
    size_t MASKSIZE = col.size()  / MEBIBYTE;
    size_t total_size= MBSIZE + MASKSIZE;

    std::vector<string> results;
    string current_path (std::filesystem::current_path());

    string filename = current_path+"/"+dataset+"_"+device+"_"+datatype+".txt";
    fstream myfile(filename,std::ios_base::app | std::ios_base::trunc);
    myfile.open(filename);



    //only write header if output file is empty
    if(myfile.peek() == std::ifstream::traits_type::eof())
    {





        ofstream myfile_out(filename);

        myfile_out << "benchmark;chunk_length;cluster;block_size;grid_size;time_popc;time_pss1;time_pss2;time_proc;time_total;selectivity,throughput" << endl;


        myfile_out.close();

    }

    myfile.close();
    myfile.open(filename);
    myfile.seekg (0, ios::end);


    for (int grid_size = grid_size_min; grid_size <= grid_size_max; grid_size *= 2) {
        for (int block_size = block_size_min; block_size <= block_size_max; block_size *= 2) {
            for (int chunk_length = chunk_length_min; chunk_length <= chunk_length_max; chunk_length *= 2) {
                std::vector<timings> timings(benchs.size());
                for (int it = 0; it < iterations; it++) {
                    for (size_t i = 0; i < benchs.size(); i++) {
                        timings[i] += benchs[i].second(chunk_length, block_size, grid_size);
                        size_t failure_count;
                        if (!validate(&id, d_validation, d_output, out_length, report_failures, &failure_count)) {
                            fprintf(
                                stderr, "validation failure in bench %s (%d, %d, %d), run %i: %zu failures\n", benchs[i].first.c_str(), chunk_length,
                                block_size, grid_size, it, failure_count);
                            // exit(EXIT_FAILURE);
                        }
                    }
                }
                for (int i = 0; i < benchs.size(); i++) {
                    myfile << benchs[i].first << ";" << chunk_length << ";" <<cluster_count<<";"<< block_size << ";" << grid_size << ";"
                              << timings[i] / static_cast<float>(iterations) << ";"
                                <<sel <<";"<<(static_cast<float>(total_size) / timings[i].total) * (float)(1000.0/1024.0)<<endl;

                }
            }
        }
    }







    myfile.close();

    CUDA_TRY(cudaFree(d_mask));
    CUDA_TRY(cudaFree(d_input));
    CUDA_TRY(cudaFree(d_output));



}





int main(int argc, char** argv)
{
    int pick_datatype=-1;
    string datatype ="";
    if(argc > 5)
    {
        pick_datatype  = atoi(argv[5]);

    }

    switch (pick_datatype)
    {
        case 1: datatype="uint8_t";
            benchmark<uint8_t>(argc, argv,datatype);
            break;
        case 2: datatype="uint16_t";
            benchmark<uint16_t>(argc, argv,datatype);
            break;
        case 3: datatype="uint32_t";
            benchmark<uint32_t>(argc, argv,datatype);
            break;
        case 4: datatype="int";
            benchmark<int>(argc, argv,datatype);
            break;
        case 5:  datatype="float";
            benchmark<float>(argc, argv,datatype);
            break;
        case 6:  datatype="double";
            benchmark<double>(argc, argv,datatype);
            break;
        case 7:  datatype="uint64_t";
            benchmark<uint64_t>(argc, argv,datatype);
            break;


    }




    return 0;
}






