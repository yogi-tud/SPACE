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
    /**
    for(int i=0;i<32;i++)
    {
    cout<<bitset[i];
    }
    cout<<endl;
    **/



    std::vector<uint8_t> final_bitmask_cpu;
    final_bitmask_cpu.resize(total_elements/8);

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

int main(int argc, char** argv)
{













    int dataset_pick=0;

    size_t datasize_MIB = 1024;
    size_t cluster_count =1;
    float sel = 0.025;
    int iterations = 3;
    bool report_failures = true;
    string dataset = "";

    //dataset pick (0 uniform, 1 1cluster, 2 multiclsuter)
    if (argc > 1) {
        dataset_pick = atoi(argv[1]);
        printf("setting dataset to %i\n", dataset_pick);

    }
    //selectivity
    if(argc > 2)
    {
         sel = atof(argv[2]);
        cout<<"SELECTIVITY: "<<sel<<endl;
    }
    //datasize
    if(argc > 3)
    {
        datasize_MIB  = atoi(argv[3]);
        cout<<"datasize: "<<datasize_MIB<<endl;
    }
    if(argc > 4)
    {
        cluster_count  = atoi(argv[4]);
        cout<<"cluste count: "<<cluster_count<<endl;
    }

    std::vector<uint8_t> mask1= create_bitmask(1,2,64);




    std::vector<float> col;
    size_t ele_count = MEBIBYTE* datasize_MIB / sizeof (float);


    col.resize(ele_count);


    size_t one_count=sel*ele_count;
    size_t mask_bytes = (col.size()/8)+1 ;
    uint8_t* pred = (uint8_t*) malloc(mask_bytes); //mask on host

    std::vector<uint8_t> im;

    //dataset pick (0 uniform, 1 1cluster, 2 multiclsuter)
    switch(dataset_pick) {
        case 0:
            genRandomInts(ele_count, 45000);
            generate_mask_uniform(pred, 0, mask_bytes, sel);
            dataset="uniform";
        break;

        case 1: //generate 1 big cluster
            im = create_bitmask(sel,1,ele_count);
            pred= im.data();
            // generate_mask_zipf(pred,one_count,0,mask_bytes);
            genRandomInts(ele_count, 45000);

            dataset="single_cluster";
        break;

       case 2:

            im = create_bitmask(sel,cluster_count,ele_count);
            pred= im.data();
            genRandomInts(ele_count, 45000);

            dataset="multi_cluster";



    }


    CUDA_TRY(cudaSetDevice(1));

    float * d_input = vector_to_gpu(col);
    float * d_output = alloc_gpu<float>(col.size() + 1);

   // generate_mask_uniform( pred,0,col.size(),0.01);
    //generate_mask_zipf(pred,col.size()/1000,0,col.size());
      cpu_buffer_print(pred,0,20);
    cout<<"ELEMENTS: "<<ele_count<<endl;
     //auto pred = gen_predicate(
    //    col, +[](float f) { return f > 2000; }, &one_count);



    uint8_t* d_mask;
    size_t size = col.size()*sizeof(uint8_t);
    CUDA_TRY(cudaMalloc(&d_mask, mask_bytes));
    CUDA_TRY(cudaMemcpy(d_mask, &pred[0], mask_bytes, cudaMemcpyHostToDevice));



    //uint8_t* d_mask = vector_to_gpu(pred);


    printf("line count: %zu, one count: %zu, percentage: %f\n", col.size(), one_count, (double)one_count / col.size());

    // gen cpu side validation
    std::vector<float> validation;
    validation.resize(col.size());
    size_t out_length = generate_validation(&col[0], &pred[0], &validation[0], col.size());
    float* d_validation = vector_to_gpu(validation);

    // prepare candidates for benchmark
    intermediate_data id{col.size(), 1024, 8}; // setup shared intermediate data

    //pair: experiment name, blocksize, gridsize, time ms

    std::vector<std::pair<std::string, float>> benchs;
    //timings for single kernels of a benchmark
    std::vector<std::pair<std::string, float>> subtimings;

    //building config string for csv output
    std::string gridblock ="";

    //set up benchmarks for different cuda configs and algorithm on same data set
        for(size_t blocksize = 256; blocksize <=1024 ; blocksize = blocksize * 2 ) {
        for (size_t gridsize = 1024; gridsize <= 32768; gridsize = gridsize * 2) {



            gridblock = ";"+std::to_string(blocksize)+";"+std::to_string(gridsize);


            benchs.emplace_back("bench1_base_variant"+ gridblock

                                    , bench1_base_variant(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));


            benchs.emplace_back(
                "bench2_base_variant_skipping"+gridblock, bench2_base_variant_skipping(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));

            benchs.emplace_back("bench3_3pass_streaming"+gridblock, bench3_3pass_streaming(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));

            benchs.emplace_back(
                "bench4_optimized_read_non_skipping_cub_pss"+gridblock,
                bench4_optimized_read_non_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));

            benchs.emplace_back(
                "bench5_3pass_optimized_read_skipping_partial_pss"+gridblock,
                bench5_3pass_optimized_read_skipping_partial_pss(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));

            benchs.emplace_back(
                "bench6_3pass_optimized_read_skipping_two_phase_pss"+gridblock,
                bench6_3pass_optimized_read_skipping_two_phase_pss(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));

            benchs.emplace_back(
                "bench7_3pass_optimized_read_skipping_cub_pss"+gridblock,
                bench7_3pass_optimized_read_skipping_cub_pss(&id, d_input, d_mask, d_output, col.size(), 1024, blocksize, gridsize));
        }
    }

    //cub has static block/thread config
    benchs.emplace_back("bench8_cub_flagged;0;0",  bench8_cub_flagged(&id, d_input, d_mask, d_output, col.size()));

    //print subtimings:

    //gpu_buffer_print(d_mask,0,500);

    // run benchmark
    std::vector<float> timings(benchs.size(), 0.0f);
    for (int it = 0; it < iterations; it++) {
        for (size_t i = 0; i < benchs.size(); i++) {
            timings[i] += benchs[i].second;
            size_t failure_count;
            if (!validate(&id, d_validation, d_output, out_length, report_failures, &failure_count)) {
                fprintf(stderr, "validation failure in bench %s, run %i: %zu failures\n", benchs[i].first.c_str(), it, failure_count);
                // exit(EXIT_FAILURE);
            }
        }
    }
    std::cout<<"Number of experiments: "<<benchs.size()<<  std::endl;

     string current_path (std::filesystem::current_path());



    string device = "_rtx8000";
    //string filename = "../data/"+dataset+device+".txt";
   // string filename = "/home/fett/edbt/EDBT_2022/data"+dataset+device+".txt";
    string filename = current_path+"/"+dataset+device+".txt";
    //string filename = dataset+device+".txt";
    //std::cout << "Current path is " << current_path << '\n'; // (1)



    write_bench_file(filename,benchs, timings,iterations ,col.size() ,dataset ,(double)one_count / col.size() );

  //  for (int i = 0; i < benchs.size(); i++) {
    //    std::cout<<subtimings[i].first<<" "<<subtimings[i].second<<std::endl;
        //std::cout << "benchmark " << benchs[i].first << " time (ms): " << (double)timings[i] / iterations << std::endl;
       // write_benchmark(col.size(),dataset,(double)one_count / col.size(),myfile,(double)timings[i] / iterations,benchs[i].first);

    //}
    return 0;
}






