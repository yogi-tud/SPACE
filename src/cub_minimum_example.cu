/**
   //datatype as input
   int pick_datatype=-1;
   string datatype ="";
   if(argc > 5)
   {
       pick_datatype  = atoi(argv[5]);

   }

   // Declare, allocate, and initialize device-accessible pointers for input, flags, and output
   int  num_items=64;              // e.g., 8
   std::vector<int> d_in_v= genRandomInts<int>(num_items,10);// e.g., [1, 2, 3, 4, 5, 6, 7, 8]
   std::vector<uint8_t> d_flags_v= create_bitmask(0.5,1,num_items);               // e.g., [1, 0, 0, 1, 0, 1, 1, 0]


   int  *d_out= alloc_gpu<int>(num_items);                 // e.g., [ ,  ,  ,  ,  ,  ,  ,  ]
   int  *d_num_selected_out=alloc_gpu<int>(1);    // e.g., [ ]
   float runtime =0;
   int * d_in = vector_to_gpu<int>(d_in_v);
   char* d_flags;
   cudaEvent_t start, stop;
   CUDA_TRY(cudaEventCreate(&start));
   CUDA_TRY(cudaEventCreate(&stop));

   CUDA_TRY(cudaMalloc(&d_flags, num_items));
   CUDA_TRY(cudaMemcpy(d_flags, &d_flags_v[0], num_items, cudaMemcpyHostToDevice));

// Determine temporary device storage requirements
   void     *d_temp_storage = NULL;
   size_t   temp_storage_bytes = 0;
   cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items);

   CUDA_TRY(cudaMalloc(&d_temp_storage, temp_storage_bytes));


   CUDA_TRY(cudaEventRecord(start));

   CUDA_TRY(cub::DeviceSelect::Flagged(d_temp_storage, temp_storage_bytes, d_in, d_flags, d_out, d_num_selected_out, num_items));
   CUDA_TRY(cudaEventRecord(stop));
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&runtime, start, stop);

   cout<<"OUT BUFFER: "<<endl;
   gpu_buffer_print(d_num_selected_out,0,1);

   cout<< "RUNTIME: "<<runtime<<endl;
   cout<<"CPU MASK: "<<endl;
    cpu_buffer_print(d_flags_v.data(),0,64);

   cout<< "INPUT: "<<endl;
   gpu_buffer_print(d_in,0,50);
   cout<< "MASK: "<<endl;
   gpu_buffer_print(d_flags,0,8);
   cout<< "OUTPUT: "<<endl;
   gpu_buffer_print(d_out,0,50);
   **/