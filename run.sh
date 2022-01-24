#running a number of experiments
# parameters: dataset selectivity datasize (input col only in MIB)
# uniform, zipf, csv, burst (0,1,2,3)
maxmb="4096"
dataset="0"

./build/gpu_compressstore2 $dataset 0.0002 4096
./build/gpu_compressstore2 $dataset 0.002 4096
./build/gpu_compressstore2 $dataset 0.02 4096
./build/gpu_compressstore2 $dataset 0.04 4096
./build/gpu_compressstore2 $dataset 0.06 4096
./build/gpu_compressstore2 $dataset 0.08 4096
./build/gpu_compressstore2 $dataset 0.10 4096
./build/gpu_compressstore2 $dataset 0.12 4096
./build/gpu_compressstore2 $dataset 0.14 4096
./build/gpu_compressstore2 $dataset 0.16 4096
./build/gpu_compressstore2 $dataset 0.18 4096
./build/gpu_compressstore2 $dataset 0.20 4096
./build/gpu_compressstore2 $dataset 0.22 4096
./build/gpu_compressstore2 $dataset 0.24 4096
./build/gpu_compressstore2 $dataset 0.26 4096
./build/gpu_compressstore2 $dataset 0.28 4096
./build/gpu_compressstore2 $dataset 0.3 4096
./build/gpu_compressstore2 $dataset 0.32 4096

dataset="1"

./build/gpu_compressstore2 $dataset 0.0002 4096
./build/gpu_compressstore2 $dataset 0.002 4096
./build/gpu_compressstore2 $dataset 0.02 4096
./build/gpu_compressstore2 $dataset 0.04 4096
./build/gpu_compressstore2 $dataset 0.06 4096
./build/gpu_compressstore2 $dataset 0.08 4096
./build/gpu_compressstore2 $dataset 0.10 4096
./build/gpu_compressstore2 $dataset 0.12 4096
./build/gpu_compressstore2 $dataset 0.14 4096
./build/gpu_compressstore2 $dataset 0.16 4096
./build/gpu_compressstore2 $dataset 0.18 4096
./build/gpu_compressstore2 $dataset 0.20 4096
./build/gpu_compressstore2 $dataset 0.22 4096
./build/gpu_compressstore2 $dataset 0.24 4096
./build/gpu_compressstore2 $dataset 0.26 4096
./build/gpu_compressstore2 $dataset 0.28 4096
./build/gpu_compressstore2 $dataset 0.3 4096
./build/gpu_compressstore2 $dataset 0.32 4096

dataset="3"

./build/gpu_compressstore2 $dataset 0.0002 4096
./build/gpu_compressstore2 $dataset 0.002 4096
./build/gpu_compressstore2 $dataset 0.02 4096
./build/gpu_compressstore2 $dataset 0.04 4096
./build/gpu_compressstore2 $dataset 0.06 4096
./build/gpu_compressstore2 $dataset 0.08 4096
./build/gpu_compressstore2 $dataset 0.10 4096
./build/gpu_compressstore2 $dataset 0.12 4096
./build/gpu_compressstore2 $dataset 0.14 4096
./build/gpu_compressstore2 $dataset 0.16 4096
./build/gpu_compressstore2 $dataset 0.18 4096
./build/gpu_compressstore2 $dataset 0.20 4096
./build/gpu_compressstore2 $dataset 0.22 4096
./build/gpu_compressstore2 $dataset 0.24 4096
./build/gpu_compressstore2 $dataset 0.26 4096
./build/gpu_compressstore2 $dataset 0.28 4096
./build/gpu_compressstore2 $dataset 0.3 4096
./build/gpu_compressstore2 $dataset 0.32 4096