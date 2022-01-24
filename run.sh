#running a number of experiments
# parameters: dataset selectivity datasize (input col only in MIB)
# uniform, zipf, csv, burst (0,1,2,3)
awk '
  BEGIN{
    for (i = 0.01; i < 1; i+ = 0.02)
    ./gpu_compressstore2 0 i 4096
  }'