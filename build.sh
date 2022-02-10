script_dir="$(dirname "$(readlink -f "$0")")"

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake --build $script_dir/build --target clean -- -j 12
cmake --build $script_dir/build --target gpu_compressstore2 -- -j 12
