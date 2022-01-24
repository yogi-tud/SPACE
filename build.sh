script_dir="$(dirname "$(readlink -f "$0")")"

build_dir="$script_dir/build"
data_dir="$script_dir/data"
mkdir -p "$script_dir"
mkdir -p "$build_dir"
cd "$build_dir"
#rm -rf *

export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc

cmake -DCMAKE_BUILD_TYPE=Release -DAVXPOWER=ON ..
cmake --build $build_dir --target gpu_compressstore2 -- -j 12