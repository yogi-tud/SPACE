# SPACE is a novel approach to improve compaction of selected items on NVIDIA GPUs

Prequisites:
Linux
CUDA 11 
GCC 9.3.0
Cmake 3.22

 build the experiment framework
./build.sh  

run benchmark from 1% to 97% selected data on the current platform
nohup python3 run.py

run low selectivity benchmark 1% ... 10^-6%
nohup python3 small.py

