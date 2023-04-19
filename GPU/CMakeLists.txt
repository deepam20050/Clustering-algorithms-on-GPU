cmake_minimum_required(VERSION 3.25)
project(GPU CUDA)

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca")

add_executable(GPU main.cu includes.cuh benchmark.cuh gpu.cuh utils.cuh)

set_target_properties(GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
