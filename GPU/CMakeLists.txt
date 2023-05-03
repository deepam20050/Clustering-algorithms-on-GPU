cmake_minimum_required(VERSION 3.25)
project(GPU CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(GPU main.cu includes.cuh benchmark.cuh gpu.cuh utils.cuh)

set_target_properties(GPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
