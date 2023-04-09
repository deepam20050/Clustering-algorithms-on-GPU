cmake_minimum_required(VERSION 3.25)
project(Main CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(Main main.cu cpu.cu cpu.cuh)

set_target_properties(Main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
