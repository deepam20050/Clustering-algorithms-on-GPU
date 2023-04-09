cmake_minimum_required(VERSION 3.25)
project(Main Languages CXX CUDA)

find_package(OpenMP)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

add_executable(Main main.cu cpu.cu cpu.cuh)

target_compile_options(<target> PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

set_target_properties(Main PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
