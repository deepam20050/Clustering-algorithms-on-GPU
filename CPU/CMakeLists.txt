cmake_minimum_required(VERSION 3.25)
project(CPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_executable(CPU main.cpp cpu.h utils.h includes.h benchmark.h)

find_package(OpenMP REQUIRED)

target_link_libraries(CPU PUBLIC OpenMP::OpenMP_CXX)