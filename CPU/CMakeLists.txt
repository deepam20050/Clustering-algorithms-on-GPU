cmake_minimum_required(VERSION 3.25)
project(CPU)

set(CMAKE_CXX_STANDARD 17)

add_executable(CPU main.cpp cpu.cpp cpu.h csv_parser.h)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(CPU PUBLIC OpenMP::OpenMP_CXX)
endif()
