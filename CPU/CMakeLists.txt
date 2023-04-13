cmake_minimum_required(VERSION 3.25)
project(CPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")


add_executable(CPU main.cpp cpu.cpp cpu.h csv_parser.h test.h)
find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_link_libraries(CPU PUBLIC OpenMP::OpenMP_CXX)
endif()
