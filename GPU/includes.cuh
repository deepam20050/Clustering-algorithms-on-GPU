//
// Created by deepam on 16/4/23.
//

#ifndef GPU_INCLUDES_CUH
#define GPU_INCLUDES_CUH

#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <fstream>
#include <chrono>
#include <cassert>
#include <algorithm>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cuda_runtime.h>
#include <utility>

using namespace std;

__device__ __host__ float l2_norm_sq (float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__device__ float gaussian_kernel (float x_1, float y_1, float x_2, float y_2) {
  return __expf(-0.5 * l2_norm_sq(x_1, y_1, x_2, y_2));
}

const int THREADS = 1024;

#endif //GPU_INCLUDES_CUH
