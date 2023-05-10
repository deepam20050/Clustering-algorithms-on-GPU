//
// Created by deepam on 8/5/23.
//

#ifndef GPU_DBSCAN_CUH
#define GPU_DBSCAN_CUH

#include "includes.cuh"

using namespace std;

__global__ void find_core (int N, float eps, int minPts, float *d_x, float *d_y, int *d_core, int *pr) {
  const int i = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  const float x0 = d_x[i], y0 = d_y[i];
  int cnt = 0;
  for (int j = 0; j < N; ++j) {
    if (sqrtf(l2_norm_sq(x0, y0, d_x[j], d_y[j])) <= eps) {
      ++cnt;
    }
  }
  if (cnt >= minPts) {
    d_core[i] = 1;
    pr[i] = i + 1;
  }
}

__global__ void merge (int N, float eps, float *d_x, float *d_y, int *d_core, int *pr, int pr_idx) {
  const int i = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  const float x0 = d_x[pr_idx], y0 = d_y[pr_idx];
  const float x = d_x[i], y = d_y[i];
  if (d_core[i] && sqrtf(l2_norm_sq(x0, y0, x, y)) <= eps) {
    atomicMin(&pr[pr_idx], pr[i]);
  }
}

__global__ void merge_cores (int N, float eps, float *d_x, float *d_y, int *pr, int pr_idx) {
  const int i = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  const int curr_pr = pr[pr_idx];
  const float x0 = d_x[pr_idx], y0 = d_y[pr_idx];
  const float x = d_x[i], y = d_y[i];
  if (sqrtf(l2_norm_sq(x0, y0, x, y)) <= eps) {
    pr[i] = curr_pr;
  }
}

pair < float, vector < int > > run_DBSCAN (vector < float > &h_x, vector < float > &h_y, float eps, int minPts, int THREADS = 1024) {
  const int N = static_cast < int > (h_x.size());
  float *d_x, *d_y;
  int *d_core, *d_pr, *h_pr = new int[N], *h_core = new int[N];

  // Creating Streams for Asynchronous execution
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  // Allocate memory on GPU
  cudaMallocAsync(&d_x, N * sizeof(float), stream1);
  cudaMallocAsync(&d_y, N * sizeof(float), stream1);
  cudaMallocAsync(&d_core, N * sizeof(int), stream1);
  cudaMallocAsync(&d_pr, N * sizeof(int), stream1);

  // Copy/Set their values
  cudaMemcpyAsync(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemsetAsync(d_pr, 0, N * sizeof(int), stream1);
  cudaMemsetAsync(d_core, 0, N * sizeof(int), stream1);

  const int BLOCKS = (N + THREADS - 1) / THREADS;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream1);

  // DBSCAN for GPU
  find_core<<<BLOCKS, THREADS, 0, stream1>>>(N, eps, minPts, d_x, d_y, d_core, d_pr);
  cudaMemcpyAsync(h_core, d_core, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_pr, d_pr, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
  // cudaStreamSynchronize(stream1);
  cudaDeviceSynchronize();
  for (int i = 0; i < N; ++i) {
    if (h_core[i] == 1) {
      int blocks = (N - i + THREADS - 2) / THREADS;
      merge<<<BLOCKS, THREADS, 0, stream1>>>(N, eps, d_x, d_y, d_core, d_pr, i);
      merge_cores<<<BLOCKS, THREADS, 0, stream1>>>(N, eps, d_x, d_y, d_pr, i);
    }
  }
  cudaMemcpyAsync(h_pr, d_pr, N * sizeof(int), cudaMemcpyDeviceToHost, stream1);
  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);
  float msecs;
  cudaEventElapsedTime(&msecs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamSynchronize(stream1);
  cudaStreamDestroy(stream1);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_core);
  cudaFree(d_pr);
  auto ret = make_pair(msecs, vector < int > (h_pr, h_pr + N));
  delete[] h_pr;
  delete[] h_core;
  return ::move(ret);
}

#endif //GPU_DBSCAN_CUH
