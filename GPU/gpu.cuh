//
// Created by deepam on 19/4/23.
//

#ifndef GPU_GPU_CUH
#define GPU_GPU_CUH

#include "includes.cuh"

using namespace std;

__device__ float l2_norm_sq (float x_1, float y_1, float x_2, float y_2) {
  return (x_1 - x_2) * (x_1 - x_2) + (y_1 - y_2) * (y_1 - y_2);
}

__global__ void compute_new_means(float*  means_x, float*  means_y, const float*  new_sum_x, const float*  new_sum_y, const int*  counts) {
  const int cluster = static_cast < int > (threadIdx.x);
  const auto count = static_cast < float > (max(1, counts[cluster]));
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;
}

__global__ void assign_clusters(int * cluster, const float*  data_x, const float*  data_y, int N, float*  means_x, float*  means_y, float*  new_sums_x, float*  new_sums_y, int K, int*  counts) {
  const int index = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= N) return;

  const float x = data_x[index];
  const float y = data_y[index];

  extern __shared__ float shared_means[];
  if (threadIdx.x < K) {
    shared_means[threadIdx.x] = means_x[threadIdx.x];
    shared_means[K + threadIdx.x] = means_y[threadIdx.x];
  }
  __syncthreads();
  float nearest_distance = FLT_MAX;
  int cluster_id = 0;
  for (int k = 0; k < K; ++k) {
    const float l2_norm_sqq = l2_norm_sq(x, y,  shared_means[k],shared_means[k + K]);
    if (l2_norm_sqq < nearest_distance) {
      nearest_distance = l2_norm_sqq;
      cluster_id = k;
    }
  }
  cluster[index] = cluster_id;
  // To optimize by final evaluation
  atomicAdd(&new_sums_x[cluster_id], x);
  atomicAdd(&new_sums_y[cluster_id], y);
  atomicAdd(&counts[cluster_id], 1);
}

pair < float, vector < int > > run_K_Means (vector < float > &h_x, vector < float > &h_y, int K, int NoOfIterations) {
  const int N = static_cast < int > (h_x.size());
  float *d_x, *d_y, *d_means_x, *d_means_y, *d_sums_x, *d_sums_y;
  int *d_counts, *d_cluster, h_cluster[N];

  // Allocate memory on GPU
  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));
  cudaMalloc(&d_means_x, K * sizeof (float));
  cudaMalloc(&d_means_y, K * sizeof (float));
  cudaMalloc(&d_sums_x, K * sizeof (float));
  cudaMalloc(&d_sums_y, K * sizeof (float));
  cudaMalloc(&d_counts, K * sizeof(int));
  cudaMalloc(&d_cluster, N * sizeof(int));

  // Copy/Set their values
  cudaMemcpy(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice);

  mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

  uniform_int_distribution < int > indices(0, N - 1);
  vector < float > h_means_x(K), h_means_y(K);
  for (int i = 0; i < K; ++i) {
    int idx = indices(rng);
    h_means_x[i] = h_x[idx];
    h_means_y[i] = h_y[idx];
  }

  cudaMemcpy(d_means_x, h_means_x.data(), K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_means_y, h_means_y.data(), K * sizeof(float), cudaMemcpyHostToDevice);

  const int THREADS = 1024;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream1);
  const int shared_memory = K * sizeof(float) * 2;

  for (int iter = 0; iter < NoOfIterations; ++iter) {
      cudaMemsetAsync(d_counts, 0, K * sizeof(int), stream1);
      cudaMemsetAsync(d_sums_x, 0.0f, K * sizeof(float), stream1);
      cudaMemsetAsync(d_sums_y, 0.0f, K * sizeof(float), stream1);
      assign_clusters<<<BLOCKS, THREADS, shared_memory, stream1>>>(d_cluster, d_x, d_y, N, d_means_x, d_means_y, d_sums_x, d_sums_y, K, d_counts);
      cudaStreamSynchronize(stream1);
      compute_new_means<<<1, K, 0, stream1>>>(d_means_x, d_means_y, d_sums_x, d_sums_y, d_counts);
      cudaStreamSynchronize(stream1);
  }
  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);
  float msecs;
  cudaEventElapsedTime(&msecs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream1);
  cudaMemcpy(h_cluster, d_cluster, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_means_x);
  cudaFree(d_means_y);
  cudaFree(d_sums_x);
  cudaFree(d_sums_y);
  cudaFree(d_counts);
  cudaFree(d_cluster);
  printf("%.9f\n", msecs);
  return ::move(make_pair(msecs, vector < int > (h_cluster, h_cluster + N)));
}

#endif //GPU_GPU_CUH
