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

__global__ void assign_clusters(int * cluster, const float*  data_x, const float*  data_y, int N, const float*  means_x, const float*  means_y, float*  new_sums_x, float*  new_sums_y, int K, int*  counts) {
  const int index = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= N) return;

  const float x = data_x[index];
  const float y = data_y[index];

  float nearest_distance = FLT_MAX;
  int cluster_id = 0;
  for (int k = 0; k < K; ++k) {
    const float l2_norm_sqq = l2_norm_sq(x, y, means_x[k], means_y[k]);
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

__global__ void compute_new_means(float*  means_x, float*  means_y, const float*  new_sum_x, const float*  new_sum_y, const int*  counts) {
  const int cluster = static_cast < int > (threadIdx.x);
  // Threshold count to turn 0/0 into 0/1.
  const auto count = static_cast < float > (max(1, counts[cluster]));
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;
}

pair < float, vector < int > > run_K_Means (vector < float > &h_x, vector < float > &h_y, int K, int NoOfIterations) {
  float *d_x, *d_y, *d_means_x, *d_means_y, *d_sums_x, *d_sums_y;
  int *d_counts, *d_cluster;

  // Allocate memory on GPU
  const int N = static_cast < int > (h_x.size());
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
  cudaMemset(d_counts, 0, K * sizeof(int));

  mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
  shuffle(h_x.begin(), h_x.end(), rng);
  shuffle(h_y.begin(), h_y.end(), rng);

  cudaMemcpy(d_means_x, h_x.data(), K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_means_y, h_y.data(), K * sizeof(float), cudaMemcpyHostToDevice);

  const int THREADS = 1024;
  const int BLOCKS = (N + THREADS - 1) / THREADS;
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);
  bool graphCreated = false;
  cudaGraph_t graph;
  cudaGraphExec_t instance;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream1);

  for (int iter = 0; iter < NoOfIterations; ++iter) {
    if (!graphCreated) {
      cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
      cudaMemset(d_counts, 0, K * sizeof(int));
      cudaMemset(d_sums_x, 0.0f, K * sizeof(float));
      cudaMemset(d_sums_y, 0.0f, K * sizeof(float));

      assign_clusters<<<BLOCKS, THREADS, 0, stream1>>>(d_cluster, d_x, d_y, N, d_means_x, d_means_y, d_sums_x, d_sums_y, K, d_counts);
      cudaDeviceSynchronize();
      compute_new_means<<<1, K, 0, stream1>>>(d_means_x, d_means_y, d_sums_x, d_sums_y, d_counts);
      cudaDeviceSynchronize();

      cudaStreamEndCapture(stream1, &graph);
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
      graphCreated=true;
    }
    cudaGraphLaunch(instance, stream1);
    cudaStreamSynchronize(stream1);
  }
  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);
  float msecs;
  cudaEventElapsedTime(&msecs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream1);

  int *h_cluster = (int *) malloc(N * sizeof(int));
  cudaMemcpy(h_cluster, d_cluster, N * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_means_x);
  cudaFree(d_means_y);
  cudaFree(d_sums_x);
  cudaFree(d_sums_y);
  cudaFree(d_counts);
  cudaFree(d_cluster);
//  for (int i = 0; i < 5; ++i) {
//    cout << h_cluster[i] << " ";
//  }
  auto r = make_pair(msecs, vector < int > (h_cluster, h_cluster + N));
  free(h_cluster);
  return ::move(r);
}

#endif //GPU_GPU_CUH
