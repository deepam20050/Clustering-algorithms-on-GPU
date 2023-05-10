//
// Created by deepam on 19/4/23.
//

#ifndef GPU_KMEANS_CUH
#define GPU_KMEANS_CUH

#include "includes.cuh"

using namespace std;

__global__ void compute_new_means(float*  means_x, float*  means_y, const float*  new_sum_x, const float*  new_sum_y, const int*  counts) {
  const int cluster = static_cast < int > (threadIdx.x);
  const auto count = static_cast < float > (max(1, counts[cluster]));
  means_x[cluster] = new_sum_x[cluster] / count;
  means_y[cluster] = new_sum_y[cluster] / count;
}

__global__ void assign_clusters_shared(int * cluster, const float*  data_x, const float*  data_y, int N, float*  means_x, float*  means_y, float*  new_sums_x, float*  new_sums_y, int K, int*  counts) {
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
  atomicAdd(&new_sums_x[cluster_id], x);
  atomicAdd(&new_sums_y[cluster_id], y);
  atomicAdd(&counts[cluster_id], 1);
}

pair < float, vector < int > > run_K_Means (vector < float > &h_x, vector < float > &h_y, int K, int NoOfIterations, int THREADS = 1024) {
  const int N = static_cast < int > (h_x.size());
  float *d_x, *d_y, *d_means_x, *d_means_y, *d_sums_x, *d_sums_y;
  int *d_counts, *d_cluster, *h_cluster = new int[N];

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

  // const int THREADS = 1024;
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
      assign_clusters_shared<<<BLOCKS, THREADS, shared_memory, stream1>>>(d_cluster, d_x, d_y, N, d_means_x, d_means_y, d_sums_x, d_sums_y, K, d_counts);
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
  auto ret = make_pair(msecs, vector < int > (h_cluster, h_cluster + N));
  delete[] h_cluster;
  return ::move(ret);
}

__global__ void mean_shift (int N, float *d_x, float *d_y, int row) {
  const int i = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= N) return;
  const float x0 = d_x[row * N + i], y0 = d_y[row * N + i];
  float net_weight = 0.0f;
  float mean_x = 0.0f, mean_y = 0.0f;
  for (int j = 0; j < N; ++j) {
    const float x = d_x[row * N + j], y = d_y[row * N + j];
    float w = gaussian_kernel(x0, y0, x, y);
    net_weight += w;
    mean_x += w * x;
    mean_y += w * y;
  }
  mean_x /= net_weight;
  mean_y /= net_weight;
  d_x[(row ^ 1) * N + i] = mean_x;
  d_y[(row ^ 1) * N + i] = mean_y;
}

__global__ void mean_shift_shared_mem (int N, float* d_x, float* d_y, int row) {
	int tx = static_cast < int > (threadIdx.x);
  const int i = static_cast < int > (blockIdx.x * blockDim.x + threadIdx.x);
  float net_weight = 0.0f;
  float mean_x = 0.0f, mean_y = 0.0f;
  __shared__ float tile[THREADS][2];
	for (int tile_i = 0; tile_i < (N + THREADS - 1) / THREADS; ++tile_i) {
		int shift = static_cast < int > (tile_i * THREADS + tx);
		int index = static_cast < int > (row * N + shift * blockDim.x);
		if (index < N) {
			tile[tx][0] = d_x[index];
			tile[tx][1] = d_y[index];
		} else {
			tile[tx][0] = 0.0;
			tile[tx][1] = 0.0;
		}
		__syncthreads();
		if (i < N) {
      const float x0 = d_x[row * N + i], y0 = d_y[row * N + i];
			for (int j = 0; j < THREADS; ++j) {
        const float x = tile[j][0], y = tile[j][1];
        float w = gaussian_kernel(x0, y0, x, y);
        net_weight += w;
        mean_x += w * x;
        mean_y += w * y;
			}
		}
		__syncthreads();
	}
	if (i < N) {
		mean_x /= net_weight;
    mean_y /= net_weight;
    d_x[(row ^ 1) * N + i] = mean_x;
    d_y[(row ^ 1) * N + i] = mean_y;
	}
}

pair < float, vector < int > > run_Mean_Shift (vector < float > &h_x, vector < float > &h_y, int NoOfIterations, int THREADS = 1024) {
  const int N = static_cast < int > (h_x.size());
  float *d_x, *d_y;
  float *h_local_x = new float[2 * N], *h_local_y = new float[2 * N];

  // Creating Streams for Asynchronous execution
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  // Allocate memory on GPU
  cudaMallocAsync(&d_x, N * 2 * sizeof(float), stream1);
  cudaMallocAsync(&d_y, N * 2 * sizeof(float), stream1);
  
  // Copy/Set their values
  cudaMemcpyAsync(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);

  const int BLOCKS = (N + THREADS - 1) / THREADS;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream1);

  // Meanshift for GPU
  for (int iter = 0; iter < NoOfIterations; ++iter) {
    mean_shift<<<BLOCKS, THREADS, 0, stream1>>>(N, d_x, d_y, iter & 1);
  }
  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);
  float msecs;
  cudaEventElapsedTime(&msecs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpyAsync(h_local_x, d_x, N * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_local_y, d_y, N * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream1);

  int idx = NoOfIterations & 1;
  vector < pair < float, float > > means(N);
  for (int i = 0; i < N; ++i) {
    means[i] = {h_local_x[idx * N + i], h_local_y[idx * N + i]};
  }
  sort(means.begin(), means.end());
  means.resize(unique(means.begin(), means.end()) - means.begin());
  vector < int > labels(N);
  for (int i = 0; i < N; ++i) {
    labels[i] = static_cast < int > (lower_bound(means.begin(), means.end(), make_pair(h_local_x[idx * N + i], h_local_y[idx * N + i])) - means.begin());
  }
  cudaStreamDestroy(stream1);
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_local_x;
  delete[] h_local_y;
  return ::move(make_pair(msecs, labels));
}

pair < float, vector < int > > run_Mean_Shift_shared (vector < float > &h_x, vector < float > &h_y, int NoOfIterations, int THREADS = 1024) {
  const int N = static_cast < int > (h_x.size());
  float *d_x, *d_y;
  float *h_local_x = new float[2 * N], *h_local_y = new float[2 * N];

  // Creating Streams for Asynchronous execution
  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  // Allocate memory on GPU
  cudaMallocAsync(&d_x, N * 2 * sizeof(float), stream1);
  cudaMallocAsync(&d_y, N * 2 * sizeof(float), stream1);
  
  // Copy/Set their values
  cudaMemcpyAsync(d_x, h_x.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);
  cudaMemcpyAsync(d_y, h_y.data(), N * sizeof(float), cudaMemcpyHostToDevice, stream1);

  const int BLOCKS = (N + THREADS - 1) / THREADS;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, stream1);

  // Meanshift for GPU
  for (int iter = 0; iter < NoOfIterations; ++iter) {
    mean_shift_shared_mem<<<BLOCKS, THREADS, 0, stream1>>>(N, d_x, d_y, iter & 1);
  }
  cudaEventRecord(stop, stream1);
  cudaEventSynchronize(stop);
  float msecs;
  cudaEventElapsedTime(&msecs, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaMemcpyAsync(h_local_x, d_x, N * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream1);
  cudaMemcpyAsync(h_local_y, d_y, N * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream1);

  int idx = NoOfIterations & 1;
  vector < pair < float, float > > means(N);
  for (int i = 0; i < N; ++i) {
    means[i] = {h_local_x[idx * N + i], h_local_y[idx * N + i]};
  }
  sort(means.begin(), means.end());
  means.resize(unique(means.begin(), means.end()) - means.begin());
  vector < int > labels(N);
  for (int i = 0; i < N; ++i) {
    labels[i] = static_cast < int > (lower_bound(means.begin(), means.end(), make_pair(h_local_x[idx * N + i], h_local_y[idx * N + i])) - means.begin());
  }
  cudaStreamDestroy(stream1);
  cudaFree(d_x);
  cudaFree(d_y);
  delete[] h_local_x;
  delete[] h_local_y;
  return ::move(make_pair(msecs, labels));
}

#endif //GPU_KMEANS_CUH
