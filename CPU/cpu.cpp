//
// Created by deepam on 10/4/23.
//

#include "cpu.h"

#include <limits>
#include <random>
#include <chrono>
#include <array>
#include <cstdio>
#include "omp.h"

using namespace std;

vector <point> cluster_cpu::K_Means(int K, int NoOfIterations) {
  auto start = std::chrono::steady_clock::now();
  static mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<size_t> indices(0, N - 1);
  array < vector < point >, 2 > means({vector< point > (K), vector < point > (K)});
  for (auto &cluster : means[0]) {
    cluster = data[indices(rng)];
  }
  vector < int > cluster(N);
  vector < int > counts(K);
  for (int iter = 0; iter < NoOfIterations; ++iter) {
    for (int i = 0; i < N; ++i) {
      float nearest_distance = numeric_limits < float >::max();
      int cluster_id = 0;
      for (int k = 0; k < K; ++k) {
        float l2 = norm(data[i] - means[iter & 1][k]);
        if (l2 < nearest_distance) {
          nearest_distance = l2;
          cluster_id = k;
        }
      }
      cluster[i] = cluster_id;
    }
    fill(counts.begin(), counts.end(), 0);
    for (int i = 0; i < N; ++i) {
      int cluster_id = cluster[i];
      means[(iter & 1) ^ 1][cluster_id] += data[i];
      ++counts[cluster_id];
    }
    for (int k = 0; k < K; ++k) {
      auto c = static_cast<float>(max(1, counts[k]));
      means[(iter & 1) ^ 1][k] /= point(c, c);
    }
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("%.9lf\n", elapsed_seconds.count());
  return means[(NoOfIterations - 1) & 1];
}

vector < point > cluster_cpu::K_Means_OpenMP (int K, int NoOfIterations) {
  auto itime = omp_get_wtime();
  static mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<size_t> indices(0, N - 1);
  array < vector < point >, 2 > means({vector< point > (K), vector < point > (K)});
  for (auto &cluster : means[0]) {
    cluster = data[indices(rng)];
  }
  vector < int > cluster(N);
  vector < int > counts(K);
  for (int iter = 0; iter < NoOfIterations; ++iter) {
#pragma omp parallel for default(none) shared(means, iter, K, cluster, data)
    for (int i = 0; i < N; ++i) {
      float nearest_distance = numeric_limits < float >::max();
      int cluster_id = 0;
      for (int k = 0; k < K; ++k) {
        float l2 = norm(data[i] - means[iter & 1][k]);
          if (l2 < nearest_distance) {
            nearest_distance = l2;
            cluster_id = k;
          }
      }
      cluster[i] = cluster_id;
    }
#pragma omp parallel for default(none) shared(counts, K)
    for (int k = 0; k < K; ++k) {
      counts[k] = 0;
    }
    for (int i = 0; i < N; ++i) {
      int cluster_id = cluster[i];
      means[(iter & 1) ^ 1][cluster_id] += data[i];
      ++counts[cluster_id];
    }
#pragma omp parallel for default(none) shared(counts, means, iter, K)
    for (int k = 0; k < K; ++k) {
      auto c = static_cast<float>(max(1, counts[k]));
      means[(iter & 1) ^ 1][k] /= point(c, c);
    }
  }
  auto otime = omp_get_wtime();
  printf("%.9lf\n", otime - itime);
  return means[(NoOfIterations - 1) & 1];
}
