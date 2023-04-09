//
// Created by deepam on 9/4/23.
//

#include "cpu.cuh"
#include <limits>
#include <random>
#include <chrono>

using namespace std;

vector <point> K_Means::cpu(int K, int NoOfIterations) {
  static mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
  std::uniform_int_distribution<size_t> indices(0, N - 1);
  vector < point > means(K);
  for (auto &cluster : means) {
    cluster = data[indices(rng)];
  }
  vector < int > cluster(N);
  for (int iter = 0; iter < NoOfIterations; ++iter) {
    for (int i = 0; i < N; ++i) {
      float nearest_distance = numeric_limits < float >::max();
      int cluster_id = 0;
      for (int k = 0; k < K; ++k) {
        float l2 = norm(data[i] - means[k]);
        if (l2 < nearest_distance) {
          nearest_distance = l2;
          cluster_id = k;
        }
      }
      cluster[i] = cluster_id;
    }
    vector < point > new_means(K);
    vector < int > counts(K);
    for (int i = 0; i < N; ++i) {
      int cluster_id = cluster[i];
      new_means[cluster_id] += data[i];
      ++counts[cluster_id];
    }
    for (int k = 0; k < K; ++k) {
      auto c = static_cast<float>(max(1, counts[k]));
      means[k] = new_means[k] / point(c, c);
    }
  }
  return means;
}