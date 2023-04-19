//
// Created by deepam on 18/4/23.
//

#ifndef GPU_BENCHMARK_CUH
#define GPU_BENCHMARK_CUH

#include "gpu.cuh"

using namespace std;

void stress_test (int K, int NoOfIterations, int M) {
  const int Ns[] = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000};
  printf("Computing average time from %d different samples...\n", M);
  for (const int N : Ns) {
    vector<float> h_x(N);
    vector<float> h_y(N);
    printf("Running on %d randomly generated data-points...\n", N);

    mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution < float > dis(0.0f, 1e9);
    float avg = 0.0f;
    for (int iter = 0; iter < M; ++iter) {
      generate(h_x.begin(), h_x.end(), [&dis, &gen]() {
          return dis(gen);
      });
      generate(h_y.begin(), h_y.end(), [&dis, &gen]() {
          return dis(gen);
      });
      auto [msecs, v] = run_K_Means(h_x, h_y, K, NoOfIterations);
      avg += msecs;
    }
    avg /= static_cast<float>(M);
    printf("N = %d | Time (in ms) = %.9f\n", N, avg);
  }
}

#endif //GPU_BENCHMARK_CUH
