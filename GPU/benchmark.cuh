//
// Created by deepam on 18/4/23.
//

#ifndef GPU_BENCHMARK_CUH
#define GPU_BENCHMARK_CUH

#include "kmeans.cuh"
#include "dbscan.cuh"

using namespace std;

void block_test () {
  const int N = 100'000;
  vector < float > h_x(N), h_y(N);
  mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
  uniform_real_distribution < float > dis(0.0f, 1e9);
  generate(h_x.begin(), h_x.end(), [&dis, &gen]() {
    return dis(gen);
  });
  generate(h_y.begin(), h_y.end(), [&dis, &gen]() {
    return dis(gen);
  });
  int type;
  scanf("%d", &type);
  double msecs = 0.0;
  for (int i = 32; i <= 1024; i += i) {
    double msecs = 0.0;
    if (!type) {
      auto [a, b] = run_K_Means(h_x, h_y, 4, 50, i);
      msecs += a;
    } else if (type == 1) {
      auto [a, b] = run_DBSCAN(h_x, h_y, 0.0375, 4);
      msecs += a;
    } else if (type == 2) {
      auto [a, b] = run_Mean_Shift(h_x, h_y, 50);
      msecs += a;
    }
    printf("%.9f\n", msecs);
  }  
}

void stress_test (int M) {
  const int Ns[] = {10, 100, 500, 1000, 5000, 10'000, 50'000, 100'000, 500'000, 1'000'000, 5'000'000, 10'000'000};
  printf("Computing average time from %d different samples...\n", M);
  printf("Clustering Algorithm : ");
  int type;
  scanf("%d", &type);
  // K-Means
  int K = 0, NoOfIterations = 20, minPts = 0;
  float eps = 0.0f;
  if (type == 0) {
    puts("K-Means...\n");
    printf("Enter K and No. of iterations\n");
    scanf("%d %d", &K, &NoOfIterations);
  } else if (type == 1) {
    puts("DBSCAN...\n");
    printf("Enter eps and minPts\n");
    scanf("%f %d", &eps, &minPts);
  } else {
    puts("Mean-Shift...\n");
    printf("Enter No. Of iterations\n");
    scanf("%d", &NoOfIterations);
  }
  for (const int N : Ns) {
    vector<float> h_x(N);
    vector<float> h_y(N);
  
    mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution < float > dis(0.0f, 1e9);
    double avg = 0.0;

    for (int iter = 0; iter < M; ++iter) {
      generate(h_x.begin(), h_x.end(), [&dis, &gen]() {
          return dis(gen);
      });
      generate(h_y.begin(), h_y.end(), [&dis, &gen]() {
          return dis(gen);
      });
      double msecs = 0.0;
      if (!type) {
        auto [a, b] = run_K_Means(h_x, h_y, K, NoOfIterations);
        msecs += a;
      } else if (type == 1) {
        auto [a, b] = run_DBSCAN(h_x, h_y, eps, minPts);
        msecs += a;
      } else if (type == 2) {
        auto [a, b] = run_Mean_Shift(h_x, h_y, NoOfIterations);
        msecs += a;
      }
      avg += msecs;
    }
    avg /= static_cast<float>(M);
    printf("%.9lf\n", avg);
  }
}

#endif //GPU_BENCHMARK_CUH
