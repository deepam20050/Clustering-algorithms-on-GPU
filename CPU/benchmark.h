//
// Created by deepam on 18/4/23.
//

#ifndef CPU_BENCHMARK_H
#define CPU_BENCHMARK_H

#include "includes.h"
#include "cpu.h"

using namespace std;

void stress_test (int M) {
  const int Ns[] = {10, 100, 1000, 10'000, 100'000, 1'000'000, 10'000'000};
  printf("Computing average time from %d different samples...\n", M);
  printf("Clustering Algorithm : ");
  int type;
  scanf("%d", &type);
  // K-Means
  int K = 0, NoOfIterations = 0, minPts = 0;
  float eps = 0.0f;
  if (type == 0) {
    puts("K-Means...\n");
    printf("Enter K and No. of iterations\n");
    scanf("%d %d", &K, &NoOfIterations);
  } else {
    printf("Enter eps and minPts\n");
    scanf("%f %d", &eps, &minPts);
  }
  for (const int N : Ns) {
    vector < point > data(N);
    printf("Running on %d randomly generated data-points...\n", N);

    mt19937 gen(chrono::steady_clock::now().time_since_epoch().count());
    uniform_real_distribution < float > dis(0.0f, 1e9);
    double avg = 0.0;

    for (int iter = 0; iter < M; ++iter) {
      generate(data.begin(), data.end(), [&dis, &gen]() {
          return point(dis(gen), dis(gen));
      });
      double msecs = 0.0;
      if (!type) {
        auto [a, b] = run_K_Means(K, NoOfIterations, data);
        msecs += a;
      } else {
        auto Rtree = build_r_tree(data);
        auto [a, b] = DBSCAN(eps, minPts, Rtree, data);
        msecs += a;
      }
      avg += msecs;
    }
    avg /= static_cast<float>(M);
    printf("N = %d | Time (in ms) = %.9lf\n", N, avg);
  }
}


#endif //CPU_BENCHMARK_H
