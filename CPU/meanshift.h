//
// Created by deepam on 4/5/23.
//

#ifndef CPU_MEANSHIFT_H
#define CPU_MEANSHIFT_H

#include "includes.h"

/* Implemented Mean Shift Clustering with
 * [1] Gaussian Kernel
 */

//using namespace std;

float gaussian_kernel (const point &a, const point &b) {
  return expf(-0.5f * norm(a - b));
}

pair < double, vector < int > > run_Mean_Shift (const int &NoOfIterations, const points &data) {
  const int N = static_cast<int>(data.size());
  points a[2] = {data, points(N)};
  for (int iter = 1; iter <= NoOfIterations; ++iter) {
    int idx = iter & 1;
    int prev = idx ^ 1;
#pragma omp parallel for default(none) shared(prev, iter, a, idx, data, N)
    for (int i = 0; i < N; ++i) {
      float net_weight = 0.0f;
      point mean = {0.0f, 0.0f};
//#pragma omp parallel for default(none) shared(prev, iter, prev, a, idx, data, N)
      for (int j = 0; j < N; ++j) {
        auto w = gaussian_kernel(a[prev][i], a[prev][j]);
        net_weight += w;
        mean += point(w, 0) * a[prev][j];
      }
      mean /= point(net_weight, 0);
      a[idx][i] = mean;
    }
  }
  int idx = NoOfIterations & 1;
  vector < pair < float, float > > means_ori(N);
  for (int i = 0; i < N; ++i) {
    means_ori[i] = {a[idx][i].real(), a[idx][i].imag()};
  }
  auto means = means_ori;
  __gnu_parallel::sort(means.begin(), means.end());
  means.resize(unique(means.begin(), means.end()) - means.begin());
  vector < int > labels(N);
#pragma omp parallel for default(none) shared(means, labels, means_ori, N)
  for (int i = 0; i < N; ++i) {
    labels[i] = static_cast < int > (lower_bound(means.begin(), means.end(), means_ori[i]) - means.begin());
  }
  float msecs = 0.0f;
  return {msecs, labels};
//  return {0.0f, vector < int > (N)};
}

#endif //CPU_MEANSHIFT_H
