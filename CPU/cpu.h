//
// Created by deepam on 10/4/23.
//

#ifndef CPU_CPU_H
#define CPU_CPU_H

#include "includes.h"

using namespace std;

pair < double, vector < int > > run_K_Means (int K, int NoOfIterations, const points &data) {
  double start_t = omp_get_wtime();
  mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
  const int N = static_cast < int > (data.size());
  uniform_int_distribution<int> indices(0, N - 1);
  array < vector < point >, 2 > means({vector< point > (K), vector < point > (K)});
  generate(means[0].begin(), means[0].end(), [&indices, &rng, &data] () {
      return data[indices(rng)];
  });
  vector < int > cluster(N);
  vector < int > counts(K);
  for (int iter = 0; iter < NoOfIterations; ++iter) {
#pragma omp parallel for default(none) shared(means, iter, K, cluster, data, N)
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
#pragma omp parallel for default(none) shared(counts, K, means, iter)
    for (int k = 0; k < K; ++k) {
      means[(iter & 1) ^ 1][k] = {0.0f, 0.0f};
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
      means[(iter & 1) ^ 1][k] /= point(c, 0.0f);
    }
  }
#pragma omp parallel for default(none) shared(means, K, cluster, data, NoOfIterations, N)
  for (int i = 0; i < N; ++i) {
    float nearest_distance = numeric_limits < float >::max();
    int cluster_id = 0;
    int idx = (NoOfIterations - 1) & 1;
    for (int k = 0; k < K; ++k) {
      float l2 = norm(data[i] - means[idx][k]);
      if (l2 < nearest_distance) {
        nearest_distance = l2;
        cluster_id = k;
      }
    }
    cluster[i] = cluster_id;
  }
  auto end_t = omp_get_wtime();
  return make_pair((end_t - start_t) * 1000.0, ::move(cluster));
}

RTree build_r_tree(const points &data) {
  const int N = static_cast<int>(data.size());
  vector < Value > values(N);
#pragma omp parallel for default(none) shared(values, data, N)
  for (int i = 0; i < N; ++i) {
    values[i] = {{data[i].real(), data[i].imag()}, i};
  }
  RTree rtree(values);
  return boost::move(rtree);
}

queue < int > spatial_query (float x, float y, float eps, const RTree &rtree) {
  Point center(x, y);
  Point searchMin(x - eps, y - eps);
  Point searchMax(x + eps, y + eps);
  bg::model::box<Point> searchBox(searchMin, searchMax);
  vector < Value > results;
  rtree.query(bgi::intersects(searchBox), back_inserter(results));
  queue < int > indices;
  for (auto & result : results) {
    auto p = result.first;
    if (bg::distance(center, p) <= eps) {
      indices.emplace(result.second);
    }
  }
  return ::move(indices);
}

pair < double, vector < int > > DBSCAN(float eps, int minPts, const RTree &rtree, const points &data) {
  double start_t = omp_get_wtime();
  const int N = static_cast < int > (data.size());
  int cluster_cnt = 1;
  vector < int > label(N, -1);
  for (int i = 0; i < N; ++i) {
    if (label[i] != -1) continue;
    auto neighbours = spatial_query(data[i].real(), data[i].imag(), eps, rtree);
    if (static_cast<int>(neighbours.size()) < minPts) {
      label[i] = NOISE;
      continue;
    }
    ++cluster_cnt;
    label[i] = cluster_cnt;
    while (!neighbours.empty()) {
      int u = neighbours.front();
      neighbours.pop();
      if (label[u] == NOISE) {
        label[u] = cluster_cnt;
        continue;
      } else if (label[u] != -1) {
        continue;
      }
      label[u] = cluster_cnt;
      auto new_neighbours = spatial_query(data[u].real(), data[u].imag(), eps, rtree);
      if (static_cast<int>(new_neighbours.size()) >= minPts) {
        while (!new_neighbours.empty()) {
          int x = new_neighbours.front();
          new_neighbours.pop();
          neighbours.emplace(x);
        }
      }
    }
  }
  auto end_t = omp_get_wtime();
  return make_pair((end_t - start_t) * 1000.0, ::move(label));
}

#endif //CPU_CPU_H
