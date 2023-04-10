//
// Created by deepam on 10/4/23.
//

#ifndef CPU_CPU_H
#define CPU_CPU_H

#include <vector>
#include <complex>
#include <iostream>
#include <cassert>

using namespace std;

using point = complex < float >;

class cluster_cpu {
private:
    vector < point > data;
    int N;
public:
    explicit cluster_cpu(const vector < point >& points, int n) {
      data = points;
      N = n;
      assert(N > 0);
    }

    vector<point> K_Means(int K, int NoOfIterations);

    vector<point> K_Means_OpenMP(int K, int NoOfIterations);
};

#endif //CPU_CPU_H
