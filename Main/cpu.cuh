//
// Created by deepam on 9/4/23.
//

#ifndef MAIN_CPU_CUH
#define MAIN_CPU_CUH

#include <vector>
#include <complex>
#include <iostream>
#include <cassert>

using namespace std;

using point = complex < float >;

class K_Means {
private:
    vector < point > data;
    int N;
public:
    explicit K_Means (const vector < point >& points, int n) {
      data = points;
      N = n;
      assert(N > 0);
    }

    vector<point> cpu(int K, int NoOfIterations);

};

#endif //MAIN_CPU_CUH
