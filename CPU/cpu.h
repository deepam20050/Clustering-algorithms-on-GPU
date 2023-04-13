//
// Created by deepam on 10/4/23.
//

#ifndef CPU_CPU_H
#define CPU_CPU_H

#include <vector>
#include <complex>
#include <iostream>
#include <cassert>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

using namespace std;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using Point = bg::model::point<float, 3, bg::cs::cartesian>;
using Value = std::pair<Point, int>;
using RTree = bgi::rtree<Value, bgi::quadratic<16>>;
using point = complex < float >;

class cluster_cpu {
private:
    vector < point > data;
    RTree rtree;
    int N;
    const int NOISE = 1;
public:
    explicit cluster_cpu(const vector < point >& points, int n) {
      data = points;
      N = n;
      assert(N > 0);
    }

    vector<point> K_Means(int K, int NoOfIterations);

    vector<point> K_Means_OpenMP(int K, int NoOfIterations);

    void build_r_tree();

    queue<int> spatial_query(float x, float y, float eps);

    vector<int> DBSCAN(float eps, int minPts);
};

#endif //CPU_CPU_H
