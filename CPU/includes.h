//
// Created by deepam on 18/4/23.
//

#ifndef CPU_INCLUDES_H
#define CPU_INCLUDES_H

#include <vector>
#include <iostream>
#include <complex>
#include <string>
#include <algorithm>
#include <fstream>
#include <parallel/algorithm>
#include <cassert>
#include <random>
#include <utility>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <omp.h>

using namespace std;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

using Point = bg::model::point<float, 2, bg::cs::cartesian>;
using point = complex < float >;
using points = vector < point > ;
using Value = std::pair<Point, int>;
using RTree = bgi::rtree<Value, bgi::rstar<16>>;

const int NOISE = 1;

#endif //CPU_INCLUDES_H
