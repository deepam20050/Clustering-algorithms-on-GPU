//
// Created by deepam on 11/4/23.
//

#ifndef CPU_TEST_H
#define CPU_TEST_H

#include <iostream>
#include <vector>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;

typedef bg::model::point<float, 3, bg::cs::cartesian> Point;
typedef std::pair<Point, int> Value;
typedef bgi::rtree<Value, bgi::quadratic<16>> RTree;

void f() {
  // Define the R-Tree and add some points to it
  RTree rtree;
  rtree.insert(std::make_pair(Point(0, 0, 0), 0));
  rtree.insert(std::make_pair(Point(1, 1, 1), 1));
  rtree.insert(std::make_pair(Point(2, 2, 2), 2));
  rtree.insert(std::make_pair(Point(3, 3, 3), 3));

  // Define the search point and the search radius
  Point searchPoint(1.5f, 1.5f, 1.5f);
  float eps = 1.0f;

  // Define the search box
  Point searchMin(searchPoint.get<0>() - eps, searchPoint.get<1>() - eps, searchPoint.get<2>() - eps);
  Point searchMax(searchPoint.get<0>() + eps, searchPoint.get<1>() + eps, searchPoint.get<2>() + eps);
  bg::model::box<Point> searchBox(searchMin, searchMax);

  // Perform the range query on the R-Tree
  std::vector<Value> results;
  rtree.query(bgi::intersects(searchBox), std::back_inserter(results));

  // Print the results
  std::cout << "Neighbors within distance " << eps << " of " << bg::wkt(searchPoint) << ":" << std::endl;
  for (auto result : results) {
    std::cout << "  " << bg::wkt(result.first) << " (id = " << result.second << ")" << std::endl;
  }
}

#endif //CPU_TEST_H
