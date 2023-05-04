#include "benchmark.h"
#include "utils.h"

using namespace std;

int main () {
//  stress_test(100);
  const string base = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/Clustering-algorithms-on-GPU/CPU/dataset generation/";
  const string base2 = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/Clustering-algorithms-on-GPU/CPU/Plotting/";
  for (int i = 0; i <= 8; ++i) {
    string f = base + string(1, i + '0') + ".csv";
    auto data = read_csv(f);
    auto Rtree = build_r_tree(data);
//    auto [a, label] = DBSCAN(0.0375, 4, Rtree, data);
//    auto [t, label] = run_K_Means(4, 1000, data);
    auto [tt, label] = run_Mean_Shift(10, data);
    auto o = base2 + "meanshift-" + string(1, i + '0') + ".csv";
    write_csv(data, label, o);
  }
  return 0;
}