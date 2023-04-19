#include "benchmark.cuh"
#include "utils.cuh"

using namespace std;

int main () {
  const string base = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/Clustering-algorithms-on-GPU/CPU/dataset generation/";
  const string base2 = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/Clustering-algorithms-on-GPU/GPU/Plotting/";
  for (int i = 0; i <= 8; ++i) {
    string f = base + string(1, i + '0') + ".csv";
    auto [h_x, h_y] = read_csv(f);
    auto [t, label] = run_K_Means(h_x, h_y, 4, 1000);
    cout << *max(label.begin(), label.end()) << '\n';
    auto o = base2 + "kmeans-" + string(1, i + '0') + ".csv";
    write_csv(h_x, h_y, label, o);
  }
  return 0;
}
