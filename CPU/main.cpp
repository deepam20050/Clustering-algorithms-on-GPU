#include <iostream>
#include <vector>
#include "cpu.h"
#include "csv_parser.h"

using namespace std;

int main() {
  const string filename = "/home/deepam/Desktop/SEM 6/CSE560 - GPU Computing/Clustering-algorithms-on-GPU/CPU/datasets/dataset.csv";
  auto data = csv_parser(filename);
  const int N = static_cast<int>(data.size());
  auto model = cluster_cpu(data, N);
  const int K = 20, NoOfIterations = 100;
  auto normal= model.K_Means(K, NoOfIterations);
  auto openmp = model.K_Means_OpenMP(K, NoOfIterations);
  return 0;
}
