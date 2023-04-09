#include <iostream>
#include <vector>
#include "cpu.cuh"

using namespace std;

int main() {
  const int N = 1e5;
  vector < point > data(N);
  for (int i = 0; i < N; ++i) {
    data[i] = point(i, i);
  }
  auto km = K_Means(data, N);
  auto x = km.cpu(2, 10);
  return 0;
}
