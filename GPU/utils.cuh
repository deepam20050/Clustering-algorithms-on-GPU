//
// Created by deepam on 19/4/23.
//

#ifndef GPU_UTILS_CUH
#define GPU_UTILS_CUH

#include "includes.cuh"

using namespace std;

pair < vector < float >, vector < float > > read_csv (const string& filename) {
  vector < float > h_x, h_y;
  ifstream file(filename);
  string line;
  while (getline(file, line)) {
    vector <float> values;
    stringstream ss(line);
    string value;
    float x, y;
    for (int i = 0; i < 2 && getline(ss, value, ','); ++i) {
      float f = stof(value);
      i == 0 ? x = f : y = f;
    }
    h_x.emplace_back(x);
    h_y.emplace_back(y);
  }
  return ::move(make_pair(h_x, h_y));
}

void write_csv (const vector < float > &h_x, const vector < float > &h_y, const vector < int > &label, const string &filename) {
  ofstream csv(filename);
  csv << "x,y,label\n";
  const int N = static_cast < int > (label.size());
  for (int i = 0; i < N; ++i) {
    csv << h_x[i] << "," << h_y[i] << "," << label[i] << '\n';
  }
  csv.close();
}


#endif //GPU_UTILS_CUH
