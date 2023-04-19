//
// Created by deepam on 10/4/23.
//

#ifndef CPU_UTILS_H
#define CPU_UTILS_H

#include "includes.h"

using namespace std;

vector < point > read_csv (const string& filename) {
  vector < point > data;
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
    data.emplace_back(x, y);
  }
  return ::move(data);
}

void write_csv (const points &data, const vector < int > &label, const string &filename) {
  ofstream csv(filename);
  csv << "x,y,label\n";
  const int N = static_cast < int > (data.size());
  for (int i = 0; i < N; ++i) {
    csv << data[i].real() << "," << data[i].imag() << "," << label[i] << '\n';
  }
  csv.close();
}

#endif //CPU_UTILS_H
