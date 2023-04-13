//
// Created by deepam on 10/4/23.
//

#ifndef CPU_CSV_PARSER_H
#define CPU_CSV_PARSER_H

#include <complex>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cassert>

using namespace std;

using point = complex < float >;

vector < point > csv_parser (const string& filename) {
  vector < point > data;
  ifstream file(filename);
  string line;
  while (getline(file, line)) {
    vector <float> values;
    stringstream ss(line);
    string value;
    float x, y;
    for (int i = 0; i < 2 && getline(ss, value, ';'); ++i) {
      float f = stof(value);
      i == 0 ? x = f : y = f;
    }
    data.emplace_back(x, y);
  }
  return data;
}

#endif //CPU_CSV_PARSER_H
