#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
#include "test.hpp"

using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* v1, FHEdouble v2, FHEdouble v3) {
  double v4 = (double)0.0e+00;
  FHEdouble v5 = FHEencrypt(v1, v4);
  FHEdouble v6 = FHEaddf(v1, v2, v3);
  size_t v7 = 0;
  size_t v8 = 38;
  size_t v9 = 1;
  FHEdouble v10;
  FHEdouble v11 = v5;
  for (size_t v12 = v7; v12 < v8; v12 += v9) {
    FHEdouble v13 = FHEaddf(v1, v11, v6);
    v11 = v13;
  }
  v10 = v11;
  return v10;
}


