#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
#include "test.hpp"

using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* v1, FHEdouble v2, FHEdouble v3) {
  double v4 = (double)2.00000000000000000e+02;
  FHEdouble v5 = FHEencrypt(v1, v4);
  double v6 = (double)1.00000000000000000e+02;
  FHEdouble v7 = FHEencrypt(v1, v6);
  FHEdouble v8 = FHEeq(v1, v2, v3);
  FHEdouble v9 = FHEselect(v1, v8, v7, v5);
  return v9;
}


