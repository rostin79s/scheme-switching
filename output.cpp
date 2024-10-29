#include "src/backend/fhe_operations.hpp"
#include "src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(CKKS_scheme& v1, FHEdouble v2, FHEdouble v3) {
  size_t v4 = 10;
  size_t v5 = 0;
  size_t v6 = 1;
  double v7 = (double)1.00000000000000000e+00;
  FHEdouble v8 = FHEencrypt(v1, v7);
  double v9 = (double)0.0e+00;
  FHEdouble v10 = FHEencrypt(v1, v9);
  FHEdouble v11;
  FHEdouble v12;
  FHEdouble v13;
  FHEdouble v14 = v10;
  FHEdouble v15 = v3;
  FHEdouble v16 = v2;
  for (size_t v17 = v5; v17 < v4; v17 += v6) {
    FHEdouble v18 = FHEaddf(v1, v16, v15);
    FHEdouble v19 = FHEeq(v1, v18, v15);
    FHEdouble v20 = FHEselect(v1, v19, v8, v14);
    FHEdouble v21 = FHEaddf(v1, v15, v20);
    v14 = v20;
    v15 = v21;
    v16 = v18;
  }
  v11 = v14;
  v12 = v15;
  v13 = v16;
  return v12;
}


