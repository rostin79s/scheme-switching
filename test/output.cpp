#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
#include "test.hpp"

using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* v1, FHEdouble v2, FHEdouble v3, FHEdouble v4) {
  double v5 = (double)1.00000000000000000e+00;
  FHEdouble v6 = FHEencrypt(v1, v5);
  double v7 = (double)0.0e+00;
  FHEdouble v8 = FHEencrypt(v1, v7);
  std::vector<double> v9 = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  FHEdouble v10 = FHEencrypt(v1, v9);
  size_t v11 = 0;
  size_t v12 = 20;
  size_t v13 = 20;
  FHEdouble v14;
  FHEdouble v15 = v10;
  for (size_t v16 = v11; v16 < v12; v16 += v13) {
    std::vector<double> v17 = std::vector<double>{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    FHEdouble v18 = FHEencrypt(v1, v17);
    std::vector<double> v19 = std::vector<double>{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    FHEdouble v20 = FHEencrypt(v1, v19);
    double v21 = (double)0.0e+00;
    FHEdouble v22 = FHEencrypt(v1, v21);
    double v23 = (double)0.0e+00;
    FHEdouble v24 = FHEencrypt(v1, v23);
    FHEdouble v25 = FHEsubf(v1, v2, v3);
    FHEdouble v26 = FHEmulf(v1, v25, v4);
    FHEdouble v27 = FHEeq(v1, v26, v4);
    FHEdouble v28 = FHEselect(v1, v27, v20, v18);
    FHEdouble v29 = FHEaddf(v1, v15, v28);
    v15 = v29;
  }
  v14 = v15;
  FHEdouble v30 = FHEvectorSum(v1, v14);
  return v30;
}


