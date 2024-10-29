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
  FHEdouble v7 = 1;
  FHEdouble v8 = 0;
  FHEdouble v9;
  FHEdouble v10;
  FHEdouble v11;
  FHEdouble v12 = v8;
  FHEdouble v13 = v3;
  FHEdouble v14 = v2;
  for (size_t v15 = v5; v15 < v4; v15 += v6) {
    FHEdouble v16 = FHEaddf(v1, v14, v13);
    FHEdouble v17 = FHEoeqf(v1, v16, v13);
    FHEdouble v18 = FHEselectf(v1, v17, v7, v12);
    FHEdouble v19 = FHEaddf(v1, v13, v18);
    v12 = v18;
    v13 = v19;
    v14 = v16;
  }
  v9 = v12;
  v10 = v13;
  v11 = v14;
  return v10;
}


