#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
#include "test.hpp"

using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* v1, FHEdouble v2, FHEdouble v3) {
  FHEdouble v4 = FHEeq(v1, v2, v3);
  FHEdouble v5 = FHEselect(v1, v4, v2, v3);
  return v5;
}


