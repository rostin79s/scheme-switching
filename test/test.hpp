#ifndef TEST_HPP
#define TEST_HPP

#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>
using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* v1, FHEdouble v2, FHEdouble v3, FHEdouble v4);

#endif // TEST_HPP