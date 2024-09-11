#include "backend/fhe_operations.hpp"
#include "backend/fhe_types.hpp"
#include <vector>
#include <iostream>

using namespace CKKS;
using namespace TFHE;

FHEdouble* _Z6rostinii(CKKS_scheme* ck, FHEdouble* _tmp0, FHEdouble* _tmp1) {
    FHEdouble* _tmp3 = ck->FHEmul(_tmp0, _tmp0);
    FHEdouble* _tmp4 = ck->FHEsubP(-2, _tmp0);
    FHEdouble* _tmp5 = ck->FHEadd(_tmp4, _tmp3);
    return _tmp5;
}

int main() {
    // Initialize CKKS_scheme with appropriate parameters
    CKKS_scheme ck; // Assume default constructor is defined correctly

    std::vector<double> inputs1 = {32};
    std::vector<double> inputs2 = {21};

    // Ensure FHEencode and FHEencrypt methods work properly
    FHEdouble* _tmp0 = ck.FHEencrypt(ck.FHEencode(inputs1));
    FHEdouble* _tmp1 = ck.FHEencrypt(ck.FHEencode(inputs2));

    FHEdouble* result = _Z6rostinii(&ck, _tmp0, _tmp1);
    FHEplain* res = ck.FHEdecrypt(result);

    std::cout << "Result: " << res->getPlaintext() << std::endl;

    // Clean up dynamically allocated memory
    delete _tmp0;
    delete _tmp1;
    delete result;
    delete res;

    return 0;
}
