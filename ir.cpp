#include "backend/fhe_operations.hpp"
#include "backend/fhe_types.hpp"
#include <vector>
#include <iostream>

using namespace CKKS;
using namespace TFHE;

FHEdouble* rostin(CKKS_scheme& ck, FHEdouble* x, FHEdouble* y) {
    FHEdouble* a1 = ck.FHEmul(x,y);
    FHEdouble* a2 = ck.FHEmul(a1, x);
    // FHEdouble* a3 = ck.FHEmul(a2, y);
    // auto a4 = ck.FHEmulP(a3,10.0);
    return a2;
}

int main() {
	CKKS_scheme ck(2,24,1);
	std::vector<double> input1 = {23};
	FHEdouble* x = ck.FHEencrypt(ck.FHEencode(input1));
	std::vector<double> input2 = {2};
	FHEdouble* y = ck.FHEencrypt(ck.FHEencode(input2));
    FHEdouble* result = rostin(ck, x, y);
    FHEplain* res = ck.FHEdecrypt(result);
    std::cout << "Result: " << res->getPlaintext() << std::endl;
    return 0;
}
