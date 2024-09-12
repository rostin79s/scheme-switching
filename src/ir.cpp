#include "backend/fhe_operations.hpp"
#include "backend/fhe_types.hpp"
#include <vector>
#include <iostream>

using namespace CKKS;
using namespace TFHE;

FHEdouble* rostin(CKKS_scheme& ck, FHEdouble* _tmp0, FHEdouble* _tmp1) {
    FHEdouble* _tmp3 = ck.FHEmul(_tmp1, _tmp0);
    FHEdouble* _tmp4 = ck.FHEmul(_tmp3, _tmp1);
    return _tmp4;
}

int main() {
	CKKS_scheme ck(2,50,1);
	std::vector<double> input1 = {23};
	FHEdouble* _tmp0 = ck.FHEencrypt(ck.FHEencode(input1));
	std::vector<double> input2 = {2};
	FHEdouble* _tmp1 = ck.FHEencrypt(ck.FHEencode(input2));
    FHEdouble* result = rostin(ck, _tmp0, _tmp1);
    FHEplain* res = ck.FHEdecrypt(result);
    std::cout << "Result: " << res->getPlaintext() << std::endl;
    return 0;
}
