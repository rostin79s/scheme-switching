#include "backend/fhe_operations.hpp"
#include "backend/fhe_types.hpp"
#include <vector>

using namespace CKKS;
using namespace TFHE;

FHEdouble* _Z6rostinii(CKKS_scheme& ck, FHEdouble* _tmp0, FHEdouble* _tmp1) {
    FHEdouble* _tmp3 = ck.FHEmul(_tmp0, _tmp0);
    FHEdouble* _tmp4 = ck.FHEsubP(-2, _tmp0);
    FHEdouble* _tmp5 = ck.FHEadd(_tmp4, _tmp3);
    FHEdouble* _tmp6 = ck.FHEadd(_tmp5, _tmp1);
    return _tmp6;
}

int main() {
	CKKS_scheme ck;
	std::vector<double> inputs = {/* User inputs */};
	std::vector<FHEdouble*> encryptedInputs;
    FHEdouble* _tmp0 = ck.FHEencrypt(inputs[_tmp0]);
    FHEdouble* _tmp1 = ck.FHEencrypt(inputs[_tmp1]);
    FHEdouble* result = _Z6rostinii(ck, _tmp0, _tmp1);
    double finalResult = ck.FHEdecrypt(result);
    std::cout << "Result: " << finalResult << std::endl;
    return 0;
}
