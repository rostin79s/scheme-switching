#include "fhe_operations.hpp"
#include "fhe_types.hpp"

using namespace CKKS;
using namespace TFHE;

void _Z6rostinii(FHEi32 _tmp0, FHEi32 _tmp1) {
CKKS_scheme ck;
    FHEi32 _tmp3 = ck.FHEmul(_tmp0, _tmp0);
    FHEi32 _tmp4 = ck.FHEaddP(_tmp0, 2);
    FHEi32 _tmp5 = ck.FHEand(_tmp3, _tmp4);
    FHEi32 _tmp6 = ck.FHEadd(_tmp5, _tmp1);
    return _tmp6;
}

int main() {
    std::vector<double> inputs = {/* User inputs */};
    std::vector<Ciphertext> encryptedInputs;
    FHEi32 _tmp0 = FHEencrypt(inputs[_tmp0]);
    FHEi32 _tmp1 = FHEencrypt(inputs[_tmp1]);
    Ciphertext result = _Z6rostinii(_tmp0, _tmp1);
    double finalResult = FHEdecrypt(result);
    std::cout << "Result: " << finalResult << std::endl;
    return 0;
}
