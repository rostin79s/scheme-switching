#include "fhe_types.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CKKS {

class FHEi32 {
public:
    FHEi32(const Ciphertext<DCRTPoly>& cipher) : ciphertext(cipher) {}
    Ciphertext<DCRTPoly> getCiphertext() const { return ciphertext; }

private:
    Ciphertext<DCRTPoly> ciphertext;
};

}  // namespace CKKS
