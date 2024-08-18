#ifndef FHE_TYPES_HPP
#define FHE_TYPES_HPP

#include <openfhe.h>

namespace CKKS {

class FHEdouble {
public:
    FHEdouble(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cipher) : ciphertext(cipher) {}
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> getCiphertext() const { return ciphertext; }

private:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;
};

} // namespace CKKS

namespace TFHE {

class FHEi32 {
public:
    FHEi32(const lbcrypto::LWECiphertext& cipher) : ciphertext(cipher) {}
    lbcrypto::LWECiphertext getCiphertext() const { return ciphertext; }

private:
    lbcrypto::LWECiphertext ciphertext;
};

} // namespace TFHE

#endif // FHE_TYPES_HPP
