#ifndef FHE_TYPES_HPP
#define FHE_TYPES_HPP

#include <openfhe.h>

namespace CKKS {

class FHEdouble {
public:
    FHEdouble() : ciphertext(nullptr) {
    }
    FHEdouble(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>& cipher) : ciphertext(cipher) {}
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> getCiphertext() const { return ciphertext; }

private:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;
};

class FHEplain {
public:
    FHEplain() = default;
    FHEplain(const lbcrypto::Plaintext& ptxt) : plaintext(ptxt) {}
    lbcrypto::Plaintext getPlaintext() const { return plaintext; }

private:
    lbcrypto::Plaintext plaintext;
};

class FHEContext {
public:
    lbcrypto::CryptoContext<lbcrypto::DCRTPoly> cc;
    std::shared_ptr<lbcrypto::BinFHEContext> ccLWE;
};

class FHEKeyPair {
public:
    lbcrypto::KeyPair<lbcrypto::DCRTPoly> keys;
    lbcrypto::LWEPrivateKey privateKeyFHEW;
};

} 

namespace CGGI {

class FHEi32 {
public:
    FHEi32(lbcrypto::LWECiphertext& cipher) : ciphertext(cipher) {}
    lbcrypto::LWECiphertext& getCiphertext() { return ciphertext; }

private:
    lbcrypto::LWECiphertext ciphertext;
};

class FHEContext {
public:
    std::shared_ptr<lbcrypto::BinFHEContext> ccLWE;
};

class FHEKeyPair {
public:
    lbcrypto::LWEPrivateKey privateKeyFHEW;
};

} // namespace CGGI

#endif // FHE_TYPES_HPP
