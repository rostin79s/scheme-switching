#include "fhe_types.hpp"
#include <openfhe.h> // Include the OpenFHE headers for implementation

namespace TFHE {

// Define FHEi32 and FHEi64 for TFHE
class FHEi32 {
public:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;

    FHEi32(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) : ciphertext(ct) {}
    // Add any necessary constructors, destructors, or member functions
};

class FHEi64 {
public:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;

    FHEi64(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) : ciphertext(ct) {}
    // Add any necessary constructors, destructors, or member functions
};

} // namespace TFHE

namespace CKKS {

// Define FHEi32 and FHEi64 for CKKS
class FHEi32 {
public:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;

    FHEi32(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) : ciphertext(ct) {}
    // Add any necessary constructors, destructors, or member functions
};

class FHEi64 {
public:
    lbcrypto::Ciphertext<lbcrypto::DCRTPoly> ciphertext;

    FHEi64(const lbcrypto::Ciphertext<lbcrypto::DCRTPoly> &ct) : ciphertext(ct) {}
    // Add any necessary constructors, destructors, or member functions
};

} // namespace CKKS
