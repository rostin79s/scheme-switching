#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CKKS {

class FHEContext {
public:
    CryptoContext<DCRTPoly> cc;
};

class FHEKeyPair {
public:
    KeyPair<DCRTPoly> keyPair;
};

CKKS_scheme::CKKS_scheme(int multDepth = 1,int scaleModSize = 50, int batchSize = 8)
    : multDepth(multDepth), scaleModSize(scaleModSize), batchSize(batchSize) {
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetBatchSize(batchSize);

    context = std::make_unique<FHEContext>();
    context->cc = GenCryptoContext(parameters);

    context->cc->Enable(PKE);
    context->cc->Enable(KEYSWITCH);
    context->cc->Enable(LEVELEDSHE);

    ringDimension = context->cc->GetRingDimension();

    keys = std::make_unique<FHEKeyPair>();
    keys->keyPair = context->cc->KeyGen();
    context->cc->EvalMultKeyGen(keys->keyPair.secretKey);
    context->cc->EvalRotateKeyGen(keys->keyPair.secretKey, {1, -2});

}

Plaintext CKKS_scheme::FHEencode(std::vector<double> a){
    Plaintext ptxt = context->cc->MakeCKKSPackedPlaintext(a);
    return ptxt;
}

FHEi32* CKKS_scheme::FHEencrypt(Plaintext a){
    auto result = context->cc->Encrypt(keys->keyPair.publicKey, a);
    return new FHEi32(result);
}
        
Plaintext CKKS_scheme::FHEdecrypt(const FHEi32* a){
    Plaintext result;
    context->cc->Decrypt(keys->keyPair.secretKey, a->getCiphertext(), &result);
    result->SetLength(batchSize);
    return result;
}

// Arithmetic Operations for FHEi32
FHEi32* CKKS_scheme::FHEadd(const FHEi32* a, const FHEi32* b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b->getCiphertext());
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEsub(const FHEi32* a, const FHEi32* b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b->getCiphertext());
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEmul(const FHEi32* a, const FHEi32* b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b->getCiphertext());
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEdiv(const FHEi32* a, const FHEi32* b) {
    auto temp = context->cc->EvalDivide(b->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEi32(temp))->getCiphertext(),a->getCiphertext());
    return new FHEi32(result);
}

// Arithmetic Operations with Plaintext
FHEi32* CKKS_scheme::FHEaddP(const FHEi32* a, double b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b);
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEsubP(const FHEi32* a, double b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b);
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEmulP(const FHEi32* a, double b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b);
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEdivP(const FHEi32* a, double b) {
    auto result = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    return new FHEi32(result);
}

FHEi32* CKKS_scheme::FHEdivP(double b, const FHEi32* a) {
    auto temp = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEi32(temp))->getCiphertext(),b);
    return new FHEi32(result);
}

}
