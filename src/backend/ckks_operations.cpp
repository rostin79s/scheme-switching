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

CKKS_scheme::CKKS_scheme(int multDepth = 1,int scaleModSize = 50, int batchSize = 1)
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

FHEdouble* CKKS_scheme::FHEencrypt(Plaintext a){
    auto result = context->cc->Encrypt(keys->keyPair.publicKey, a);
    return new FHEdouble(result);
}
        
Plaintext CKKS_scheme::FHEdecrypt(const FHEdouble* a){
    Plaintext result;
    context->cc->Decrypt(keys->keyPair.secretKey, a->getCiphertext(), &result);
    result->SetLength(batchSize);
    return result;
}

// Arithmetic Operations for FHEdouble
FHEdouble* CKKS_scheme::FHEadd(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEsub(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEmul(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdiv(const FHEdouble* a, const FHEdouble* b) {
    auto temp = context->cc->EvalDivide(b->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEdouble(temp))->getCiphertext(),a->getCiphertext());
    return new FHEdouble(result);
}

// Arithmetic Operations with Plaintext
FHEdouble* CKKS_scheme::FHEaddP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEsubP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEmulP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdivP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdivP(double b, const FHEdouble* a) {
    auto temp = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEdouble(temp))->getCiphertext(),b);
    return new FHEdouble(result);
}

}
