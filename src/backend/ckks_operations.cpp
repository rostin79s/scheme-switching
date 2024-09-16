#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CKKS {

CKKS_scheme::CKKS_scheme(int multDepth, int scaleModSize, int batchSize)
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
    context->cc->Enable(FHE);

    ringDimension = context->cc->GetRingDimension();

    keys = std::make_unique<FHEKeyPair>();
    keys->keyPair = context->cc->KeyGen();
    context->cc->EvalMultKeyGen(keys->keyPair.secretKey);
    context->cc->EvalRotateKeyGen(keys->keyPair.secretKey, {1, -2});

}

FHEplain* CKKS_scheme::FHEencode(const std::vector<double>& a){
    FHEplain ptxt = context->cc->MakeCKKSPackedPlaintext(a);
    return new FHEplain(ptxt);
}

FHEdouble* CKKS_scheme::FHEencrypt(const FHEplain* a){
    auto result = context->cc->Encrypt(keys->keyPair.publicKey, a->getPlaintext());
    return new FHEdouble(result);
}

FHEplain* CKKS_scheme::FHEdecrypt(const FHEdouble* a){
    Plaintext result;
    context->cc->Decrypt(keys->keyPair.secretKey, a->getCiphertext(), &result);
    FHEplain res = result;
    return new FHEplain(result);
}

// Arithmetic Operations for FHEdouble
FHEdouble* CKKS_scheme::FHEaddf(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEsubf(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEmulf(const FHEdouble* a, const FHEdouble* b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdivf(const FHEdouble* a, const FHEdouble* b) {
    auto temp = context->cc->EvalDivide(b->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEdouble(temp))->getCiphertext(),a->getCiphertext());
    return new FHEdouble(result);
}

// Arithmetic Operations with Plaintext
FHEdouble* CKKS_scheme::FHEaddfP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalAdd(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEsubfP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalSub(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEsubfP(double b, const FHEdouble* a) {
    auto result = context->cc->EvalSub(b, a->getCiphertext());
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEmulfP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalMult(a->getCiphertext(), b);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdivfP(const FHEdouble* a, double b) {
    auto result = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    return new FHEdouble(result);
}

FHEdouble* CKKS_scheme::FHEdivfP(double b, const FHEdouble* a) {
    auto temp = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult((new FHEdouble(temp))->getCiphertext(),b);
    return new FHEdouble(result);
}

}
