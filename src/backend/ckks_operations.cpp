#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CKKS {

CKKS_scheme::CKKS_scheme(int multDepth, int scaleModSize, int batchSize)
    : multDepth(multDepth), scaleModSize(scaleModSize), batchSize(batchSize) {


    // uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
    // uint32_t slots        = 1;  
    // uint32_t batchSize    = slots;

    ScalingTechnique scTech = FLEXIBLEAUTO;
    // uint32_t multDepth      = 17;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetNumLargeDigits(3);

    context = std::make_unique<FHEContext>();

    context->cc = GenCryptoContext(parameters);

    context->cc->Enable(PKE);
    context->cc->Enable(KEYSWITCH);
    context->cc->Enable(LEVELEDSHE);
    context->cc->Enable(ADVANCEDSHE);
    context->cc->Enable(SCHEMESWITCH);


    keys = std::make_unique<FHEKeyPair>();
    keys->keys = context->cc->KeyGen();
    // context->cc->EvalMultKeyGen(keys->keyPair.secretKey);
    // context->cc->EvalRotateKeyGen(keys->keyPair.secretKey, {1, -2});

    SchSwchParams params;
    params.SetSecurityLevelCKKS(sl);
    params.SetSecurityLevelFHEW(slBin);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    params.SetNumSlotsCKKS(batchSize);
    params.SetNumValues(batchSize);
    auto privateKeyFHEW = context->cc->EvalSchemeSwitchingSetup(params);
    context->ccLWE = context->cc->GetBinCCForSchemeSwitch();

    context->ccLWE->BTKeyGen(privateKeyFHEW);
    context->cc->EvalSchemeSwitchingKeyGen(keys->keys, privateKeyFHEW);

    auto modulus_LWE     = 1 << logQ_ccLWE;
    auto beta            = context->ccLWE->GetBeta().ConvertToInt();
    auto pLWE2           = modulus_LWE / (2 * beta);  // Large precision
    double scaleSignFHEW = 1.0;
    context->cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);


}

std::vector<CGGI::FHEi32*> CKKS_scheme::CKKStoCGGI(FHEdouble* a) {
    auto LWECiphertexts = context->cc->EvalCKKStoFHEW(a->getCiphertext(), this->batchSize);
    std::vector<CGGI::FHEi32*> result;

    // Assuming LWECiphertexts is a vector of LWECiphertext
    for (auto& lweCipher : LWECiphertexts) {
        // Create a new FHEi32 object for each LWECiphertext and store in the result vector
        result.push_back(new CGGI::FHEi32(lweCipher));  
    }

    // Return the vector of FHEi32 pointers
    return result;

}

FHEdouble* CKKS_scheme::CGGItoCKKS(std::vector<CGGI::FHEi32*> a) {
    std::vector<lbcrypto::LWECiphertext> lweCiphertexts;
    // Unpack the vector of CGGI::FHEi32* to LWECiphertext
    for (auto& fhei32 : a) {
        lweCiphertexts.push_back(fhei32->getCiphertext());
    }
    auto ctxt = context->cc->EvalFHEWtoCKKS(lweCiphertexts,this->batchSize,this->batchSize);

    return new FHEdouble(ctxt);

}


FHEplain* CKKS_scheme::FHEencode(const std::vector<double>& a){
    FHEplain ptxt = context->cc->MakeCKKSPackedPlaintext(a);
    return new FHEplain(ptxt);
}

FHEdouble* CKKS_scheme::FHEencrypt(const FHEplain* a){
    auto result = context->cc->Encrypt(keys->keys.publicKey, a->getPlaintext());
    return new FHEdouble(result);
}

FHEplain* CKKS_scheme::FHEdecrypt(const FHEdouble* a){
    Plaintext result;
    context->cc->Decrypt(keys->keys.secretKey, a->getCiphertext(), &result);
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



namespace CGGI {   
    
    // Arithmetic Operations for FHEdouble
FHEi32* CGGI_scheme::FHEaddi(FHEi32* a, FHEi32* b) {
    context->ccLWE->GetLWEScheme()->EvalAddEq(a->getCiphertext(), b->getCiphertext());
    return a;
}

FHEi32* CGGI_scheme::FHEsubi(FHEi32* a, FHEi32* b) {
    context->ccLWE->GetLWEScheme()->EvalSubEq(a->getCiphertext(), b->getCiphertext());
    return a;
}

// Arithmetic Operations with Plaintext
FHEi32* CGGI_scheme::FHEaddiP(FHEi32* a, int b) {
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(a->getCiphertext(), b);
    return a;
}

FHEi32* CGGI_scheme::FHEsubiP(FHEi32* a, int b) {
    context->ccLWE->GetLWEScheme()->EvalSubConstEq(a->getCiphertext(), b);
    return a;
}

FHEi32* CGGI_scheme::FHEsubiP(int b, FHEi32* a) {
    auto temp = context->ccLWE->EvalNOT(a->getCiphertext());
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(temp, 1);
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(temp,b);
    return new FHEi32(temp);
}

FHEi32* CGGI_scheme::FHEmuliP(FHEi32* a, int b) {
    context->ccLWE->GetLWEScheme()->EvalMultConstEq(a->getCiphertext(), b);
    return a;
}

// FHEi32* CGGI_scheme::FHEmuli(FHEi32* a, FHEi32* b) {
//     auto result = context->cc->EvalMult(a->getCiphertext(), b->getCiphertext());
//     return new FHEi32(result);
// }

// FHEi32* CGGI_scheme::FHEdivi(FHEi32* a, FHEi32* b) {
//     auto temp = context->cc->EvalDivide(b->getCiphertext(), 1.0, 4294967295.0, 10);
//     auto result = context->cc->EvalMult((new FHEi32(temp))->getCiphertext(),a->getCiphertext());
//     return new FHEi32(result);
// }

// FHEi32* CGGI_scheme::FHEdiviP(FHEi32* a, int b) {
//     auto result = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
//     return new FHEi32(result);
// }

// FHEi32* CGGI_scheme::FHEdiviP(int b, FHEi32* a) {
//     auto temp = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
//     auto result = context->cc->EvalMult((new FHEi32(temp))->getCiphertext(),b);
//     return new FHEi32(result);
// }

}