#include "fhe_operations.hpp"
#include "fhe_types.hpp"
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

std::vector<CGGI::FHEi32> CKKS_scheme::CKKStoCGGI(FHEdouble a) {
    auto LWECiphertexts = context->cc->EvalCKKStoFHEW(a.getCiphertext(), this->batchSize);
    std::vector<CGGI::FHEi32> result;

    // Assuming LWECiphertexts is a vector of LWECiphertext
    for (auto& lweCipher : LWECiphertexts) {
        // Create a FHEi32 object for each LWECiphertext and store in the result vector
        result.push_back(CGGI::FHEi32(lweCipher));  
    }

    // Return the vector of FHEi32 pointers
    return result;

}

FHEdouble CKKS_scheme::CGGItoCKKS(std::vector<CGGI::FHEi32> a) {
    std::vector<lbcrypto::LWECiphertext> lweCiphertexts;
    // Unpack the vector of CGGI::FHEi32 to LWECiphertext
    for (auto& fhei32 : a) {
        lweCiphertexts.push_back(fhei32.getCiphertext());
    }
    auto ctxt = context->cc->EvalFHEWtoCKKS(lweCiphertexts,this->batchSize,this->batchSize);

    return FHEdouble(ctxt);

}


FHEplain CKKS_scheme::FHEencode(const std::vector<double>& a){
    FHEplain ptxt = context->cc->MakeCKKSPackedPlaintext(a);
    return FHEplain(ptxt);
}

FHEdouble CKKS_scheme::FHEencrypt(const FHEplain a){
    auto result = context->cc->Encrypt(keys->keys.publicKey, a.getPlaintext());
    return FHEdouble(result);
}

FHEplain CKKS_scheme::FHEdecrypt(const FHEdouble a){
    Plaintext result;
    context->cc->Decrypt(keys->keys.secretKey, a.getCiphertext(), &result);
    FHEplain res = result;
    return FHEplain(result);
}

// Arithmetic Operations for FHEdouble
FHEdouble CKKS_scheme::FHEaddf(const FHEdouble a, const FHEdouble b) {
    auto result = context->cc->EvalAdd(a.getCiphertext(), b.getCiphertext());
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEsubf(const FHEdouble a, const FHEdouble b) {
    auto result = context->cc->EvalSub(a.getCiphertext(), b.getCiphertext());
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEmulf(const FHEdouble a, const FHEdouble b) {
    auto result = context->cc->EvalMult(a.getCiphertext(), b.getCiphertext());
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEdivf(const FHEdouble a, const FHEdouble b) {
    auto temp = context->cc->EvalDivide(b.getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult(temp,a.getCiphertext());
    return FHEdouble(result);
}

// Arithmetic Operations with Plaintext
FHEdouble CKKS_scheme::FHEaddfP(const FHEdouble a, double b) {
    auto result = context->cc->EvalAdd(a.getCiphertext(), b);
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEsubfP(const FHEdouble a, double b) {
    auto result = context->cc->EvalSub(a.getCiphertext(), b);
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEsubfP(double b, const FHEdouble a) {
    auto result = context->cc->EvalSub(b, a.getCiphertext());
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEmulfP(const FHEdouble a, double b) {
    auto result = context->cc->EvalMult(a.getCiphertext(), b);
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEdivfP(const FHEdouble a, double b) {
    auto result = context->cc->EvalDivide(a.getCiphertext(), 1.0, 4294967295.0, 10);
    return FHEdouble(result);
}

FHEdouble CKKS_scheme::FHEdivfP(double b, const FHEdouble a) {
    auto temp = context->cc->EvalDivide(a.getCiphertext(), 1.0, 4294967295.0, 10);
    auto result = context->cc->EvalMult(temp, b);
    return FHEdouble(result);
}

}



namespace CGGI {   
    
    // Arithmetic Operations for FHEdouble
FHEi32 CGGI_scheme::FHEaddi(FHEi32 a, FHEi32 b) {
    context->ccLWE->GetLWEScheme()->EvalAddEq(a.getCiphertext(), b.getCiphertext());
    return a;
}

FHEi32 CGGI_scheme::FHEsubi(FHEi32 a, FHEi32 b) {
    context->ccLWE->GetLWEScheme()->EvalSubEq(a.getCiphertext(), b.getCiphertext());
    return a;
}

// Arithmetic Operations with Plaintext
FHEi32 CGGI_scheme::FHEaddiP(FHEi32 a, int b) {
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(a.getCiphertext(), b);
    return a;
}

FHEi32 CGGI_scheme::FHEsubiP(FHEi32 a, int b) {
    context->ccLWE->GetLWEScheme()->EvalSubConstEq(a.getCiphertext(), b);
    return a;
}

FHEi32 CGGI_scheme::FHEsubiP(int b, FHEi32 a) {
    auto temp = context->ccLWE->EvalNOT(a.getCiphertext());
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(temp, 1);
    context->ccLWE->GetLWEScheme()->EvalAddConstEq(temp,b);
    return FHEi32(temp);
}

FHEi32 CGGI_scheme::FHEmuliP(FHEi32 a, int b) {
    context->ccLWE->GetLWEScheme()->EvalMultConstEq(a.getCiphertext(), b);
    return a;
}

// FHEi32 CGGI_scheme::FHEmuli(FHEi32 a, FHEi32 b) {
//     auto result = context->cc->EvalMult(a->getCiphertext(), b->getCiphertext());
//     return FHEi32(result);
// }

// FHEi32 CGGI_scheme::FHEdivi(FHEi32 a, FHEi32 b) {
//     auto temp = context->cc->EvalDivide(b->getCiphertext(), 1.0, 4294967295.0, 10);
//     auto result = context->cc->EvalMult( FHEi32(temp))->getCiphertext(),a->getCiphertext());
//     return FHEi32(result);
// }

// FHEi32 CGGI_scheme::FHEdiviP(FHEi32 a, int b) {
//     auto result = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
//     return FHEi32(result);
// }

// FHEi32 CGGI_scheme::FHEdiviP(int b, FHEi32 a) {
//     auto temp = context->cc->EvalDivide(a->getCiphertext(), 1.0, 4294967295.0, 10);
//     auto result = context->cc->EvalMult( FHEi32(temp))->getCiphertext(),b);
//     return FHEi32(result);
// }

}


std::vector<CGGI::FHEi32> CKKStoCGGI(CKKS::CKKS_scheme& ck, CKKS::FHEdouble a){
    return ck.CKKStoCGGI(a);
}
CKKS::FHEdouble CGGItoCKKS(CKKS::CKKS_scheme& ck, std::vector<CGGI::FHEi32> a){
    return ck.CGGItoCKKS(a);
}

CKKS::FHEdouble FHEaddf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b){
    return ck.FHEaddf(a,b);
}
CKKS::FHEdouble FHEsubf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b){
    return ck.FHEsubf(a,b);
}
CKKS::FHEdouble FHEmulf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b){
    return ck.FHEmulf(a,b);
}
CKKS::FHEdouble FHEdivf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b){
    return ck.FHEdivf(a,b);
}

// Arithmetic Operations with Plaintext
CKKS::FHEdouble FHEaddfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b){
    return ck.FHEaddfP(a,b);
}
CKKS::FHEdouble FHEsubfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b){
    return ck.FHEsubfP(a,b);
}
CKKS::FHEdouble FHEsubfP(CKKS::CKKS_scheme& ck, double b, const CKKS::FHEdouble a){
    return ck.FHEsubfP(b,a);
}
CKKS::FHEdouble FHEmulfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b){
    return ck.FHEmulfP(a,b);
}
CKKS::FHEdouble FHEdivfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b){
    return ck.FHEdivfP(a,b);
}
CKKS::FHEdouble FHEdivfP(CKKS::CKKS_scheme& ck, double b, const CKKS::FHEdouble a){
    return ck.FHEdivfP(b,a);
}

// Use the Plaintext class from fhe_types.hpp
CKKS::FHEplain FHEencode(CKKS::CKKS_scheme& ck, const std::vector<double>& a){
    return ck.FHEencode(a);
}
CKKS::FHEdouble FHEencrypt(CKKS::CKKS_scheme& ck, const CKKS::FHEplain p){
    return ck.FHEencrypt(p);
}
CKKS::FHEplain FHEdecrypt(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a){
    return ck.FHEdecrypt(a);
}

CKKS::FHEdouble FHEencrypt(CKKS::CKKS_scheme& ck, const double a){
    return ck.FHEencrypt(ck.FHEencode({a}));
}