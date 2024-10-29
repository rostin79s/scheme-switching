#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CGGI {   

FHEi32 CGGI_scheme::FHEor(FHEi32 a, FHEi32 b){
    auto res = context->ccLWE->EvalBinGate(OR,a.getCiphertext(),b.getCiphertext());
    return FHEi32(res);

}

    
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

// CGGI::FHEi32 FHEor(CGGI::FHEi32 a, CGGI::FHEi32 b){
//     return conFHEor(a,b);
// }

