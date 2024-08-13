#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace CKKS {

// Arithmetic Operations for FHEi32
FHEi32* FHEadd(const FHEi32* a, const FHEi32* b) {
    auto result = EvalAdd(a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEsub(const FHEi32* a, const FHEi32* b) {
    auto result = EvalSub(a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEmul(const FHEi32* a, const FHEi32* b) {
    auto result = EvalMult(a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEdiv(const FHEi32* a, const FHEi32* b) {
    auto result = EvalDiv(a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

// Arithmetic Operations with Plaintext
FHEi32* FHEaddP(const FHEi32* a, double b) {
    auto result = EvalAdd(a->ciphertext, b);
    return new FHEi32(result);
}

FHEi32* FHEsubP(const FHEi32* a, double b) {
    auto result = EvalSub(a->ciphertext, b);
    return new FHEi32(result);
}

FHEi32* FHEmulP(const FHEi32* a, double b) {
    auto result = EvalMult(a->ciphertext, b);
    return new FHEi32(result);
}

FHEi32* FHEdivP(const FHEi32* a, double b) {
    auto result = EvalDiv(a->ciphertext, b);
    return new FHEi32(result);
}

// Boolean Operations for FHEi32 (CKKS may not fully support these, but placeholders are provided)
FHEi32* FHEand(const FHEi32* a, const FHEi32* b) {
    // Placeholder implementation
    return nullptr;
}

FHEi32* FHEor(const FHEi32* a, const FHEi32* b) {
    // Placeholder implementation
    return nullptr;
}

FHEi32* FHEXor(const FHEi32* a, const FHEi32* b) {
    // Placeholder implementation
    return nullptr;
}

FHEi32* FHEnot(const FHEi32* a) {
    // Placeholder implementation
    return nullptr;
}

} // namespace CKKS
