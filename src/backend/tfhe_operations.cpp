#include "fhe_operations.hpp"
#include <openfhe.h>

using namespace lbcrypto;

namespace TFHE {

// Arithmetic Operations for FHEi32
FHEi32* FHEadd(const FHEi32* a, const FHEi32* b) {
    auto result = cc.EvalBinGate((a->ciphertext, b->ciphertext);
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

// Arithmetic Operations with Plaintext
FHEi32* FHEaddP(const FHEi32* a, int32_t b) {
    auto result = EvalAdd(a->ciphertext, b);
    return new FHEi32(result);
}

FHEi32* FHEsubP(const FHEi32* a, int32_t b) {
    auto result = EvalSub(a->ciphertext, b);
    return new FHEi32(result);
}

FHEi32* FHEmulP(const FHEi32* a, int32_t b) {
    auto result = EvalMult(a->ciphertext, b);
    return new FHEi32(result);
}

// Boolean Operations for FHEi32
FHEi32* FHEand(const FHEi32* a, const FHEi32* b) {
    auto result = EvalBinGate(AND, a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEor(const FHEi32* a, const FHEi32* b) {
    auto result = EvalBinGate(OR, a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEXor(const FHEi32* a, const FHEi32* b) {
    auto result = EvalBinGate(XOR, a->ciphertext, b->ciphertext);
    return new FHEi32(result);
}

FHEi32* FHEnot(const FHEi32* a) {
    auto result = EvalNOT(a->ciphertext);
    return new FHEi32(result);
}

} // namespace TFHE
