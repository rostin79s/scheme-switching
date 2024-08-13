#ifndef FHE_OPERATIONS_HPP
#define FHE_OPERATIONS_HPP

#include "fhe_types.hpp"

namespace TFHE {

// Arithmetic Operations for FHEi32
FHEi32* FHEadd(const FHEi32* a, const FHEi32* b);
FHEi32* FHEsub(const FHEi32* a, const FHEi32* b);
FHEi32* FHEmul(const FHEi32* a, const FHEi32* b);

// Arithmetic Operations with Plaintext
FHEi32* FHEaddP(const FHEi32* a, int b);
FHEi32* FHEsubP(const FHEi32* a, int b);
FHEi32* FHEmulP(const FHEi32* a, int b);

// Boolean Operations for FHEi32
FHEi32* FHEand(const FHEi32* a, const FHEi32* b);
FHEi32* FHEor(const FHEi32* a, const FHEi32* b);
FHEi32* FHEXor(const FHEi32* a, const FHEi32* b);
FHEi32* FHEnot(const FHEi32* a);

} // namespace TFHE

namespace CKKS {

// Arithmetic Operations for FHEi32
FHEi32* FHEadd(const FHEi32* a, const FHEi32* b);
FHEi32* FHEsub(const FHEi32* a, const FHEi32* b);
FHEi32* FHEmul(const FHEi32* a, const FHEi32* b);
FHEi32* FHEdiv(const FHEi32* a, const FHEi32* b);

// Arithmetic Operations with Plaintext
FHEi32* FHEaddP(const FHEi32* a, double b);
FHEi32* FHEsubP(const FHEi32* a, double b);
FHEi32* FHEmulP(const FHEi32* a, double b);
FHEi32* FHEdivP(const FHEi32* a, double b);

// Boolean Operations for FHEi32 (CKKS may not fully support these, but provided as placeholders)
FHEi32* FHEand(const FHEi32* a, const FHEi32* b);
FHEi32* FHEor(const FHEi32* a, const FHEi32* b);
FHEi32* FHEXor(const FHEi32* a, const FHEi32* b);
FHEi32* FHEnot(const FHEi32* a);

} // namespace CKKS

#endif // FHE_OPERATIONS_HPP
