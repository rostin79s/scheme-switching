#ifndef FHE_OPERATIONS_HPP
#define FHE_OPERATIONS_HPP

#include "fhe_types.hpp"

namespace TFHE {
    class TFHE_scheme {
    public:
        FHEi32* FHEadd(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEsub(const FHEi32* a, const FHEi32* b);

        // Arithmetic Operations with Plaintext
        FHEi32* FHEaddP(const FHEi32* a, int b);
        FHEi32* FHEsubP(const FHEi32* a, int b);
        FHEi32* FHEmulP(const FHEi32* a, int b);

        // Boolean Operations for FHEi32
        FHEi32* FHEand(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEor(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEXor(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEnot(const FHEi32* a);

    private:
    };
} // namespace TFHE

namespace CKKS {
    class FHEContext;  // Forward declaration of abstract context class
    class FHEKeyPair;  // Forward declaration of abstract key pair class

    class CKKS_scheme {
    public:
        CKKS_scheme(int multDepth = 1, int scaleModSize = 50, int batchSize = 8);

        FHEdouble* FHEadd(const FHEdouble* a, const FHEdouble* b);
        FHEdouble* FHEsub(const FHEdouble* a, const FHEdouble* b);
        FHEdouble* FHEmul(const FHEdouble* a, const FHEdouble* b);
        FHEdouble* FHEdiv(const FHEdouble* a, const FHEdouble* b);

        // Arithmetic Operations with Plaintext
        FHEdouble* FHEaddP(const FHEdouble* a, double b);
        FHEdouble* FHEsubP(const FHEdouble* a, double b);
        FHEdouble* FHEsubP(double b, const FHEdouble* a);
        FHEdouble* FHEmulP(const FHEdouble* a, double b);
        FHEdouble* FHEdivP(const FHEdouble* a, double b);
        FHEdouble* FHEdivP(double b, const FHEdouble* a);

        // Use the Plaintext class from fhe_types.hpp
        FHEplain* FHEencode(const std::vector<double>& a);
        FHEdouble* FHEencrypt(const CKKS::FHEplain* p);
        FHEplain* FHEdecrypt(const FHEdouble* a);

    private:
        int multDepth;
        int scaleModSize;
        int batchSize;
        int ringDimension;

        std::unique_ptr<FHEContext> context;   // Use unique_ptr for opaque pointer to the implementation
        std::unique_ptr<FHEKeyPair> keys; 
    };
}

#endif // FHE_OPERATIONS_HPP
