#ifndef FHE_OPERATIONS_HPP
#define FHE_OPERATIONS_HPP

#include "fhe_types.hpp"

namespace TFHE {
    class TFHE_scheme{
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

        FHEi32* FHEadd(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEsub(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEmul(const FHEi32* a, const FHEi32* b);
        FHEi32* FHEdiv(const FHEi32* a, const FHEi32* b);

        // Arithmetic Operations with Plaintext
        FHEi32* FHEaddP(const FHEi32* a, const double b);
        FHEi32* FHEsubP(const FHEi32* a, const double b);
        FHEi32* FHEmulP(const FHEi32* a, const double b);
        FHEi32* FHEdivP(const FHEi32* a, const double b);
        FHEi32* FHEdivP(double b, const FHEi32* a);

        Plaintext CKKS_scheme::FHEencode(std::vector<double> a);
        FHEi32* FHEencrypt(Plaintext p);
        Plaintext FHEdecrypt(const FHEi32* a);

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
