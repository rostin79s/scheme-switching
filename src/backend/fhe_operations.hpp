#ifndef FHE_OPERATIONS_HPP
#define FHE_OPERATIONS_HPP

#include "fhe_types.hpp"

namespace CGGI {
    class FHEContext;  // Forward declaration of abstract context class
    class FHEKeyPair;

    class CGGI_scheme {
    public:
        FHEi32 FHEaddi( FHEi32 a, FHEi32 b);
        FHEi32 FHEsubi(FHEi32 a, FHEi32 b);

        // Arithmetic Operations with Plaintext
        FHEi32 FHEaddiP(FHEi32 a, int b);
        FHEi32 FHEsubiP(FHEi32 a, int b);
        FHEi32 FHEsubiP(int b, FHEi32 a);
        FHEi32 FHEmuliP(FHEi32 a, int b);

        // Boolean Operations for FHEi32
        FHEi32 FHEand(FHEi32 a, FHEi32 b);
        FHEi32 FHEor(FHEi32 a, FHEi32 b);
        FHEi32 FHEXor(FHEi32 a, FHEi32 b);
        FHEi32 FHEnot(FHEi32 a);

    private:
        std::unique_ptr<FHEContext> context;   // Use unique_ptr for opaque pointer to the implementation
        std::unique_ptr<FHEKeyPair> keys;
    };
} // namespace CGGI

namespace CKKS {
    class FHEContext;  // Forward declaration of abstract context class
    class FHEKeyPair;  // Forward declaration of abstract key pair class

    class CKKS_scheme {
    public:
        CKKS_scheme(int multDepth = 17, int scaleModSize = 50, int batchSize = 1);

        std::vector<CGGI::FHEi32> CKKStoCGGI(FHEdouble a);
        FHEdouble CGGItoCKKS(std::vector<CGGI::FHEi32> a);

        FHEdouble FHEaddf(const FHEdouble a, const FHEdouble b);
        FHEdouble FHEsubf(const FHEdouble a, const FHEdouble b);
        FHEdouble FHEmulf(const FHEdouble a, const FHEdouble b);
        FHEdouble FHEdivf(const FHEdouble a, const FHEdouble b);

        // Arithmetic Operations with Plaintext
        FHEdouble FHEaddfP(const FHEdouble a, double b);
        FHEdouble FHEsubfP(const FHEdouble a, double b);
        FHEdouble FHEsubfP(double b, const FHEdouble a);
        FHEdouble FHEmulfP(const FHEdouble a, double b);
        FHEdouble FHEdivfP(const FHEdouble a, double b);
        FHEdouble FHEdivfP(double b, const FHEdouble a);

        // Use the Plaintext class from fhe_types.hpp
        FHEplain FHEencode(const std::vector<double>& a);
        FHEdouble FHEencrypt(const CKKS::FHEplain p);
        FHEplain FHEdecrypt(const FHEdouble a);

    private:
        int multDepth;
        int scaleModSize;
        int batchSize;
        int ringDimension;

        std::unique_ptr<FHEContext> context;   // Use unique_ptr for opaque pointer to the implementation
        std::unique_ptr<FHEKeyPair> keys; 
    };
}


std::vector<CGGI::FHEi32> CKKStoCGGI(CKKS::CKKS_scheme& ck, CKKS::FHEdouble a);
CKKS::FHEdouble CGGItoCKKS(CKKS::CKKS_scheme& ck, std::vector<CGGI::FHEi32> a);

CKKS::FHEdouble FHEaddf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b);
CKKS::FHEdouble FHEsubf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b);
CKKS::FHEdouble FHEmulf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b);
CKKS::FHEdouble FHEdivf(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, const CKKS::FHEdouble b);

// Arithmetic Operations with Plaintext
CKKS::FHEdouble FHEaddfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b);
CKKS::FHEdouble FHEsubfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b);
CKKS::FHEdouble FHEsubfP(CKKS::CKKS_scheme& ck, double b, const CKKS::FHEdouble a);
CKKS::FHEdouble FHEmulfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b);
CKKS::FHEdouble FHEdivfP(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a, double b);
CKKS::FHEdouble FHEdivfP(CKKS::CKKS_scheme& ck, double b, const CKKS::FHEdouble a);

// Use the Plaintext class from fhe_types.hpp
CKKS::FHEplain FHEencode(CKKS::CKKS_scheme& ck, const std::vector<double>& a);
CKKS::FHEdouble FHEencrypt(CKKS::CKKS_scheme& ck, const CKKS::FHEplain p);
CKKS::FHEplain FHEdecrypt(CKKS::CKKS_scheme& ck, const CKKS::FHEdouble a);

CKKS::FHEdouble FHEencrypt(CKKS::CKKS_scheme& ck, const double a);

#endif // FHE_OPERATIONS_HPP
