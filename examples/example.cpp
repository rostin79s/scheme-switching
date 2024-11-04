#include "openfhe.h"
#include "scheme/ckksrns/ckksrns-schemeswitching.h"

using namespace lbcrypto;

void rostin();

void ComparisonViaSchemeSwitching();

int main(int argc, char* argv[]) {
    // rostin();
    ComparisonViaSchemeSwitching();
}

void ComparisonViaSchemeSwitching() {
    std::cout << "\n-----ComparisonViaSchemeSwitching-----\n" << std::endl;
    std::cout << "Output precision is only wrt the operations in CKKS after switching back.\n" << std::endl;

    // Step 1: Setup CryptoContext for CKKS
    ScalingTechnique scTech = FLEXIBLEAUTO;
    uint32_t multDepth      = 17;
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;

    uint32_t scaleModSize = 50;
    uint32_t firstModSize = 60;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
    uint32_t slots        = 16;  // sparsely-packed
    uint32_t batchSize    = slots;

    CCParams<CryptoContextCKKSRNS> parameters;
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

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(SCHEMESWITCH);

    std::cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension();
    std::cout << ", number of slots " << slots << ", and supports a multiplicative depth of " << multDepth << std::endl
              << std::endl;

    // Generate encryption keys
    auto keys = cc->KeyGen();

    // Step 2: Prepare the FHEW cryptocontext and keys for FHEW and scheme switching
    SchSwchParams params;
    params.SetSecurityLevelCKKS(sl);
    params.SetSecurityLevelFHEW(slBin);
    params.SetCtxtModSizeFHEWLargePrec(logQ_ccLWE);
    params.SetNumSlotsCKKS(slots);
    params.SetNumValues(slots);
    auto privateKeyFHEW = cc->EvalSchemeSwitchingSetup(params);
    auto ccLWE          = cc->GetBinCCForSchemeSwitch();

    ccLWE->BTKeyGen(privateKeyFHEW);
    cc->EvalSchemeSwitchingKeyGen(keys, privateKeyFHEW);

    std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    std::cout << ", logQ " << logQ_ccLWE;
    std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Set the scaling factor to be able to decrypt; the LWE mod switch is performed on the ciphertext at the last level
    // auto pLWE1           = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    auto modulus_LWE     = 1 << logQ_ccLWE;
    auto beta            = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2           = modulus_LWE / (2 * beta);  // Large precision
    double scaleSignFHEW = 1.0;
    cc->EvalCompareSwitchPrecompute(pLWE2, scaleSignFHEW);

    std::cout << "plwe1: " << ccLWE->GetMaxPlaintextSpace().ConvertToInt() << std::endl;

    // Step 3: Encoding and encryption of inputs
    // Inputs
    std::vector<double> x1 = {0.0, 6, 2.0, 3.0, 4.0, 5.5, 6.0, 7, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0};
    std::vector<double> x2(slots, 6);

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr, slots);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, 0, nullptr, slots);

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    // Compute the difference to compare to zero
    auto cDiff = cc->EvalSub(c1, c2);

    auto LWECiphertexts = cc->EvalCKKStoFHEW(cDiff, slots);

    auto ckksctxts = cc->EvalFHEWtoCKKS(LWECiphertexts, batchSize, batchSize);

    // Decrypt the result
    Plaintext result;
    cc->Decrypt(keys.secretKey, ckksctxts, &result);
    std::cout << "Result: " << result << std::endl;

    size_t sag = cc->GetRingDimension() / 2;
    std::cout << "Batch size: " << sag << std::endl;

    LWEPlaintext plainLWE;
    
    std::vector<LWECiphertext> LWESign(LWECiphertexts.size());
    for (uint32_t i = 0; i < LWECiphertexts.size(); ++i) {
        LWESign[i] = ccLWE->EvalSign(LWECiphertexts[i]);
        ccLWE->Decrypt(privateKeyFHEW, LWESign[i], &plainLWE, 2);
        std::cout << plainLWE << " ";
    }
    std::cout << "\n";

}
   