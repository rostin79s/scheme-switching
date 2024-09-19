#include "openfhe.h"
#include "binfhecontext.h"

using namespace lbcrypto;

void SwitchFHEWtoCKKS() {
    std::cout << "\n-----SwitchFHEWtoCKKS-----\n" << std::endl;
    std::cout << "Output precision is only wrt the operations in CKKS after switching back.\n" << std::endl;

    // Step 1: Setup CryptoContext for CKKS to be switched into

    // A. Specify main parameters
    ScalingTechnique scTech = FIXEDAUTO;
    // for r = 3 in FHEWtoCKKS, Chebyshev max depth allowed is 9, 1 more level for postscaling
    uint32_t multDepth = 3 + 9 + 1;
    if (scTech == FLEXIBLEAUTOEXT)
        multDepth += 1;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 8192;
    SecurityLevel sl      = HEStd_NotSet;  // If this is not HEStd_NotSet, ensure ringDim is compatible
    uint32_t logQ_ccLWE   = 29;

    // uint32_t slots = ringDim/2; // Uncomment for fully-packed
    uint32_t slots     = 16;  // sparsely-packed
    uint32_t batchSize = slots;

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetScalingTechnique(scTech);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);

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

    // Generate encryption keys.
    auto keys = cc->KeyGen();

    // Step 2: Prepare the FHEW cryptocontext and keys for FHEW and scheme switching
    auto ccLWE = std::make_shared<BinFHEContext>();
    ccLWE->BinFHEContext::GenerateBinFHEContext(TOY, false, logQ_ccLWE, 0, GINX, false);

    // LWE private key
    LWEPrivateKey lwesk;
    lwesk = ccLWE->KeyGen();

    std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    std::cout << ", logQ " << logQ_ccLWE;
    std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Step 3. Precompute the necessary keys and information for switching from FHEW to CKKS
    cc->EvalFHEWtoCKKSSetup(ccLWE, slots, logQ_ccLWE);
    cc->SetBinCCForSchemeSwitch(ccLWE);

    cc->EvalFHEWtoCKKSKeyGen(keys, lwesk);
    // Step 4: Encoding and encryption of inputs
    // For correct CKKS decryption, the messages have to be much smaller than the FHEW plaintext modulus!

    // auto pLWE1       = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    // uint32_t pLWE2   = 256;                                           // Medium precision
    auto modulus_LWE = 1 << logQ_ccLWE;
    std::cout<<"modulus_LWE: "<<modulus_LWE <<std::endl;
    // auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE3       = 1048576*128;  // Large precision
    // Inputs
    std::cout << "beta: "<<ccLWE->GetBeta().ConvertToInt() << "logQ_ccLWE: "<<logQ_ccLWE;
    std::vector<int> x1 = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
    std::vector<int> x2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 110, 1200, 13000, 140000, 1500000};
    if (x1.size() < slots) {
        std::vector<int> zeros(slots - x1.size(), 0);
        x1.insert(x1.end(), zeros.begin(), zeros.end());
        x2.insert(x2.end(), zeros.begin(), zeros.end());
    }

    // // Encrypt
    // std::vector<LWECiphertext> ctxtsLWE1(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE1[i] =
    //         ccLWE->Encrypt(lwesk, x1[i]);  // encrypted under small plantext modulus p = 4 and ciphertext modulus
    // }

    // std::vector<LWECiphertext> ctxtsLWE2(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE2[i] =
    //         ccLWE->Encrypt(lwesk, x1[i], FRESH,
    //                        pLWE1);  // encrypted under larger plaintext modulus p = 16 but small ciphertext modulus
    // }

    // std::vector<LWECiphertext> ctxtsLWE3(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE3[i] =
    //         ccLWE->Encrypt(lwesk, x2[i], FRESH, pLWE2,
    //                        modulus_LWE);  // encrypted under larger plaintext modulus and large ciphertext modulus
    // }

    std::vector<LWECiphertext> ctxtsLWE4(slots);
    for (uint32_t i = 0; i < slots; i++) {
        ctxtsLWE4[i] =
            ccLWE->Encrypt(lwesk, x2[i], FRESH, pLWE3,
                           modulus_LWE);  // encrypted under large plaintext modulus and large ciphertext modulus
    }

    LWEPlaintext result;
    ccLWE->GetLWEScheme()->EvalAddEq(ctxtsLWE4[0],ctxtsLWE4[1]);
    ccLWE->Decrypt(lwesk,ctxtsLWE4[0],&result,modulus_LWE);
    std::cout<<"addition: "<<result<<"\n";

    // // Step 5. Perform the scheme switching
    // auto cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE1, slots, slots);

    // std::cout << "\n---Input x1: " << x1 << " encrypted under p = " << 4 << " and Q = " << ctxtsLWE1[0]->GetModulus()
    //           << "---" << std::endl;

    // // Step 6. Decrypt
    // Plaintext plaintextDec;
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // std::cout << "Switched CKKS decryption 1: " << plaintextDec << std::endl;

    // // Step 5'. Perform the scheme switching
    // cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE2, slots, slots, pLWE1, 0, pLWE1);

    // std::cout << "\n---Input x1: " << x1 << " encrypted under p = " << NativeInteger(pLWE1)
    //           << " and Q = " << ctxtsLWE2[0]->GetModulus() << "---" << std::endl;

    // // Step 6'. Decrypt
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // std::cout << "Switched CKKS decryption 2: " << plaintextDec << std::endl;

    // // Step 5''. Perform the scheme switching
    // cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE3, slots, slots, pLWE2, 0, pLWE2);

    // std::cout << "\n---Input x2: " << x2 << " encrypted under p = " << pLWE2
    //           << " and Q = " << ctxtsLWE3[0]->GetModulus() << "---" << std::endl;

    // // Step 6''. Decrypt
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // std::cout << "Switched CKKS decryption 3: " << plaintextDec << std::endl;

    // Step 5'''. Perform the scheme switching
    
    // std::setprecision(logQ_ccLWE +50);
    auto cTemp2 = cc->EvalFHEWtoCKKS(ctxtsLWE4, slots, slots, pLWE3, 0, pLWE3);

    std::cout << "\n---Input x2: " << x2 << " encrypted under p = " << NativeInteger(pLWE3)
              << " and Q = " << ctxtsLWE4[0]->GetModulus() << "---" << std::endl;

    // Step 6'''. Decrypt
    Plaintext plaintextDec2;
    cc->Decrypt(keys.secretKey, cTemp2, &plaintextDec2);
    plaintextDec2->SetLength(slots);
    std::cout << "Switched CKKS decryption 4: " << plaintextDec2 << std::endl;
}

void test(){
    auto cc = BinFHEContext();

    // STD128 is the security level of 128 bits of security based on LWE Estimator
    // and HE standard. Other common options are TOY, MEDIUM, STD192, and STD256.
    // MEDIUM corresponds to the level of more than 100 bits for both quantum and
    // classical computer attacks.
    cc.GenerateBinFHEContext(STD128);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    std::cout << "Completed the key generation." << std::endl;

    // Sample Program: Step 3: Encryption

    // Encrypt two ciphertexts representing Boolean True (1).
    // By default, freshly encrypted ciphertexts are bootstrapped.
    // If you wish to get a fresh encryption without bootstrapping, write
    // auto   ct1 = cc.Encrypt(sk, 1, FRESH);
    int p = 1 << 27;
    int q = 1 << 30;
    auto ct1 = cc.Encrypt(sk, 1000000,FRESH,p,q);
    auto ct2 = cc.Encrypt(sk, 58623,FRESH,p,q);
    

    // cc.GetLWEScheme()->EvalMultConstEq(ct1,23);

    LWEPlaintext result;
    // cc.GetLWEScheme()->EvalAddEq(ct1,ct2);
    cc.GetLWEScheme()->EvalMultConstEq(ct1,16);
    cc.Decrypt(sk,ct1,&result,p);
    std::cout<<"result: "<<result<<"\n";

}

Ciphertext<DCRTPoly> SWITCHCKKSRNS::EvalCompareSchemeSwitching(ConstCiphertext<DCRTPoly> ciphertext1,
                                                               ConstCiphertext<DCRTPoly> ciphertext2, uint32_t numCtxts,
                                                               uint32_t numSlots, uint32_t pLWE, double scaleSign,
                                                               bool unit) {
    auto ccCKKS             = ciphertext1->GetCryptoContext();
    const auto cryptoParams = std::dynamic_pointer_cast<CryptoParametersCKKSRNS>(ccCKKS->GetCryptoParameters());

    auto cDiff = ccCKKS->EvalSub(ciphertext1, ciphertext2);

    if (unit) {
        if (pLWE == 0)
            OPENFHE_THROW("To scale to the unit circle, pLWE must be non-zero.");
        else {
            cDiff = ccCKKS->EvalMult(cDiff, 1.0 / static_cast<double>(pLWE));
            cDiff = ccCKKS->Rescale(cDiff);
        }
    }

    // The precomputation has already been performed, but if it is scaled differently than desired, recompute it
    if (pLWE != 0) {
        double scaleCF = 1.0;
        if ((pLWE != 0) && (!unit)) {
            scaleCF = 1.0 / pLWE;
        }
        scaleCF *= scaleSign;

        EvalCKKStoFHEWPrecompute(*ccCKKS, scaleCF);
    }

    auto LWECiphertexts = EvalCKKStoFHEW(cDiff, numCtxts);

    std::vector<LWECiphertext> cSigns(LWECiphertexts.size());
#pragma omp parallel for
    for (uint32_t i = 0; i < LWECiphertexts.size(); i++) {
        cSigns[i] = m_ccLWE->EvalSign(LWECiphertexts[i], true);
    }

    return EvalFHEWtoCKKS(cSigns, numCtxts, numSlots, 4, -1.0, 1.0, 0);
}

int main() {
    // SwitchFHEWtoCKKS();
    test();
    return 0;
}