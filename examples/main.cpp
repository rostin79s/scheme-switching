#include "binfhe-constants.h"
#include "openfhe.h"
#include "binfhecontext.h"
#include <iostream>

using namespace lbcrypto;
using namespace std;

void sag(){
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    // Set the ciphertext modulus to be 1 << 17
    // Note that normally we do not use this way to obtain the input ciphertext.
    // Instead, we assume that an LWE ciphertext with large ciphertext
    // modulus is already provided (e.g., by extracting from a CKKS ciphertext).
    // However, we do not provide such a step in this example.
    // Therefore, we use a brute force way to create a large LWE ciphertext.

    cc.GenerateBinFHEContext(TOY, GINX);

    auto sk = cc.KeyGen();

    cout << "Generating the bootstrapping keys..." << endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    cout << "Completed the key generation." << endl;

    // Sample Program: Step 3: Extract the MSB and decrypt to check the result
    // Note that we check for 8 different numbers


        // Get the MSB

    // auto p          = 6;
    auto a = cc.Encrypt(sk, 3,lbcrypto::FRESH, 1 << 4);
    auto b = cc.Encrypt(sk, 2,lbcrypto::FRESH, 1 << 4);
    auto c = cc.Encrypt(sk, 0,lbcrypto::FRESH, 1 << 4);
    // cc.GetLWEScheme()->EvalSubEq(a,b);
    // auto s = cc.EvalSign(a);
    cc.GetLWEScheme()->EvalAddEq(a,b);
    // vector<LWECiphertext> ctxts;
    // ctxts.push_back(a);
    // ctxts.push_back(b);
    // ctxts.push_back(c);
    // auto ctres = cc.EvalBinGate(CMUX, ctxts);
    LWEPlaintext result;
    cc.Decrypt(sk, a, &result, 1 << 8);
    cout << "Result: " << result << endl;

}

void cmux() {
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    cc.GenerateBinFHEContext(STD128_4, GINX);

    // Sample Program: Step 2: Key Generation

    // Generate the secret key
    auto sk = cc.KeyGen();

    cout << "Generating the bootstrapping keys..." << endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    cout << "Completed the key generation." << endl;

    // Sample Program: Step 3: Encryption

    // Encrypt several ciphertexts representing Boolean True (1) or False (0).
    // plaintext modulus is set higher than 4 to 2 * num_of_inputs
    auto p          = 6;
    auto ct1_3input = cc.Encrypt(sk, 1, SMALL_DIM, p);
    auto ct2_3input = cc.Encrypt(sk, 1, SMALL_DIM, p);
    auto ct3_3input = cc.Encrypt(sk, 0, SMALL_DIM, p);

    // 1, 1, 0
    vector<LWECiphertext> ct123;
    ct123.push_back(ct1_3input);
    ct123.push_back(ct2_3input);
    ct123.push_back(ct3_3input);

    // 1, 1, 0
    auto ctAND3 = cc.EvalBinGate(AND3, ct123);

    // 1, 1, 0
    auto ctOR3 = cc.EvalBinGate(OR3, ct123);
    // Sample Program: Step 5: Decryption

    LWEPlaintext result;
    cc.Decrypt(sk, ctAND3, &result, p);
    if (result != 0)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of AND(1, 1, 0) = " << result << endl;

    cc.Decrypt(sk, ctOR3, &result, p);
    if (result != 1)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of OR(1, 1, 0) = " << result << endl;

    // majority gate and cmux for 3 input does not need higher plaintext modulus
    p                  = 4;
    auto ct1_3input_p4 = cc.Encrypt(sk, 1, lbcrypto::FRESH, p);
    auto ct2_3input_p4 = cc.Encrypt(sk, 1, lbcrypto::FRESH, p);
    auto ct3_3input_p4 = cc.Encrypt(sk, 0, lbcrypto::FRESH, p);
    auto ct4_3input_p4 = cc.Encrypt(sk, 0, lbcrypto::FRESH, p);

    // 1, 1, 0
    vector<LWECiphertext> ct123_p4;
    ct123_p4.push_back(ct1_3input_p4);
    ct123_p4.push_back(ct2_3input_p4);
    ct123_p4.push_back(ct3_3input_p4);

    // 1, 0, 0
    vector<LWECiphertext> ct134_p4;
    ct134_p4.push_back(ct1_3input_p4);
    ct134_p4.push_back(ct3_3input_p4);
    ct134_p4.push_back(ct4_3input_p4);

    // 1, 0, 1
    vector<LWECiphertext> ct132_p4;
    ct132_p4.push_back(ct1_3input_p4);
    ct132_p4.push_back(ct3_3input_p4);
    ct132_p4.push_back(ct2_3input_p4);

    // 1, 1, 0
    auto ctMajority = cc.EvalBinGate(MAJORITY, ct123_p4);

    // 1, 0, 1
    auto ctCMUX0 = cc.EvalBinGate(CMUX, ct132_p4);

    // 1, 0, 0
    auto ctCMUX1 = cc.EvalBinGate(CMUX, ct134_p4);

    cc.Decrypt(sk, ctMajority, &result);
    if (result != 1)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of Majority(1, 1, 0) = " << result << endl;

    cc.Decrypt(sk, ctCMUX1, &result);
    if (result != 1)
        OPENFHE_THROW("Decryption failure");
    cout << "Result of encrypted computation of CMUX(1, 0, 0) = " << result << endl;

    cc.Decrypt(sk, ctCMUX0, &result);
    if (result != 0)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of CMUX(1, 0, 1) = " << result << endl;

    // for 4 input gates
    p               = 8;
    auto ct1_4input = cc.Encrypt(sk, 1, SMALL_DIM, p);
    auto ct2_4input = cc.Encrypt(sk, 0, SMALL_DIM, p);
    auto ct3_4input = cc.Encrypt(sk, 0, SMALL_DIM, p);
    auto ct4_4input = cc.Encrypt(sk, 0, SMALL_DIM, p);

    // 1, 0, 0, 0
    vector<LWECiphertext> ct1234;
    ct1234.push_back(ct1_4input);
    ct1234.push_back(ct2_4input);
    ct1234.push_back(ct3_4input);
    ct1234.push_back(ct4_4input);

    // Sample Program: Step 4: Evaluation

    // 1, 0, 0, 0
    auto ctAND4 = cc.EvalBinGate(AND4, ct1234);

    // 1, 0, 0, 0
    auto ctOR4 = cc.EvalBinGate(OR4, ct1234);

    // Sample Program: Step 5: Decryption
    cc.Decrypt(sk, ctAND4, &result, p);
    if (result != 0)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of AND(1, 0, 0, 0) = " << result << endl;

    cc.Decrypt(sk, ctOR4, &result, p);
    if (result != 1)
        OPENFHE_THROW("Decryption failure");

    cout << "Result of encrypted computation of OR(1, 0, 0, 0) = " << result << endl;

}

void SwitchFHEWtoCKKS() {
    cout << "\n-----SwitchFHEWtoCKKS-----\n" << endl;
    cout << "Output precision is only wrt the operations in CKKS after switching back.\n" << endl;

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

    cout << "CKKS scheme is using ring dimension " << cc->GetRingDimension();
    cout << ", number of slots " << slots << ", and supports a multiplicative depth of " << multDepth << endl
              << endl;

    // Generate encryption keys.
    auto keys = cc->KeyGen();

    // Step 2: Prepare the FHEW cryptocontext and keys for FHEW and scheme switching
    auto ccLWE = make_shared<BinFHEContext>();
    ccLWE->BinFHEContext::GenerateBinFHEContext(TOY, false, logQ_ccLWE, 0, GINX, false);

    // LWE private key
    LWEPrivateKey lwesk;
    lwesk = ccLWE->KeyGen();

    ccLWE->BTKeyGen(lwesk);

    cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    cout << ", logQ " << logQ_ccLWE;
    cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << endl << endl;

    // Step 3. Precompute the necessary keys and information for switching from FHEW to CKKS
    cc->EvalFHEWtoCKKSSetup(ccLWE, slots, logQ_ccLWE);
    cc->SetBinCCForSchemeSwitch(ccLWE);

    cc->EvalFHEWtoCKKSKeyGen(keys, lwesk);
    // Step 4: Encoding and encryption of inputs
    // For correct CKKS decryption, the messages have to be much smaller than the FHEW plaintext modulus!

    // auto pLWE1       = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    // uint32_t pLWE2   = 256;                                           // Medium precision
    auto modulus_LWE = 1 << logQ_ccLWE;
    cout<<"modulus_LWE: "<<modulus_LWE <<endl;
    // auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE3       = 1<<10;  // Large precision
    // Inputs
    cout << "beta: "<<ccLWE->GetBeta().ConvertToInt() << "logQ_ccLWE: "<<logQ_ccLWE;
    vector<int> x1 = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0};
    vector<int> x2 = {23, 12, 1, 3, 4, 5, 6, 7, 8, 9, 10, 110, 1200, 13000, 140000, 1500000};
    if (x1.size() < slots) {
        vector<int> zeros(slots - x1.size(), 0);
        x1.insert(x1.end(), zeros.begin(), zeros.end());
        x2.insert(x2.end(), zeros.begin(), zeros.end());
    }

    // // Encrypt
    // vector<LWECiphertext> ctxtsLWE1(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE1[i] =
    //         ccLWE->Encrypt(lwesk, x1[i]);  // encrypted under small plantext modulus p = 4 and ciphertext modulus
    // }

    // vector<LWECiphertext> ctxtsLWE2(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE2[i] =
    //         ccLWE->Encrypt(lwesk, x1[i], FRESH,
    //                        pLWE1);  // encrypted under larger plaintext modulus p = 16 but small ciphertext modulus
    // }

    // vector<LWECiphertext> ctxtsLWE3(slots);
    // for (uint32_t i = 0; i < slots; i++) {
    //     ctxtsLWE3[i] =
    //         ccLWE->Encrypt(lwesk, x2[i], FRESH, pLWE2,
    //                        modulus_LWE);  // encrypted under larger plaintext modulus and large ciphertext modulus
    // }

    vector<LWECiphertext> ctxtsLWE4(slots);
    for (uint32_t i = 0; i < slots; i++) {
        ctxtsLWE4[i] =
            ccLWE->Encrypt(lwesk, x2[i], FRESH, pLWE3,
                           modulus_LWE);  // encrypted under large plaintext modulus and large ciphertext modulus
    }

    LWEPlaintext result;
    ccLWE->GetLWEScheme()->EvalAddEq(ctxtsLWE4[0],ctxtsLWE4[1]);
    // auto sag = ccLWE->EvalBinGate(CMUX,);
    ccLWE->Decrypt(lwesk,ctxtsLWE4[0],&result,pLWE3);
    cout<<"\naddition: "<<result<<"\n";

    // // Step 5. Perform the scheme switching
    // auto cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE1, slots, slots);

    // cout << "\n---Input x1: " << x1 << " encrypted under p = " << 4 << " and Q = " << ctxtsLWE1[0]->GetModulus()
    //           << "---" << endl;

    // // Step 6. Decrypt
    // Plaintext plaintextDec;
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // cout << "Switched CKKS decryption 1: " << plaintextDec << endl;

    // // Step 5'. Perform the scheme switching
    // cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE2, slots, slots, pLWE1, 0, pLWE1);

    // cout << "\n---Input x1: " << x1 << " encrypted under p = " << NativeInteger(pLWE1)
    //           << " and Q = " << ctxtsLWE2[0]->GetModulus() << "---" << endl;

    // // Step 6'. Decrypt
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // cout << "Switched CKKS decryption 2: " << plaintextDec << endl;

    // // Step 5''. Perform the scheme switching
    // cTemp = cc->EvalFHEWtoCKKS(ctxtsLWE3, slots, slots, pLWE2, 0, pLWE2);

    // cout << "\n---Input x2: " << x2 << " encrypted under p = " << pLWE2
    //           << " and Q = " << ctxtsLWE3[0]->GetModulus() << "---" << endl;

    // // Step 6''. Decrypt
    // cc->Decrypt(keys.secretKey, cTemp, &plaintextDec);
    // plaintextDec->SetLength(slots);
    // cout << "Switched CKKS decryption 3: " << plaintextDec << endl;

    // Step 5'''. Perform the scheme switching

    // setprecision(logQ_ccLWE +50);
    auto cTemp2 = cc->EvalFHEWtoCKKS(ctxtsLWE4, slots, slots, pLWE3, 0, pLWE3);

    cout << "\n---Input x2: " << x2 << " encrypted under p = " << NativeInteger(pLWE3)
              << " and Q = " << ctxtsLWE4[0]->GetModulus() << "---" << endl;

    // Step 6'''. Decrypt
    Plaintext plaintextDec2;
    cc->Decrypt(keys.secretKey, cTemp2, &plaintextDec2);
    plaintextDec2->SetLength(slots);
    cout << "Switched CKKS decryption 4: " << plaintextDec2 << endl;
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

    cout << "Generating the bootstrapping keys..." << endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    cout << "Completed the key generation." << endl;

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
    cout<<"result: "<<result<<"\n";

}

void decomp() {
    // Sample Program: Step 1: Set CryptoContext

    auto cc = BinFHEContext();

    // Set the ciphertext modulus to be 1 << 23
    // Note that normally we do not use this way to obtain the input ciphertext.
    // Instead, we assume that an LWE ciphertext with large ciphertext
    // modulus is already provided (e.g., by extracting from a CKKS ciphertext).
    // However, we do not provide such a step in this example.
    // Therefore, we use a brute force way to create a large LWE ciphertext.
    uint32_t logQ = 23;
    cc.GenerateBinFHEContext(TOY, false, logQ, 0, GINX, false);

    uint32_t Q = 1 << logQ;

    int q      = 4096;                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    uint64_t P = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space

    // Sample Program: Step 2: Key Generation
    // Generate the secret key
    auto sk = cc.KeyGen(); 

    cout << "Generating the bootstrapping keys..." << endl;

    // Generate the bootstrapping keys (refresh and switching keys)
    cc.BTKeyGen(sk);

    cout << "Completed the key generation." << endl;

    // Sample Program: Step 3: Encryption
    auto value = 23;
    auto ct1 = cc.Encrypt(sk, value, FRESH, P, Q);
    cout << "Encrypted value: " << value << endl;

    // Sample Program: Step 4: Evaluation
    // Decompose the large ciphertext into small ciphertexts that fit in q

    auto mod = ct1->GetptModulus();
    cout << "mod: " << mod << endl;
    auto decomp = cc.EvalDecomp(ct1);
    // decomp.
    cout << decomp.size() << " decomposed ciphertexts" << endl;
    // Sample Program: Step 5: Decryption
    uint64_t p = cc.GetMaxPlaintextSpace().ConvertToInt();
    cout << "Decomposed value: " << p << endl;


    auto ctxt1 = decomp[0];

    auto c = cc.EvalSign(ct1);
    auto cmod = c->GetptModulus();
    cout << "cmod: " << cmod << endl;
    LWEPlaintext result1;
    cc.Decrypt(sk, c, &result1,2);
    cout << "Result of encrypted computation of Sign: " << result1 << endl;

    auto pmod = decomp[1]->GetptModulus();
    cout << "pmod: " << pmod << endl;
    auto ctxt2 = decomp[1];
    auto or1 = cc.EvalBinGate(OR,decomp[0],decomp[1]);
    decomp[3]->SetModulus(4096);
    auto mod1 = decomp[3]->GetModulus();
    cout << "mod1: " << mod1 << endl;
    auto mod2 = decomp[2]->GetModulus();
    cout << "mod2: " << mod2 << endl;
    auto or2 = cc.EvalBinGate(OR,decomp[2],decomp[3]);
    auto or3 = cc.EvalBinGate(OR,or1,or2);

    LWEPlaintext result;
    cc.Decrypt(sk,or3,&result,4);
    cout<<"result: "<<result<<"\n";

    for (size_t i = 0; i < decomp.size(); i++) {
        ct1 = decomp[i];
        LWEPlaintext result;
        if (i == decomp.size() - 1) {
            // after every evalfloor, the least significant digit is dropped so the last modulus is computed as log p = (log P) mod (log GetMaxPlaintextSpace)
            auto logp = GetMSB(P - 1) % GetMSB(p - 1);
            p         = 1 << logp;
        }
        cc.Decrypt(sk, ct1, &result, p);
        cout << "(" << result << " * " << cc.GetMaxPlaintextSpace() << "^" << i << ")";
        if (i != decomp.size() - 1) {
            cout << " + ";
        }
    }
    cout << endl;
}

Ciphertext<DCRTPoly> SWITCHCKKSRNS::EvalCompareSchemeSwitching(ConstCiphertext<DCRTPoly> ciphertext1,
                                                               ConstCiphertext<DCRTPoly> ciphertext2, uint32_t numCtxts,
                                                               uint32_t numSlots, uint32_t pLWE, double scaleSign,
                                                               bool unit) {
    auto ccCKKS             = ciphertext1->GetCryptoContext();
    const auto cryptoParams = dynamic_pointer_cast<CryptoParametersCKKSRNS>(ccCKKS->GetCryptoParameters());

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

    vector<LWECiphertext> cSigns(LWECiphertexts.size());
#pragma omp parallel for
    for (uint32_t i = 0; i < LWECiphertexts.size(); i++) {
        cSigns[i] = m_ccLWE->EvalSign(LWECiphertexts[i], true);
    }

    return EvalFHEWtoCKKS(cSigns, numCtxts, numSlots, 4, -1.0, 1.0, 0);
}

int main() {
    // SwitchFHEWtoCKKS();
    decomp();
    // test();
    // sag();
    return 0;
}