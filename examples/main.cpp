#include "binfhe-constants.h"
#include "math/hal/nativeintbackend.h"
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

void SwitchCKKSToFHEW() {

    std::cout << "\n-----SwitchCKKSToFHEW-----\n" << std::endl;

    // Step 1: Setup CryptoContext for CKKS

    // Specify main parameters
    uint32_t multDepth    = 3;
    uint32_t firstModSize = 60;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 4096;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 26;
    // uint32_t slots        = ringDim / 2;  // Uncomment for fully-packed
    uint32_t slots     = 16;  // sparsely-packed
    uint32_t batchSize = slots;

    CCParams<CryptoContextCKKSRNS> parameters;

    parameters.SetMultiplicativeDepth(multDepth);
    parameters.SetFirstModSize(firstModSize);
    parameters.SetScalingModSize(scaleModSize);
    parameters.SetScalingTechnique(FLEXIBLEAUTOEXT);
    parameters.SetSecurityLevel(sl);
    parameters.SetRingDim(ringDim);
    parameters.SetBatchSize(batchSize);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);

    // Enable the features that you wish to use
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
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
    // params.SetArbitraryFunctionEvaluation(true);
    // params.GetArbitraryFunctionEvaluation();
    auto privateKeyFHEW = cc->EvalCKKStoFHEWSetup(params);


    auto ccLWE = std::make_shared<lbcrypto::BinFHEContext>();
    auto logQ = 27;
    ccLWE->GenerateBinFHEContext(TOY, true, logQ,GINX);
    cc->SetBinCCForSchemeSwitch(ccLWE);
    // auto ccLWE          = cc->GetBinCCForSchemeSwitch();
    cc->EvalCKKStoFHEWKeyGen(keys, privateKeyFHEW);
    

    std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    std::cout << ", logQ " << logQ_ccLWE;
    std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Compute the scaling factor to decrypt correctly in FHEW; under the hood, the LWE mod switch will performed on the ciphertext at the last level
    auto pLWE1       = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    std::cout << "pLWE1: " << pLWE1 << std::endl;
    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2       = modulus_LWE / (2 * beta);  // Large precision
    std::cout << "pLWE2: " << pLWE2 << std::endl;
    double scale1 = 1.0 / pLWE1;
    double scale2 = 1.0 / pLWE2;

    // Perform the precomputation for switching
    cc->EvalCKKStoFHEWPrecompute(scale1);
    

    // Step 3: Encoding and encryption of inputs

    // Inputs
    std::vector<double> x2  = {5,6};

    uint32_t encodedLength2 = x2.size();

    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, 0, nullptr);

    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    cc->EvalCKKStoFHEWPrecompute(scale2);

    // Transform the ciphertext from CKKS to FHEW (only for the number of inputs given)
    auto cTemp2 = cc->EvalCKKStoFHEW(c2, encodedLength2);

    std::cout << "\n---Decrypting switched ciphertext with large precision (plaintext modulus " << NativeInteger(pLWE2)
              << ")---\n"
              << std::endl;

    ptxt2->SetLength(encodedLength2);
    LWEPlaintext result;
    std::cout << "Input x2: " << ptxt2->GetRealPackedValue() << std::endl;
    std::cout << "FHEW decryption: ";
    for (uint32_t i = 0; i < cTemp2.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, cTemp2[i], &result, pLWE2);
        std::cout << result << " ";
    }
    std::cout << "\n" << std::endl;


    // Generate the bootstrapping keys (refresh and switching keys)
    ccLWE->BTKeyGen(privateKeyFHEW);


    auto modtemp = cTemp2[0]->GetModulus();
    std::cout << "modtemp: " << modtemp << std::endl;
    auto qtemp = ccLWE->GetParams()->GetLWEParams()->Getq();
    std::cout << "qtemp: " << qtemp << std::endl;

    auto dec1 = ccLWE->EvalDecomp(cTemp2[0]);
    auto dec2 = ccLWE->EvalDecomp(cTemp2[1]);




    auto saglength = dec1[0]->GetLength();
    auto qmod = dec1[0]->GetModulus();
    auto pmod = dec1[0]->GetptModulus();
    auto N = ccLWE->GetParams()->GetLWEParams()->GetN();
    std::cout << "N: " << N << std::endl;   
    std::cout << "pmod: " << pmod << std::endl;
    std::cout << "qmod: " << qmod << std::endl;
    std::cout << "saglengths: " << saglength << std::endl;


    int p = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space
    std::cout << "p: " << p << std::endl;


    



    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        auto l = (m>>1)%2;
        auto r = (m)%2;
        return NativeInteger(l.ConvertToInt() & r.ConvertToInt());
    };

    auto d1 = dec1[0];
    auto d2 = dec2[0];
    ccLWE->GetLWEScheme()->EvalMultConstEq(d1,2);
    ccLWE->GetLWEScheme()->EvalAddEq(d1,d2);

    


    auto lut = ccLWE->GenerateLUTviaFunction(fp, p);
    auto ctfunc1 = ccLWE->EvalFunc(d1, lut);

    LWEPlaintext resultDecomp;
    std::cout << "decryption eval func: ";
    ccLWE->Decrypt(privateKeyFHEW, ctfunc1, &resultDecomp, p);
    std::cout << resultDecomp << std::endl;


    

    for (uint32_t j = 0; j < cTemp2.size(); j++) {
        // Decompose the large ciphertext into small ciphertexts that fit in q
        auto decomp = ccLWE->EvalDecomp(cTemp2[j]);

        // Decryption
        auto p = ccLWE->GetMaxPlaintextSpace().ConvertToInt();
        std::cout << "p: " << p << std::endl;
        LWECiphertext ct;
        for (size_t i = 0; i < decomp.size(); i++) {
            ct = decomp[i];
            LWEPlaintext resultDecomp;
            // The last digit should be up to P / p^floor(log_p(P))
            if (i == decomp.size() - 1) {
                p = pLWE2 / std::pow(static_cast<double>(pLWE1), std::floor(std::log(pLWE2) / std::log(pLWE1)));
                std::cout << "\np last digit: " << p << std::endl;
            }
            ccLWE->Decrypt(privateKeyFHEW, ct, &resultDecomp, p);
            std::cout << "(" << resultDecomp << " * " << NativeInteger(pLWE1) << "^" << i << ")";
            if (i != decomp.size() - 1) {
                std::cout << " + ";
            }
        }
        std::cout << std::endl;
    }
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
    auto logQ = 27;
    cc.GenerateBinFHEContext(TOY, true, logQ,GINX);

    auto ccq = cc.GetParams()->GetLWEParams()->Getq();
    std::cout << "ccq: " << ccq << std::endl;
    auto N = cc.GetParams()->GetLWEParams()->GetN();
    std::cout << "N: " << N << std::endl;
    // Sample Program: Step 2: Key Generation
    uint32_t Q = 1 << logQ;

    int q      = ccq.ConvertToInt();                                               // q
    int factor = 1 << int(logQ - log2(q));                           // Q/q
    uint64_t P = cc.GetMaxPlaintextSpace().ConvertToInt() * factor;  // Obtain the maximum plaintext space

    std::cout << "P: " << P << std::endl;
    // Generate the secret key
    auto sk = cc.KeyGen();

    std::cout << "Generating the bootstrapping keys..." << std::endl;

    // Generate the bootstrapping keys (refresh, switching and public keys)
    cc.BTKeyGen(sk, PUB_ENCRYPT);

    auto pk = cc.GetPublicKey();

    std::cout << "Completed the key generation." << std::endl;

    auto num = 3;
    auto ct1 = cc.Encrypt(sk, num, LARGE_DIM, P, Q);
    std::cout << "Encrypted value: " << num<< std::endl;

    // Sample Program: Step 4: Evaluation
    // Decompose the large ciphertext into small ciphertexts that fit in q
    auto decomp = cc.EvalDecomp(ct1);
    // Sample Program: Step 5: Decryption
    uint64_t p = cc.GetMaxPlaintextSpace().ConvertToInt();
    std::cout << "p: " << p << std::endl;
    std::cout << "Decomposed value: ";
    for (size_t i = 0; i < decomp.size(); i++) {
        ct1 = decomp[i];
        LWEPlaintext result;
        if (i == decomp.size() - 1) {
            // after every evalfloor, the least significant digit is dropped so the last modulus is computed as log p = (log P) mod (log GetMaxPlaintextSpace)
            auto logp = GetMSB(P - 1) % GetMSB(p - 1);
            p         = 1 << logp;
        }
        cc.Decrypt(sk, ct1, &result, p);
        std::cout << "(" << result << " * " << cc.GetMaxPlaintextSpace() << "^" << i << ")";
        if (i != decomp.size() - 1) {
            std::cout << " + ";
        }
    }
    std::cout << std::endl;


    // Sample Program: Step 3: Create the to-be-evaluated funciton and obtain its corresponding LUT
    int psag = cc.GetMaxPlaintextSpace().ConvertToInt();  // Obtain the maximum plaintext space
    std::cout << "psag: " << psag << std::endl;
    // Initialize Function f(x) = x^3 % p
    auto fp = [](NativeInteger m, NativeInteger p1) -> NativeInteger {
        return (m >> 2)%2;
    };

    // Generate LUT from function f(x)
    p = cc.GetMaxPlaintextSpace().ConvertToInt();
    auto lut = cc.GenerateLUTviaFunction(fp, p);
    std::cout << "Evaluate x^3%" << p << "." << std::endl;

    auto d1 = decomp[0];

    auto dmod = d1->GetModulus();
    std::cout << "dmod: " << dmod << std::endl;
    auto pmod = d1->GetptModulus();
    std::cout << "pmod: " << pmod << std::endl;

    auto dres = cc.EvalFunc(d1, lut);
    LWEPlaintext resd1;
    cc.Decrypt(sk, dres, &resd1, p);
    std::cout << "Decrypted result: " << resd1 << std::endl;

    // // Sample Program: Step 4: evalute f(x) homomorphically and decrypt
    // // Note that we check for all the possible plaintexts.
    // for (int i = 0; i < p; i++) {
    //     auto ct1 = cc.Encrypt(pk, i % p, SMALL_DIM, p);

    //     auto ct_cube = cc.EvalFunc(ct1, lut);

    //     LWEPlaintext result;

    //     cc.Decrypt(sk, ct_cube, &result, p);

    //     std::cout << "Input: " << i << ". Expected: " << fp(i, p) << ". Evaluated = " << result << std::endl;
    // }
}



int main() {
    SwitchCKKSToFHEW();
    // decomp();
    // test();
    // sag();
    return 0;
}