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
    /*
  Example of switching a packed ciphertext from CKKS to multiple FHEW ciphertexts.
 */

    std::cout << "\n-----SwitchCKKSToFHEW-----\n" << std::endl;

    // Step 1: Setup CryptoContext for CKKS

    // Specify main parameters
    uint32_t multDepth    = 3;
    uint32_t firstModSize = 60;
    uint32_t scaleModSize = 50;
    uint32_t ringDim      = 4096;
    SecurityLevel sl      = HEStd_NotSet;
    BINFHE_PARAMSET slBin = TOY;
    uint32_t logQ_ccLWE   = 25;
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
    auto privateKeyFHEW = cc->EvalCKKStoFHEWSetup(params);
    auto ccLWE          = cc->GetBinCCForSchemeSwitch();
    cc->EvalCKKStoFHEWKeyGen(keys, privateKeyFHEW);

    std::cout << "FHEW scheme is using lattice parameter " << ccLWE->GetParams()->GetLWEParams()->Getn();
    std::cout << ", logQ " << logQ_ccLWE;
    std::cout << ", and modulus q " << ccLWE->GetParams()->GetLWEParams()->Getq() << std::endl << std::endl;

    // Compute the scaling factor to decrypt correctly in FHEW; under the hood, the LWE mod switch will performed on the ciphertext at the last level
    auto pLWE1       = ccLWE->GetMaxPlaintextSpace().ConvertToInt();  // Small precision
    auto modulus_LWE = 1 << logQ_ccLWE;
    auto beta        = ccLWE->GetBeta().ConvertToInt();
    auto pLWE2       = modulus_LWE / (2 * beta);  // Large precision

    double scale1 = 1.0 / pLWE1;
    double scale2 = 1.0 / pLWE2;

    // Perform the precomputation for switching
    cc->EvalCKKStoFHEWPrecompute(scale1);

    // Step 3: Encoding and encryption of inputs

    // Inputs
    std::vector<double> x1  = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> x2  = {3, 271.0, 30000.0, static_cast<double>(pLWE2) - 2};
    uint32_t encodedLength1 = x1.size();
    uint32_t encodedLength2 = x2.size();

    // Encoding as plaintexts
    Plaintext ptxt1 = cc->MakeCKKSPackedPlaintext(x1, 1, 0, nullptr);
    Plaintext ptxt2 = cc->MakeCKKSPackedPlaintext(x2, 1, 0, nullptr);

    // Encrypt the encoded vectors
    auto c1 = cc->Encrypt(keys.publicKey, ptxt1);
    auto c2 = cc->Encrypt(keys.publicKey, ptxt2);

    // Step 4: Scheme switching from CKKS to FHEW

    // A: First scheme switching case

    // Transform the ciphertext from CKKS to FHEW
    auto cTemp = cc->EvalCKKStoFHEW(c1, encodedLength1);

    std::cout << "\n---Decrypting switched ciphertext with small precision (plaintext modulus " << NativeInteger(pLWE1)
              << ")---\n"
              << std::endl;

    std::vector<int32_t> x1Int(encodedLength1);
    std::transform(x1.begin(), x1.end(), x1Int.begin(), [&](const double& elem) {
        return static_cast<int32_t>(static_cast<int32_t>(std::round(elem)) % pLWE1);
    });
    ptxt1->SetLength(encodedLength1);
    std::cout << "Input x1: " << ptxt1->GetRealPackedValue() << "; which rounds to: " << x1Int << std::endl;
    std::cout << "FHEW decryption: ";
    LWEPlaintext result;
    for (uint32_t i = 0; i < cTemp.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, cTemp[i], &result, pLWE1);
        std::cout << result << " ";
    }
    std::cout << "\n" << std::endl;

    // B: Second scheme switching case

    // Perform the precomputation for switching
    cc->EvalCKKStoFHEWPrecompute(scale2);

    // Transform the ciphertext from CKKS to FHEW (only for the number of inputs given)
    auto cTemp2 = cc->EvalCKKStoFHEW(c2, encodedLength2);

    std::cout << "\n---Decrypting switched ciphertext with large precision (plaintext modulus " << NativeInteger(pLWE2)
              << ")---\n"
              << std::endl;

    ptxt2->SetLength(encodedLength2);
    std::cout << "Input x2: " << ptxt2->GetRealPackedValue() << std::endl;
    std::cout << "FHEW decryption: ";
    for (uint32_t i = 0; i < cTemp2.size(); ++i) {
        ccLWE->Decrypt(privateKeyFHEW, cTemp2[i], &result, pLWE2);
        std::cout << result << " ";
    }
    std::cout << "\n" << std::endl;

    // C: Decompose the FHEW ciphertexts in smaller digits
    std::cout << "Decomposed values for digit size of " << NativeInteger(pLWE1) << ": " << std::endl;
    // Generate the bootstrapping keys (refresh and switching keys)
    ccLWE->BTKeyGen(privateKeyFHEW);


    // ccLWE->GetLWEScheme()->EvalAddEq(cTemp2[1],cTemp2[3]);
    // LWEPlaintext resultadd;
    // ccLWE->Decrypt(privateKeyFHEW, cTemp2[1], &resultadd, pLWE2);
    // std::cout << "addition: " << resultadd << std::endl;



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
        auto im = m.ConvertToInt();
        auto r = im & 0b0011;
        auto l = im & 0b1100;
        r = r & (r/2);
        l = l & (l/2);


        // auto lhs = (m/p1) % p1;
        // auto rhs= (m%p1) % p1;
        // auto l = lhs.ConvertToInt();
        // auto r = rhs.ConvertToInt();
        auto res = r;
        NativeInteger result(res);
        return result;
    };
    auto fshift = [](NativeInteger m, NativeInteger p2) -> NativeInteger {
        auto im = m.ConvertToInt();
        auto res = im & 0b0001;
        return NativeInteger(res);
    };
    auto lutshift = ccLWE->GenerateLUTviaFunction(fshift, p);
    auto ctxshift1 = ccLWE->EvalFunc(dec1[0],lutshift);
    auto ctxshift2 = ccLWE->EvalFunc(dec2[0],lutshift);

    ccLWE->GetLWEScheme()->EvalMultConstEq(ctxshift1,2);
    ccLWE->GetLWEScheme()->EvalAddEq(ctxshift1,ctxshift2);

    auto lutand = ccLWE->GenerateLUTviaFunction(fp,p);
    auto ctand1 = ccLWE->EvalFunc(ctxshift1, lutand);
    LWEPlaintext resultand;
    ccLWE->Decrypt(privateKeyFHEW, ctand1, &resultand, p);
    std::cout << "and: " << resultand << std::endl;

    LWEPlaintext resultshift;
    ccLWE->Decrypt(privateKeyFHEW, ctxshift1, &resultshift, p);
    std::cout << "shift: " << resultshift << std::endl;






    
    // Generate LUT from function f(x)
    ccLWE->GetLWEScheme()->EvalMultConstEq(dec1[0], 16);
    ccLWE->GetLWEScheme()->EvalAddEq(dec1[0],dec2[0]);

    LWEPlaintext resultadd;
    ccLWE->Decrypt(privateKeyFHEW, dec1[0], &resultadd, p);
    std::cout << "addition: " << resultadd << std::endl;

    auto lut = ccLWE->GenerateLUTviaFunction(fp, p);
    auto ctfunc1 = ccLWE->EvalFunc(dec1[0], lut);
    // auto ctfunc2 = ccLWE->EvalFunc(dec2[0], lut);

    // auto ctfunc3 = ccLWE->EvalBinGate(AND,ctfunc1,ctfunc2);

    // ccLWE->GetLWEScheme()->EvalAddEq(ctfunc1,ctfunc2);
    // ccLWE->GetLWEScheme()->EvalAddEq(ctfunc1,ctfunc2);

    LWEPlaintext resultDecomp;
    // auto p = ccLWE->GetMaxPlaintextSpace().ConvertToInt();
    std::cout << "decryption: ";
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



int main() {
    SwitchCKKSToFHEW();
    // test();
    // sag();
    return 0;
}