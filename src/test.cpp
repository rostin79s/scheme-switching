#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext

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
    auto ct1 = cc.Encrypt(sk, 0);
    auto ct2 = cc.Encrypt(sk, 0);

    // Sample Program: Step 4: Evaluation

    // Compute (1 AND 1) = 1; Other binary gate options are OR, NAND, and NOR
    auto ctAND1 = cc.EvalBinGate(OR, ct1, ct2);

    // Compute (NOT 1) = 0
    // auto ct2Not = cc.EvalNOT(ct2);

    // // Compute (1 AND (NOT 1)) = 0
    // auto ctAND2 = cc.EvalBinGate(AND, ct2Not, ct1);

    // // Computes OR of the results in ctAND1 and ctAND2 = 1
    // auto ctResult = cc.EvalBinGate(OR, ctAND1, ctAND2);

    // Sample Program: Step 5: Decryption

    LWEPlaintext result;

    cc.Decrypt(sk, ctAND1, &result);

    std::cout << "Result of encrypted computation is: " << result << std::endl;

    return 0;
}