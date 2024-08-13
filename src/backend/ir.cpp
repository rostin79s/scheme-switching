#include "fhe_operations.hpp"

void _Z6rostinii(Ciphertext %1, Ciphertext %0) {
    %3 = FHEmul(%0, %0);
    %4 = FHEaddP(%0, 2);
    %5 = FHEand(%3, %4);
    %6 = FHEadd(%5, %1);
    return %6;
}

int main() {
    std::vector<double> inputs = {/* User inputs */};
    std::vector<Ciphertext> encryptedInputs;
    Ciphertext %1 = FHEencrypt(inputs[%1]);
    Ciphertext %0 = FHEencrypt(inputs[%0]);
    Ciphertext result = _Z6rostinii(%1, %0);
    double finalResult = FHEdecrypt(result);
    std::cout << "Result: " << finalResult << std::endl;
    return 0;
}
