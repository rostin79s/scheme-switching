#include "src/backend/fhe_operations.hpp"
#include "src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>

using namespace CKKS;
using namespace CGGI;

FHEdouble* rostin(CKKS_scheme& ck, FHEdouble* arg0, FHEdouble* arg1) {
    FHEdouble* var0 = ck.FHEaddf(arg0, arg1);
    FHEdouble* var1 = ck.FHEaddf(var0, arg0);
    FHEdouble* var2 = ck.FHEsubf(var1, arg1);
    FHEdouble* var3 = ck.FHEmulf(var2, var2);
    FHEdouble* var4 = ck.FHEmulf(arg1, var3);
    return var4;
}

int main() {
	CKKS_scheme ck(2,50,1);
	std::vector<double> input1 = {2};
	FHEdouble* arg0 = ck.FHEencrypt(ck.FHEencode(input1));
	std::vector<double> input2 = {3};
	FHEdouble* arg1 = ck.FHEencrypt(ck.FHEencode(input2));
    FHEdouble* result = rostin(ck, arg0, arg1);
    FHEplain* res = ck.FHEdecrypt(result);
    std::cout << "Result: " << res->getPlaintext() << std::endl;
    return 0;
}
