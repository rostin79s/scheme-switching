#include "../src/backend/fhe_operations.hpp"
#include <vector>
#include <iostream>

#include "test.hpp"
#include "test_plain.hpp"

using namespace CKKS;
using namespace CGGI;

int main(){
    double a = 2;
    double b = 2;
    double res_plain = rostin(a, b);
    std::cout << "Result: " << res_plain << std::endl;

    CKKS_scheme ck(40,50,1);
    CGGI_scheme cg(ck.getContext());
    FHEcontext* ctx = new FHEcontext(ck.getContext(), cg.getContext());
	FHEdouble arg0 = FHEencrypt(ctx,a); 
	FHEdouble arg1 = FHEencrypt(ctx,b);
    FHEdouble result = rostin(ctx, arg0, arg1);
    FHEplainf res = FHEdecrypt(ctx,result);
    std::cout << "Result: " << res.getPlaintext() << std::endl;
    return 0;
}