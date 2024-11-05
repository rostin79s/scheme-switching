#include "../src/backend/fhe_operations.hpp"
#include "../src/backend/fhe_types.hpp"
#include <vector>
#include <iostream>

using namespace CKKS;
using namespace CGGI;

FHEdouble rostin(FHEcontext* ck, FHEdouble arg0, FHEdouble arg1) {
    FHEdouble var0 = FHEaddf(ck,arg0, arg1);
    FHEdouble var1 = FHEaddf(ck,var0, arg0);
    FHEdouble var2 = FHEsubf(ck,var1, arg1);
    FHEdouble var3 = FHEmulf(ck,var2, var2);
    FHEdouble var4 = FHEmulf(ck,arg1, var3);
    return var4;
}

int main() {
	CKKS_scheme ck(2,50,8);
    CGGI_scheme cg(ck.getContext());
    FHEcontext* ctx = new FHEcontext(ck.getContext(), cg.getContext());
	FHEdouble arg0 = FHEencrypt(ctx, 2); 
	FHEdouble arg1 = FHEencrypt(ctx,3);
    FHEdouble result = rostin(ctx, arg0, arg1);
    FHEplainf res = FHEdecrypt(ctx,result);
    std::cout << "Result: " << res.getPlaintext() << std::endl;
    return 0;
}
