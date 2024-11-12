#include "../src/backend/fhe_operations.hpp"
#include <vector>
#include <iostream>
#include <chrono>

#include "test.hpp"
#include "test_plain.hpp"

using namespace CKKS;
using namespace CGGI;

int main(){
    double a[N] = {22, 3.0, 4.0, 5.0, 6.0};  // Sample input array a
    double b[N] = {1.0, 2.0, 3.0, 4.0, 5.0};  // Sample input array b
    double x = 3;  // Sample scalar input x

    // Call rostin with plaintext values
    double res_plain = rostin(a, b, x);
    std::cout << "Result: " << res_plain << std::endl;

    CKKS_scheme ck(15,50,16);
    CGGI_scheme cg(ck.getContext());
    FHEcontext* ctx = new FHEcontext(ck.getContext(), cg.getContext());

    std::cout << "Context created" << std::endl;

    std::vector<double> veca(a, a + N);
    std::vector<double> vecb(b, b + N);
    std::vector<double> vecx(N,x);
	FHEdouble arg0 = FHEencrypt(ctx,veca);
	FHEdouble arg1 = FHEencrypt(ctx,vecb);
    FHEdouble arg2 = FHEencrypt(ctx,vecx);

    std::cout << "Starting computation" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    FHEdouble result = rostin(ctx, arg0, arg1, arg2);

    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;

    std::cout << "Execution time: " << duration.count() << " seconds" << std::endl;

    FHEplainf res = FHEdecrypt(ctx,result);
    std::cout << "Result: " << res.getPlaintext() << std::endl;
    return 0;
}