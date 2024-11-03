#include "test_plain.hpp"
#define N 40
// double* rostin(double a[N], double b[N]){
//     double* c = new double[N];
//     for (int i = 0; i < N-2; i++){
//         c[i] = a[i] + b[i+2];
//     }
//     return c;
// }

double rostin(double a, double b){
    double x = 0;
    for (int i = 0; i < N-2; i++){
        x += a + b;
    }
    return x;
}