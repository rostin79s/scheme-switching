#include "test_plain.hpp"
#define N 40
#include <vector>
double rostin(double a[N], double b[N], double x){
    // double *c = new double[N];
    double c[N];
    for (int i = 0; i < N-20; i++){
        c[i] = (a[i] - b[i]);
        c[i] = c[i]*x;
    }
    // return c;
    double count = 0;
    for (int i = 0; i < N-20; i++){
        double temp = 0;
        if (c[i] == x){
            temp = 1;
        }
        count += temp;
    }

    return count;
}

// vector<double> rostin(double a, double b){
//     double x = 0;
//     for (int i = 0; i < N-2; i++){
//         x += a + b;
//     }
//     return x;
// }