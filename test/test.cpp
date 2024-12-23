#include "test_plain.hpp"

// double* rostin(double a[10], double b[10]){
//     double* c = new double[10];
//     for (int i = 0; i < 10; ++i) {
//         c[i] = a[i] + b[i];
//     }
//     return c;
// }


// int min(int a) {
//     return -1;
// }

void min_index(int arr[10], int result[10]) {
    // Step 1: Find the minimum value
    // int min_value = min(arr[0]);
    int min_value = arr[0];
    for (int i = 1; i < 10; ++i) {
        if (arr[i] < min_value) {
            min_value = arr[i];
        }
    }

    // Step 2: Mark indices where the value equals the minimum
    for (int i = 0; i < 10; ++i) {
        result[i] = (arr[i] == min_value) ? 1 : 0;
    }
}