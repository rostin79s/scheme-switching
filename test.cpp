// #include <cstdio>

double rostin(double a, double b){
    double temp = 0;
    for (int i = 0; i < 10; i++){
        a = a + b;
        if (a == b){
            temp = 1;
        }
        b += temp;
    }
    return b;
}