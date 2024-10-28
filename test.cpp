// #include <cstdio>

double rostin(double a, double b){
    for (int i = 0; i < 10; i++){
        a = a + b;
        double temp = 0;
        if (a == b){
            temp = 1;
        }
        b += temp;
    }
    return b;
}