// #include <cstdio>

double rostin(double a, double b){
    double temp = a+b;
    double temp2 = temp + a;
    double temp3 = temp2 - b;
    double temp4 = temp3 * temp3;
    double temp5 = b * temp4;
    return temp5;
}