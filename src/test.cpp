#include <cmath>

int rostin(int a, int b){
    int y = a*a;
    int z = a+2;
    int res = y-z;
    int tmp = abs(a);
    return res + tmp;
}
