#include <cstdio>

int rostin(int a, int b){
    int y = a*a;
    int z = a+2;
    int res = y-z;
    return res;
}

int main(){
    int a = 32;
    int b = 21;
    int res = rostin(a,b);
    printf("%d\n", res);
}
