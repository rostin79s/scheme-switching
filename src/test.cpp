#include <cstdio>

int rostin(int a, int b){
    int x = a*b;
    int res = x*b;
    return res;
}

int main(){
    int a = 23;
    int b = 2;
    int res = rostin(a,b);
    printf("%d\n", res);
}
