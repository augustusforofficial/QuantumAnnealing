#include "annealing.h"

int main(){
    printf("test\n");
    int i,j,k;
    /*定数宣言*/
    int ni[N] = {2,3,5,10};
    double J[Nums][Nums] = {0.0};
    double H[Nums] = {0.0};
    double complex f1[Nums] = {0.0 + 0.0 * I};

    double B0 = 1.0;
    int Time = 10000;
    double tau = 1.0;

    /*J[i][j]の定義.これは問題によって定義する*/
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            if(i!=j) J[i][j] = -1 * ni[i] * ni[j];
        }
    }
   
}
