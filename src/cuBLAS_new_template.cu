#include "cuBLAS_annealing.h"

int main(){
    int i,j;
    /*定数宣言*/
    int ni[N] = {1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31};
    double J[N][N] = {0.0};
    cuDoubleComplex *f1 = (cuDoubleComplex *)malloc(Nums * sizeof(cuDoubleComplex));

    double B0 = 1.0;
    int Time = 10000;
    double tau = 1.0;

    /*J[i][j]の定義.これは問題によって定義する*/
    for(i=0;i<N;i++){
        for(j=0;j<N;j++){
            if(i!=j) J[i][j] = -1 * ni[i] * ni[j];
        }
    }
    
    /*時間発展*/
    time_evolution_cu(f1, J, Time, B0, tau);

    /*最終出力 - convert to double complex for printing*/
    cuDoubleComplex *psi = (cuDoubleComplex *)malloc(Nums * sizeof(cuDoubleComplex));
    for(int i=0; i<Nums; i++){
        psi[i] = f1[i];
    }
    print_state(psi);
    free(psi);

    printf("Done\n");
    free(f1);
}