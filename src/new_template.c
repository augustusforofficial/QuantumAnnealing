#include "./annealing.h"

int main(){
    int i,j,k;
    /*定数宣言*/
    int ni[N] = {2,3,5};
    double J[N][N] = {0.0};
    double H[Nums] = {0.0};
    double complex f1[Nums] = {0.0 + 0.0 * I};

    double B0 = 1.0;
    int Time = 100000;
    double tau = 1.0;

    /*J[i][j]の定義.これは問題によって定義する*/
    for(i=0;i<N;i++){
        for(j=i+1;j<N;j++){
            if(i!=j) J[i][j] =  ni[i] * ni[j];
        }
    }
    
    embed_diagonal_H(H, J);
    Show_vector_Nums(H);

    /*時間発展*/
    time_evolution(f1,J,Time,B0,tau);

    /*正規化.時間発展中で毎回行うのが実際だが、計算上は最後にまとめて行っても良い。*/
    normalize(f1);

    /*最終出力*/
    print_state(f1);
}
